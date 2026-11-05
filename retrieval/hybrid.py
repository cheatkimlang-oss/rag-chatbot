"""Stage 05a — hybrid retrieval (dense + sparse, RRF) with project grouping.

One batched Qdrant round trip runs three searches:
  1. the fused RRF query (dense prefetch + lexical prefetch) — this is the ranking;
  2. dense-only, top `prefetch_k` — gives the cosine used by the abstention gate;
  3. lexical-only, top `prefetch_k` — tells us whether the keyword path fired.
RRF scores are rank-based and not comparable across queries, so they are never
used as an abstention threshold (Architecture Spec §9.5).

Everything is configurable through `SearchConfig` (built from Settings):
  mode      hybrid (default) | dense | lexical — single-path modes exist for
            ablations ("what does the lexical path add?"); the gate always uses
            the dense similarity, so abstention behaves the same in every mode
  fusion    rrf (with configurable k) | dbsf
  hnsw_ef / exact   search-time HNSW beam width, or brute force (recall ground truth)
  distance  the collection's metric; dense scores are mapped to "higher = better"
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import models

from store.vector_schema import COLLECTION, DENSE, LEXICAL, to_similarity


class SearchUnavailable(RuntimeError):
    """Qdrant could not be reached; callers degrade instead of crashing."""


@dataclass(frozen=True)
class SearchConfig:
    prefetch_k: int = 40
    fusion_limit: int = 10
    mode: str = "hybrid"
    fusion: str = "rrf"
    rrf_k: int = 60                 # 0 = server's built-in RRF (Qdrant default k=2)
    distance: str = "cosine"
    hnsw_ef: int | None = None
    exact: bool = False

    @classmethod
    def from_settings(cls, s) -> "SearchConfig":
        return cls(prefetch_k=s.prefetch_k, fusion_limit=s.fusion_limit, mode=s.search_mode,
                   fusion=s.fusion, rrf_k=s.rrf_k, distance=s.distance,
                   hnsw_ef=s.hnsw_ef, exact=s.exact_search)

    def params(self) -> models.SearchParams | None:
        if self.hnsw_ef is None and not self.exact:
            return None
        return models.SearchParams(hnsw_ef=self.hnsw_ef, exact=self.exact)

    def fusion_query(self):
        if self.fusion == "dbsf":
            return models.FusionQuery(fusion=models.Fusion.DBSF)
        if self.rrf_k:
            return models.RrfQuery(rrf=models.Rrf(k=self.rrf_k))
        return models.FusionQuery(fusion=models.Fusion.RRF)


@dataclass
class ProjectHit:
    project_id: str
    score: float                    # best fused RRF score among this project's chunks
    payload: dict                   # payload of that best chunk
    dense_score: float | None = None    # best dense cosine for this project, if in dense top-k
    matched_by: list[str] = field(default_factory=list)   # "dense", "lexical", or both
    chunk_ids: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)      # payloads of every fused chunk, best first


def _sparse(sv) -> models.SparseVector:
    return models.SparseVector(indices=list(sv[0]), values=list(sv[1]))


def _as_similarity(points, distance: str):
    """Rewrite dense scores in place so every caller can rank / gate on 'higher = better'."""
    if distance in ("euclid", "manhattan"):
        for p in points:
            p.score = to_similarity(p.score, distance)
    return points


def hybrid_retrieve(client, embedder, query: str, lexical_query: str | None = None,
                    flt: models.Filter | None = None, *, collection: str = COLLECTION,
                    search: SearchConfig | None = None, prefetch_k: int | None = None,
                    fusion_limit: int | None = None,
                    top_k: int | None = None) -> list[ProjectHit]:
    cfg = search or SearchConfig()
    prefetch_k = prefetch_k or cfg.prefetch_k
    fusion_limit = fusion_limit or cfg.fusion_limit
    params = cfg.params()

    lexical_query = lexical_query or query
    encode = getattr(embedder, "encode_query", embedder.encode)
    dense, sparse = encode([query])
    dense_vec, sparse_vec = dense[0], sparse[0]
    if lexical_query != query:                 # e.g. Khmer, segmented for the sparse path
        sparse_vec = encode([lexical_query])[1][0]
    has_lexical = bool(sparse_vec[0])
    if cfg.mode == "lexical" and not has_lexical:
        return []                              # nothing for the keyword path to match

    dense_req = models.QueryRequest(query=dense_vec, using=DENSE, filter=flt, params=params,
                                    limit=prefetch_k, with_payload=["project_id"])
    lex_req = models.QueryRequest(query=_sparse(sparse_vec), using=LEXICAL, filter=flt,
                                  limit=prefetch_k, with_payload=["project_id"]) if has_lexical else None

    if cfg.mode == "dense":
        ranking = models.QueryRequest(query=dense_vec, using=DENSE, filter=flt, params=params,
                                      limit=fusion_limit, with_payload=True)
    elif cfg.mode == "lexical":
        ranking = models.QueryRequest(query=_sparse(sparse_vec), using=LEXICAL, filter=flt,
                                      limit=fusion_limit, with_payload=True)
    else:
        prefetch = [models.Prefetch(query=dense_vec, using=DENSE, limit=prefetch_k,
                                    filter=flt, params=params)]
        if has_lexical:
            prefetch.append(models.Prefetch(query=_sparse(sparse_vec), using=LEXICAL,
                                            limit=prefetch_k, filter=flt))
        ranking = models.QueryRequest(prefetch=prefetch, query=cfg.fusion_query(),
                                      filter=flt, limit=fusion_limit, with_payload=True)

    # The dense request always runs: it feeds the abstention gate in every mode.
    requests = [ranking, dense_req] + ([lex_req] if lex_req else [])
    try:
        responses = client.query_batch_points(collection_name=collection, requests=requests)
    except Exception as exc:                   # connection refused, timeout, missing collection
        raise SearchUnavailable(str(exc)) from exc

    ranked = responses[0].points
    if cfg.mode == "dense":                    # ranking scores are raw distances here
        _as_similarity(ranked, cfg.distance)
    dense_pts = _as_similarity(responses[1].points, cfg.distance)
    lex_pts = responses[2].points if lex_req else []
    return group_by_project(ranked, dense_pts, lex_pts, rank_cutoff=fusion_limit, top_k=top_k)


def group_by_project(fused, dense_pts=(), lex_pts=(), *, rank_cutoff: int = 10,
                     top_k: int | None = None) -> list[ProjectHit]:
    """Prevent 'one project, five times'. Group AFTER fusion, BEFORE truncation."""
    dense_best: dict[str, float] = {}
    dense_rank: dict[str, int] = {}
    for rank, p in enumerate(dense_pts, start=1):
        pid = p.payload["project_id"]
        dense_best[pid] = max(dense_best.get(pid, float("-inf")), p.score)
        dense_rank.setdefault(pid, rank)
    lex_rank: dict[str, int] = {}
    for rank, p in enumerate(lex_pts, start=1):
        lex_rank.setdefault(p.payload["project_id"], rank)

    best: dict[str, ProjectHit] = {}
    for p in fused:
        pid = p.payload["project_id"]
        hit = best.get(pid)
        if hit is None:
            hit = best[pid] = ProjectHit(project_id=pid, score=p.score, payload=p.payload,
                                         dense_score=dense_best.get(pid))
        elif p.score > hit.score:
            hit.score, hit.payload = p.score, p.payload
        hit.chunk_ids.append(p.payload.get("chunk_id", str(p.id)))
        hit.chunks.append(p.payload)          # evidence for sentence-level answer extraction

    for pid, hit in best.items():
        # "matched by" a path = that path ranked one of the project's chunks near the top,
        # not merely somewhere in a 40-deep prefetch.
        if dense_rank.get(pid, rank_cutoff + 1) <= rank_cutoff:
            hit.matched_by.append("dense")
        if lex_rank.get(pid, rank_cutoff + 1) <= rank_cutoff:
            hit.matched_by.append("lexical")

    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return ranked[:top_k] if top_k else ranked
