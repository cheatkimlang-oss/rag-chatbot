"""Runtime settings shared by the ingestion, retrieval and answer layers.

Every value can be overridden from the environment so the bot, the CLI and the
evaluator all run the exact same configuration (and every run can log it).

Three groups of retrieval knobs, by what a change costs:

    chunking      KB_CHUNK_*            re-chunks changed files on the next ingestion run
    vector index  KB_DISTANCE, KB_HNSW_M, KB_HNSW_EF_CONSTRUCT
                                        collection contract -> `ingestion.run --rebuild`
    search        KB_PREFETCH_K, KB_FUSION*, KB_RRF_K, KB_HNSW_EF, KB_EXACT_SEARCH,
                  KB_SEARCH_MODE, KB_DISPLAY_TOP_K ...   per query, no re-index

    python -m config        prints the effective configuration
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

# Provisional answer gates per scorer (and embedder, for the dense fallback). "*" = any.
# cross 2.0 — measured 2026-11-05, Xenova/ms-marco-MiniLM-L-6-v2 over the 3-document
#   corpus (fastembed retrieval), 10 answerable + 10 unanswerable questions:
#   refuses 10/10 unanswerable (PRD abstention target); serves 3 answers, 1 of them a
#   wrong sentence; also refuses 1 correct answer scoring 1.06. At 0.0: 9/10 refused,
#   3 wrong answers served. 20 questions is a tiny sample — replace with the golden-set sweep.
# cross 3.0 — re-measured 2026-11-06 with eval/run_qa.py (149 answerable + 20 unanswerable,
#   8-file corpus incl. *_facts.md). Lowest correct answers: 3.95, 4.54, 4.98; served
#   negatives: 2.97, 3.70, 4.02. 3.0 drops one wrong answer at no cost; 4.5 would drop all
#   three but also refuse "what do I value?" (3.95) — too close to call on this sample.
# dense 0.5 — NOT measured (only used when the cross-encoder is unavailable).
ANSWER_GATES = {
    "cross": {"*": 3.0},                               # MS-MARCO logits
    "dense": {"*": 0.5},                               # cosine + 0.2 * keyword overlap
}

CHUNK_UNITS = ("chars", "tokens")
DISTANCES = ("cosine", "dot", "euclid", "manhattan")
SEARCH_MODES = ("hybrid", "dense", "lexical")
FUSIONS = ("rrf", "dbsf")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # infrastructure — empty qdrant_url = Qdrant local mode in `qdrant_path`
    # (no Docker needed; fine for development, validate hybrid numbers on the server).
    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant_local"
    sqlite_path: str = "data/kb.sqlite3"
    collection: str = "rupp_senior_projects_v1"

    # sources
    corpus_dir: str = "data/corpus"
    projects_csv: str = "data/projects.csv"
    seniors_csv: str = "data/seniors.csv"

    # embeddings (pinned). "bge-m3" = the spec model (FlagEmbedding + torch, ~2 GB);
    # "fastembed" = light CPU demo stack (Spec §6 fallbacks, English only).
    embed_backend: str = "bge-m3"
    embed_model: str = "BAAI/bge-m3"
    fastembed_dense: str = "BAAI/bge-small-en-v1.5"
    fastembed_sparse: str = "Qdrant/bm25"
    # Cross-encoder that picks the answer sentence inside retrieved chunks ("" = off).
    answer_scorer: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # LLM (pinned — never "latest")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b-it-qat"
    enable_llm: bool = False

    # chunking (Stage 02, Spec §8.2). Sizes are measured in `chunk_unit`:
    # "chars", or "tokens" of the embedder's own tokenizer (must fit its max length).
    chunk_unit: str = "chars"
    chunk_size: int = 700
    chunk_overlap: int = 100
    slide_chunk_size: int = 300
    slide_chunk_overlap: int = 50
    min_chunk_chars: int = 80            # noise filter (headers, page numbers); always chars

    # vector index (collection contract, Spec §7.3) — changes need --rebuild
    distance: str = "cosine"             # cosine | dot | euclid | manhattan
    hnsw_m: int = 16                     # graph degree; 0 disables the HNSW graph
    hnsw_ef_construct: int = 200         # build-time beam width

    # retrieval knobs (Appendix D) — per query
    search_mode: str = "hybrid"          # hybrid | dense | lexical (lexical/dense = ablations)
    fusion: str = "rrf"                  # rrf | dbsf (distribution-based score fusion)
    # RRF constant. Qdrant's built-in default is k=2; the spec calls for 60.
    # 0 = send plain Fusion.RRF (server default) for servers without parametrized RRF.
    rrf_k: int = 60
    hnsw_ef: int | None = None           # search-time beam width; None = Qdrant default
    exact_search: bool = False           # brute force: the ground truth for HNSW recall
    prefetch_k: int = 40
    fusion_limit: int = 10
    display_top_k: int = 5
    weak_match_count: int = 3
    # Compared against the best project's DENSE SIMILARITY (cosine, or the metric
    # mapped onto it), never against RRF scores. Model-specific: None = use the
    # backend default below until the eval sweep runs.
    abstain_threshold: float | None = None
    # Answer-level gate (step 2 of grounding): retrieval found related sources, but
    # does any sentence in them actually answer the question? Below this score the
    # reply is "I do not know." with the related sources listed. Scale depends on the
    # scorer that ran: cross-encoder logits, or dense cosine + keyword overlap.
    # None = the provisional defaults in `answer_gate()` until the golden-set sweep.
    answer_threshold: float | None = None
    fuzzy_name_threshold: int = 88

    def __post_init__(self):
        problems = self.problems()
        if problems:
            raise ValueError("invalid settings:\n  - " + "\n  - ".join(problems))

    def problems(self) -> list[str]:
        p = []

        def one_of(name, allowed):
            if getattr(self, name) not in allowed:
                p.append(f"{name}={getattr(self, name)!r} must be one of {', '.join(allowed)}")

        one_of("chunk_unit", CHUNK_UNITS)
        one_of("distance", DISTANCES)
        one_of("search_mode", SEARCH_MODES)
        one_of("fusion", FUSIONS)
        for size, overlap in (("chunk_size", "chunk_overlap"),
                              ("slide_chunk_size", "slide_chunk_overlap")):
            if getattr(self, size) <= 0:
                p.append(f"{size} must be > 0")
            elif not 0 <= getattr(self, overlap) < getattr(self, size):
                p.append(f"{overlap} must be >= 0 and smaller than {size} "
                         f"({getattr(self, overlap)} vs {getattr(self, size)})")
        if self.min_chunk_chars < 0:
            p.append("min_chunk_chars must be >= 0")
        if self.hnsw_m < 0:
            p.append("hnsw_m must be >= 0")
        if self.hnsw_ef_construct < 4:
            p.append("hnsw_ef_construct must be >= 4")
        if self.hnsw_ef is not None and self.hnsw_ef <= 0:
            p.append("hnsw_ef must be > 0 (or unset)")
        if self.rrf_k < 0:
            p.append("rrf_k must be >= 0")
        for name in ("prefetch_k", "fusion_limit", "display_top_k"):
            if getattr(self, name) <= 0:
                p.append(f"{name} must be > 0")
        if self.distance == "manhattan" and self.abstain_threshold is None:
            p.append("distance=manhattan has no cosine equivalent: set KB_ABSTAIN_THRESHOLD "
                     "from a golden-set sweep")
        return p

    def chunk_config(self) -> dict:
        """Part of every file's fingerprint: a change here re-chunks on the next run."""
        return {k: getattr(self, k) for k in ("chunk_unit", "chunk_size", "chunk_overlap",
                                              "slide_chunk_size", "slide_chunk_overlap",
                                              "min_chunk_chars")}

    def index_config(self) -> dict:
        """The collection contract: a change here needs a full rebuild."""
        return {"distance": self.distance, "hnsw_m": self.hnsw_m,
                "hnsw_ef_construct": self.hnsw_ef_construct}

    @property
    def gate(self) -> float:
        if self.abstain_threshold is not None:
            return self.abstain_threshold
        # fastembed 0.56: measured 2026-11-03 on 12 queries over one DOCX
        # (on-topic 0.55–0.81, off-topic 0.37–0.55). Provisional until the golden-set sweep.
        return {"fastembed": 0.56, "bge-m3": 0.45}.get(self.embed_backend, 0.45)

    def answer_gate(self, scored_by: str) -> float:
        """Minimum extraction score for a sentence to be served as THE answer."""
        if self.answer_threshold is not None:
            return self.answer_threshold
        return ANSWER_GATES.get(scored_by, {}).get(self.embed_backend, ANSWER_GATES[scored_by]["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        threshold = os.getenv("KB_ABSTAIN_THRESHOLD")
        answer_t = os.getenv("KB_ANSWER_THRESHOLD")
        ef = os.getenv("KB_HNSW_EF")
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", d.qdrant_url),
            qdrant_path=os.getenv("QDRANT_PATH", d.qdrant_path),
            sqlite_path=os.getenv("KB_SQLITE", d.sqlite_path),
            collection=os.getenv("KB_COLLECTION", d.collection),
            corpus_dir=os.getenv("KB_CORPUS", d.corpus_dir),
            projects_csv=os.getenv("KB_PROJECTS_CSV", d.projects_csv),
            seniors_csv=os.getenv("KB_SENIORS_CSV", d.seniors_csv),
            embed_backend=os.getenv("EMBED_BACKEND", d.embed_backend),
            embed_model=os.getenv("EMBED_MODEL", d.embed_model),
            fastembed_dense=os.getenv("FASTEMBED_DENSE", d.fastembed_dense),
            fastembed_sparse=os.getenv("FASTEMBED_SPARSE", d.fastembed_sparse),
            answer_scorer=os.getenv("KB_ANSWER_SCORER", d.answer_scorer),
            ollama_url=os.getenv("OLLAMA_URL", d.ollama_url),
            ollama_model=os.getenv("OLLAMA_MODEL", d.ollama_model),
            enable_llm=_env_bool("ENABLE_LLM", d.enable_llm),
            chunk_unit=os.getenv("KB_CHUNK_UNIT", d.chunk_unit).strip().lower(),
            chunk_size=int(os.getenv("KB_CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(os.getenv("KB_CHUNK_OVERLAP", d.chunk_overlap)),
            slide_chunk_size=int(os.getenv("KB_SLIDE_CHUNK_SIZE", d.slide_chunk_size)),
            slide_chunk_overlap=int(os.getenv("KB_SLIDE_CHUNK_OVERLAP", d.slide_chunk_overlap)),
            min_chunk_chars=int(os.getenv("KB_MIN_CHUNK_CHARS", d.min_chunk_chars)),
            distance=os.getenv("KB_DISTANCE", d.distance).strip().lower(),
            hnsw_m=int(os.getenv("KB_HNSW_M", d.hnsw_m)),
            hnsw_ef_construct=int(os.getenv("KB_HNSW_EF_CONSTRUCT", d.hnsw_ef_construct)),
            search_mode=os.getenv("KB_SEARCH_MODE", d.search_mode).strip().lower(),
            fusion=os.getenv("KB_FUSION", d.fusion).strip().lower(),
            rrf_k=int(os.getenv("KB_RRF_K", d.rrf_k)),
            hnsw_ef=int(ef) if ef else None,
            exact_search=_env_bool("KB_EXACT_SEARCH", d.exact_search),
            prefetch_k=int(os.getenv("KB_PREFETCH_K", d.prefetch_k)),
            fusion_limit=int(os.getenv("KB_FUSION_LIMIT", d.fusion_limit)),
            display_top_k=int(os.getenv("KB_DISPLAY_TOP_K", d.display_top_k)),
            weak_match_count=int(os.getenv("KB_WEAK_MATCHES", d.weak_match_count)),
            abstain_threshold=float(threshold) if threshold else None,
            answer_threshold=float(answer_t) if answer_t else None,
            fuzzy_name_threshold=int(os.getenv("KB_FUZZY_NAME_THRESHOLD", d.fuzzy_name_threshold)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _print_effective() -> int:
    try:
        s = Settings.from_env()
    except ValueError as exc:
        print(exc)
        return 1
    groups = [("chunking (re-chunks on next ingestion run)", s.chunk_config()),
              ("vector index (needs ingestion --rebuild)", s.index_config()),
              ("search (per query)", {k: getattr(s, k) for k in (
                  "search_mode", "fusion", "rrf_k", "prefetch_k", "fusion_limit",
                  "display_top_k", "hnsw_ef", "exact_search")}
               | {"abstain_gate": s.gate})]
    for title, values in groups:
        print(title)
        for k, v in values.items():
            print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(_print_effective())
