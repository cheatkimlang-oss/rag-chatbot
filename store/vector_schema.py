"""Qdrant collection contract (Architecture Spec §7.3).

Shared by both planes: ingestion writes points in this shape, retrieval reads
them. Changing anything here means a full re-index.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models

COLLECTION = "rupp_senior_projects_v1"
DENSE = "dense"
LEXICAL = "lexical"
DENSE_DIM = 1024

KEYWORD_FIELDS = ["project_id", "doc_id", "doc_type", "program", "supervisor_id",
                  "senior_ids", "domain", "tech_stack"]
INTEGER_FIELDS = ["year"]
TEXT_FIELD = "author_names_normalized"

# Keys every point must carry (ingestion/verify.py asserts these).
REQUIRED_PAYLOAD = ("project_id", "title_en", "senior_ids", "author_names_normalized",
                    "doc_id", "doc_type", "year", "program", "section",
                    "page_number", "chunk_id", "text")


DISTANCE = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID, "manhattan": models.Distance.MANHATTAN}
DEFAULT_INDEX = {"distance": "cosine", "hnsw_m": 16, "hnsw_ef_construct": 200}


def to_similarity(score: float, distance: str = "cosine") -> float:
    """Map a dense score onto 'higher = more similar', on the cosine scale where possible.

    Qdrant returns similarities for cosine/dot but DISTANCES for euclid/manhattan
    (lower = better). Both embedders emit L2-normalized vectors, so dot == cosine
    and euclid maps exactly: cos = 1 - d^2 / 2. Manhattan has no cosine
    equivalent (1 / (1 + d) keeps the order only), so its gate must be swept.
    """
    if distance == "euclid":
        return 1.0 - score * score / 2.0
    if distance == "manhattan":
        return 1.0 / (1.0 + score)
    return score


def index_config_of(client: QdrantClient, collection: str) -> dict | None:
    """What an existing collection was actually built with (None if it does not exist)."""
    if not client.collection_exists(collection):
        return None
    params = client.get_collection(collection).config
    vec = params.params.vectors[DENSE] if isinstance(params.params.vectors, dict) else params.params.vectors

    def hnsw(field: str):
        # The vector's hnsw_config is a partial override of the collection defaults.
        for cfg in (vec.hnsw_config, params.hnsw_config):
            value = getattr(cfg, field, None) if cfg is not None else None
            if value is not None:
                return value
        return DEFAULT_INDEX[f"hnsw_{field}"]

    by_enum = {v: k for k, v in DISTANCE.items()}
    return {"distance": by_enum.get(vec.distance, str(vec.distance).lower()),
            "hnsw_m": hnsw("m"), "hnsw_ef_construct": hnsw("ef_construct")}


def index_mismatch(client: QdrantClient, collection: str, wanted: dict) -> str | None:
    """A readable reason the collection does not match the configured contract."""
    have = index_config_of(client, collection)
    if have is None or have == wanted:
        return None
    diff = ", ".join(f"{k} {have[k]!r} -> {wanted[k]!r}" for k in wanted if have.get(k) != wanted[k])
    return (f"collection '{collection}' was built with a different vector index ({diff}): "
            "re-run ingestion with --rebuild")


def ensure_collection(client: QdrantClient, collection: str = COLLECTION,
                      dim: int = DENSE_DIM, sparse_idf: bool = False,
                      index: dict | None = None) -> None:
    """`sparse_idf=True` for BM25-style encoders that emit raw term weights.
    `index` = {distance, hnsw_m, hnsw_ef_construct} (Settings.index_config())."""
    if client.collection_exists(collection):
        return
    index = {**DEFAULT_INDEX, **(index or {})}
    client.create_collection(
        collection_name=collection,
        vectors_config={DENSE: models.VectorParams(
            size=dim, distance=DISTANCE[index["distance"]],
            hnsw_config=models.HnswConfigDiff(m=index["hnsw_m"],
                                              ef_construct=index["hnsw_ef_construct"]))},
        sparse_vectors_config={LEXICAL: models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False),
            modifier=models.Modifier.IDF if sparse_idf else None)},
    )
    for f in KEYWORD_FIELDS:
        client.create_payload_index(collection, f, field_schema=models.PayloadSchemaType.KEYWORD)
    for f in INTEGER_FIELDS:
        client.create_payload_index(collection, f, field_schema=models.PayloadSchemaType.INTEGER)
    # Full-text index: what makes name lookup work at all.
    client.create_payload_index(
        collection, TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD,
            min_token_len=2, lowercase=True),
    )
