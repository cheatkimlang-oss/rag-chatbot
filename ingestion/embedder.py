"""Dense + sparse encoders — the ONE place embedding models are loaded.

Swapping in the Semester-2 fine-tuned model is a change here plus a full
re-index (FR-14). The same instance must encode chunks and queries; a revision
mismatch degrades retrieval without raising, so `revision` is stored with the
index at ingestion time and checked at startup.

Interface every embedder implements:
    revision: str            pinned model identity
    dim: int                 dense dimension (collection contract)
    sparse_idf: bool         sparse values are raw term weights needing Qdrant's IDF modifier
    encode(texts)            documents -> (dense[], sparse[(indices, values)])
    encode_query(texts)      queries   -> same shape (may add query instructions)
    count_tokens(text)       the model's own token count (KB_CHUNK_UNIT=tokens)
    max_tokens: int          inputs longer than this are silently truncated
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SparseVec = tuple[list[int], list[float]]
# fastembed defaults to the OS temp folder, which Disk Cleanup empties (-> re-download).
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))


class Embedder:
    """BAAI/bge-m3: dense 1024-d + learned lexical weights from one model."""
    dim = 1024
    sparse_idf = False

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool | None = None,
                 max_length: int = 512):
        import torch
        from FlagEmbedding import BGEM3FlagModel   # heavy import: only when really used

        if use_fp16 is None:                       # fp16 helps on GPU, hurts (or fails) on CPU
            use_fp16 = torch.cuda.is_available()
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.revision = model_name
        self.max_length = self.max_tokens = max_length
        log.info("embedder loaded: %s (fp16=%s)", model_name, use_fp16)

    def count_tokens(self, text: str) -> int:
        # No truncation here: the point is to measure text longer than max_length.
        return len(self.model.tokenizer(text, add_special_tokens=False,
                                        truncation=False)["input_ids"])

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        out = self.model.encode(texts, return_dense=True, return_sparse=True,
                                return_colbert_vecs=False, max_length=self.max_length)
        dense = out["dense_vecs"].tolist()                  # (n, 1024), L2-normalized
        sparse = [([int(t) for t in w], [float(v) for v in w.values()])
                  for w in out["lexical_weights"]]
        return dense, sparse

    encode_query = encode          # bge-m3 needs no query instruction


class FastEmbedder:
    """Light CPU stack via fastembed (ONNX): a small dense model + BM25 sparse."""
    sparse_idf = True               # Qdrant/bm25 emits term frequencies; IDF is applied by Qdrant

    def __init__(self, dense_model: str = "BAAI/bge-small-en-v1.5",
                 sparse_model: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.dense = TextEmbedding(dense_model, cache_dir=FASTEMBED_CACHE)
        self.sparse = SparseTextEmbedding(sparse_model, cache_dir=FASTEMBED_CACHE)
        self.dim = len(next(iter(self.dense.embed(["dimension probe"]))))
        self.sparse_idf = sparse_model.lower().endswith("bm25")
        self.revision = f"fastembed:{dense_model}+{sparse_model}"
        self.max_tokens = 512                   # BERT-family dense models
        self._tokenizer = None
        log.info("embedder loaded: %s (dim=%d)", self.revision, self.dim)

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            loaded = getattr(getattr(self.dense, "model", None), "tokenizer", None)
            if loaded is None:
                raise ValueError("fastembed did not expose its tokenizer; use KB_CHUNK_UNIT=chars")
            # fastembed's copy truncates at 512; a private untruncated copy measures honestly.
            self._tokenizer = Tokenizer.from_str(loaded.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    @staticmethod
    def _pack(dense, sparse):
        return ([v.tolist() for v in dense],
                [(s.indices.tolist(), s.values.tolist()) for s in sparse])

    def encode(self, texts):
        return self._pack(self.dense.embed(texts), self.sparse.embed(texts))

    def encode_query(self, texts):
        return self._pack(self.dense.query_embed(texts), self.sparse.query_embed(texts))


def make_embedder(settings):
    if settings.embed_backend == "bge-m3":
        return Embedder(settings.embed_model)
    if settings.embed_backend == "fastembed":
        return FastEmbedder(settings.fastembed_dense, settings.fastembed_sparse)
    raise ValueError(f"unknown EMBED_BACKEND {settings.embed_backend!r}")
