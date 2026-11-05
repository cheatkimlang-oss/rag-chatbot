"""Retrieval metrics (Architecture Spec §13.2). Binary relevance; pure functions."""
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], expected: set[str], k: int = 5) -> float:
    """1.0 if any expected project is in the top k (the spec's per-query Recall@k)."""
    return float(bool(expected & set(ranked[:k])))


def reciprocal_rank(ranked: list[str], expected: set[str], k: int = 10) -> float:
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], expected: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, pid in enumerate(ranked[:k], start=1) if pid in expected)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else None
