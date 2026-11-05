"""Stage 05b — senior entity resolution: name -> exactly one senior_id, or a question.

Order: (1) curated alias table, (2) Qdrant full-text index, (3) rapidfuzz.
Two seniors sharing a name are never merged: that case returns `ambiguous`
and the interface shows a candidate keyboard.
"""
from __future__ import annotations

import logging

from qdrant_client import models
from rapidfuzz import fuzz

from store.vector_schema import COLLECTION, TEXT_FIELD

log = logging.getLogger(__name__)


def name_score(query: str, name: str, **_) -> float:
    """token_set handles reordering ('dara sok'); the space-free ratio handles
    run-together typos ('sokdaraa'), which token_set alone scores around 50."""
    return max(fuzz.token_set_ratio(query, name),
               fuzz.ratio(query.replace(" ", ""), name.replace(" ", "")))


def _scored_directory(normalized: str, db) -> list[tuple[str, float]]:
    """Best score per senior over their canonical name and every alias, descending."""
    best: dict[str, float] = {}
    for sid, name in db.name_directory():
        s = name_score(normalized, name)
        if s > best.get(sid, -1):
            best[sid] = s
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def _candidate(db, sid: str, score: float) -> dict:
    return {"senior_id": sid, "score": round(score, 1), **db.senior_card(sid)}


def _text_index_hits(normalized: str, db, client, collection: str) -> set[str]:
    """Seniors on projects whose author string contains every query token.

    A matching point carries ALL its co-authors' ids, so each candidate is
    re-checked against that senior's own names before it counts.
    """
    pts, _ = client.scroll(
        collection,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key=TEXT_FIELD, match=models.MatchText(text=normalized))]),
        limit=50, with_payload=["senior_ids"],
    )
    ids = {sid for p in pts for sid in p.payload.get("senior_ids", [])}
    tokens = set(normalized.split())
    names: dict[str, list[str]] = {}
    for sid, name in db.name_directory():
        names.setdefault(sid, []).append(name)
    return {sid for sid in ids
            if any(tokens <= set(n.split()) for n in names.get(sid, []))}


def resolve_senior(normalized: str, db, client=None, *, collection: str = COLLECTION,
                   fuzzy_threshold: int = 88) -> dict:
    if not normalized:
        return {"status": "not_found", "closest": []}

    # 1) curated alias table — the Khmer <-> Latin bridge
    exact = db.lookup_alias(normalized)
    if len(exact) == 1:
        return {"status": "resolved", "senior_id": exact[0], "confidence": 1.00, "via": "alias"}
    if len(exact) > 1:
        return {"status": "ambiguous", "via": "alias",
                "candidates": [_candidate(db, sid, 100) for sid in exact]}

    # 2) Qdrant full-text index over indexed author names
    if client is not None:
        try:
            hits = _text_index_hits(normalized, db, client, collection)
        except Exception as exc:                  # Qdrant down: SQLite paths still work
            log.warning("text-index lookup skipped: %s", exc)
            hits = set()
        if len(hits) == 1:
            return {"status": "resolved", "senior_id": hits.pop(), "confidence": 0.90,
                    "via": "text_index"}

    # 3) fuzzy over the directory + aliases
    scored = _scored_directory(normalized, db)
    above = [(sid, sc) for sid, sc in scored if sc >= fuzzy_threshold]
    if len(above) == 1:
        sid, sc = above[0]
        return {"status": "resolved", "senior_id": sid, "confidence": round(sc / 100, 2),
                "via": "fuzzy"}
    if above:
        # Never pick the higher score: 91 vs 89 is noise, a wrong portfolio is a
        # factual error about a real person.
        return {"status": "ambiguous", "via": "fuzzy",
                "candidates": [_candidate(db, sid, sc) for sid, sc in above[:5]]}

    return {"status": "not_found",
            "closest": [{**_candidate(db, sid, sc), "confirmed": False} for sid, sc in scored[:3]]}
