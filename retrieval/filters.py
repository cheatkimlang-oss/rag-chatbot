"""Facet -> Qdrant payload filter (FR-08).

Filters are applied inside Qdrant (pre-filtering). Post-filtering in Python
after retrieval destroys recall whenever a filter is selective.
"""
from __future__ import annotations

from qdrant_client import models

FACET_KEYS = ("year", "program", "supervisor_id", "domain", "tech_stack", "senior_ids")


def _values(v) -> list:
    if v is None or v == "" or v == []:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def merge_facets(routed: dict | None, facets: dict | None) -> dict:
    """Explicit facets (chips, deep links) win; routed years/programs fill the gaps."""
    out = {k: _values(v) for k, v in (facets or {}).items() if k in FACET_KEYS and _values(v)}
    if routed:
        if routed.get("years") and "year" not in out:
            out["year"] = list(routed["years"])
        if routed.get("programs") and "program" not in out:
            out["program"] = list(routed["programs"])
    return out


def build_filter(facets: dict | None) -> models.Filter | None:
    must = []
    for key in FACET_KEYS:
        values = _values((facets or {}).get(key))
        if not values:
            continue
        if key == "year":
            values = [int(v) for v in values]
        match = (models.MatchValue(value=values[0]) if len(values) == 1
                 else models.MatchAny(any=values))
        must.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=must) if must else None
