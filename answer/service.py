"""THE entry point of the retrieval plane (Stage 06).

The Telegram bot, the CLI and eval/run_eval.py all call these functions and
nothing below them, so evaluation measures exactly what students see. No
interface code may re-implement ranking, thresholds or name resolution.

Public API
    answer(query, kb, facets=None, page=0)   free text: routes to the right path
    senior(name, kb)                         /senior <name>
    portfolio(senior_id, kb)                 deep link / Portfolio button
    browse(facets, kb, page=0)               /year 2024, facet chips
    log_interaction(kb, result, ...)         query_log row (training-pair shaped)

Every function returns a plain dict with the same keys (see `_result`).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from answer.extract import extract_answer, make_scorer
from answer.prompts import passages_for, synthesize
from answer.validate import (CITE_RE, IDK, check_support, is_idk, norm_title, validate_citations,
                             verify_quotes)
from config import Settings
from retrieval import intent as I
from retrieval.entity import resolve_senior
from retrieval.filters import build_filter, merge_facets
from retrieval.hybrid import SearchConfig, SearchUnavailable, hybrid_retrieve
from store.vector_schema import index_mismatch

log = logging.getLogger(__name__)
validation_log = logging.getLogger("answer.validation")

# Grounding contract (step 3): an answer is either quoted / checked source text,
# or "I do not know." — never a best guess. Two independent gates lead to IDK:
#   1. retrieval gate  nothing in the index is close enough      -> status "abstain"
#   2. answer gate     related sources exist, but no sentence in them answers the
#                      question (or a draft fails verification)  -> status "ok",
#                      sources listed as related, answer = NO_ANSWER
# Every topic-path result carries `grounding` saying which happened and why.
ABSTAIN = (f"{IDK} No indexed source covers this question. Try a topic keyword, "
           "a senior's name, or a year filter.")
NO_ANSWER = f"{IDK} The sources below are related, but none of them states the answer."
DEGRADED = ("Search is temporarily unavailable. Portfolios and year/program "
            "browsing still work.")


@dataclass
class KnowledgeBase:
    """Everything a request needs. Built once per process (bot, CLI, eval)."""
    client: Any
    embedder: Any
    db: Any
    settings: Settings = field(default_factory=Settings)
    scorer: Any = None            # answer-sentence cross-encoder (answer/extract.py); optional

    @classmethod
    def from_settings(cls, settings: Settings | None = None, *,
                      load_embedder: bool = True) -> "KnowledgeBase":
        from store.db import KBStore

        s = settings or Settings.from_env()
        embedder = scorer = None
        if load_embedder:
            from ingestion.embedder import make_embedder
            embedder = make_embedder(s)
            scorer = make_scorer(s.answer_scorer)
        return cls(open_qdrant(s), embedder, KBStore(s.sqlite_path), s, scorer)

    def close(self) -> None:
        """Release the local-mode Qdrant lock before interpreter shutdown."""
        try:
            self.client.close()
        except Exception:
            pass

    def models(self) -> dict:
        return {"embed": getattr(self.embedder, "revision", None),
                "llm": self.settings.ollama_model if self.settings.enable_llm else None}

    def health(self) -> list[str]:
        """Startup checks. An empty index must fail loudly, not answer nothing."""
        s, problems = self.settings, []
        where = s.qdrant_url or s.qdrant_path
        try:
            if not self.client.collection_exists(s.collection):
                problems.append(f"collection '{s.collection}' does not exist — run ingestion first")
            elif self.client.count(s.collection, exact=True).count == 0:
                problems.append(f"collection '{s.collection}' is empty — run ingestion first")
            elif (why := index_mismatch(self.client, s.collection, s.index_config())):
                problems.append(why)          # e.g. KB_DISTANCE changed since the last rebuild
        except Exception as exc:
            problems.append(f"Qdrant unreachable at {where}: {exc}")
        indexed = self.db.get_meta("embed_revision")
        rev = getattr(self.embedder, "revision", None)
        if self.embedder is not None and indexed and rev != indexed:
            problems.append(f"index was built with {indexed!r} but queries use {rev!r} — re-run "
                            "ingestion with --rebuild")
        return problems


def open_qdrant(s: Settings):
    from qdrant_client import QdrantClient

    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url)
    try:
        return QdrantClient(path=s.qdrant_path)     # local mode: one process at a time
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            raise RuntimeError(
                f"{s.qdrant_path} is open in another process (a query session or ingestion). "
                "Close it first, or run the Qdrant server and set QDRANT_URL.") from exc
        raise


# --------------------------------------------------------------------- shapes

def _result(route: str, query: str, normalized: str, **kw) -> dict:
    base = {
        "route": route, "status": "ok", "mode": None,
        "query": query, "normalized": normalized,
        "answer": None, "answer_source": None, "llm_sources": [],
        "results": [], "citations": [], "weak_matches": [],
        "portfolio": None, "other_projects": [], "senior_id": None,
        "candidates": [], "closest": [],
        "facets": {}, "counts": None, "note": None,
        "page": 0, "has_more": False, "gate_score": None,
        "grounding": None,        # topic path: {verdict, method, reason, evidence}
    }
    base.update(kw)
    return base


def _hit_card(hit, db) -> dict | None:
    meta = db.project(hit.project_id)
    if meta is None:
        # In the index but not granted/known in the metadata store (e.g. a
        # takedown not yet re-indexed). The manifest decides; never show it.
        return None
    p = hit.payload
    text, crumb = p.get("text", ""), p.get("breadcrumb")
    if crumb and text.startswith(crumb):
        text = text[len(crumb):]
    return {
        "project_id": hit.project_id,
        "title": meta["title_en"],
        "authors": [a["name"] for a in meta["authors"]],
        "senior_ids": meta["senior_ids"],
        "year": meta["year"],
        "program": meta["program"],
        "supervisor_id": meta["supervisor_id"],
        "repo_url": meta.get("repo_url"),
        "doc_id": p.get("doc_id"),
        "section": p.get("section"),
        "page": p.get("page_number"),
        "page_label": p.get("page_label") or f"p.{p.get('page_number')}",
        "heading": p.get("heading"),
        "source": p.get("source_url") or p.get("source_file"),
        "snippet": text.strip(),          # verbatim chunk body; the formatter trims it
        # every fused chunk of this project, best first — input to answer extraction
        "evidence": [{k: c.get(k) for k in ("chunk_id", "text", "breadcrumb", "page_number",
                                            "page_label", "heading")} for c in hit.chunks],
        "matched": "+".join(hit.matched_by) or "fusion",
        "score": round(hit.score, 4),
        "dense_score": None if hit.dense_score is None else round(hit.dense_score, 4),
    }


def _project_card(p: dict) -> dict:
    return {
        "project_id": p["project_id"], "title": p["title_en"],
        "authors": [a["name"] for a in p["authors"]], "senior_ids": p["senior_ids"],
        "year": p["year"], "program": p["program"], "supervisor_id": p["supervisor_id"],
        "repo_url": p.get("repo_url"), "doc_id": None, "section": "abstract", "page": None,
        "snippet": p["abstract"], "matched": "filter", "score": None, "dense_score": None,
    }


# ---------------------------------------------------------------- topic path

def _search(kb: KnowledgeBase, routed: dict, facets: dict) -> tuple[list[dict], float]:
    if kb.embedder is None:
        raise RuntimeError("embedder not loaded: topic search needs bge-m3")
    s = kb.settings
    hits = hybrid_retrieve(kb.client, kb.embedder, routed["topic"] or routed["raw"],
                           lexical_query=routed["lexical"], flt=build_filter(facets),
                           collection=s.collection, search=SearchConfig.from_settings(s))
    cards = [c for h in hits if (c := _hit_card(h, kb.db))]
    # Gate on dense cosine: RRF scores are rank-based and not comparable across queries.
    gate = max((c["dense_score"] or 0.0 for c in cards), default=0.0)
    if gate >= s.gate:
        # Above the gate, drop the tail that neither path really matched: fusion
        # always fills `fusion_limit`, even with unrelated projects in a small corpus.
        cards = [c for c in cards if "lexical" in c["matched"]
                 or (c["dense_score"] or 0.0) >= s.gate]
    return cards, gate


def _topic_result(kb: KnowledgeBase, routed: dict, facets: dict, page: int,
                  enable_llm: bool, route: str | None = None) -> dict:
    s = kb.settings
    route = route or routed["intent"]
    common = dict(query=routed["raw"], normalized=routed["normalized"], facets=facets)
    try:
        cards, gate = _search(kb, routed, facets)
    except SearchUnavailable as exc:
        log.warning("hybrid search unavailable: %s", exc)
        return _result(route, status="degraded", note=DEGRADED, **common)

    if not cards or gate < s.gate:
        weak = [{**c, "low_confidence": True} for c in cards[:s.weak_match_count]]
        return _result(route, status="abstain", answer=ABSTAIN, weak_matches=weak, gate_score=gate,
                       grounding=_unknown("retrieval_gate",
                                          f"best dense similarity {gate:.3f} < gate {s.gate}"),
                       **common)

    k = s.display_top_k
    shown = cards[page * k:(page + 1) * k]
    res = _result(route, mode="extractive", results=shown, citations=shown, page=page,
                  has_more=len(cards) > (page + 1) * k, gate_score=gate, **common)
    if shown:
        _extractive_answer(kb, routed["topic"] or routed["raw"], shown, res)
    if routed["intent"] == I.ANSWER and enable_llm and shown:
        _llm_answer(kb, routed["raw"], res)
    return res


def _unknown(reason: str, detail: str) -> dict:
    return {"verdict": "unknown", "method": None, "reason": reason, "detail": detail, "evidence": []}


def _extractive_answer(kb: KnowledgeBase, question: str, shown: list[dict], res: dict) -> None:
    """The sentence(s) that answer the question, verbatim — or I do not know."""
    ex = extract_answer(question, shown, kb.embedder, kb.scorer)
    if ex is None:
        res.update(answer=NO_ANSWER, grounding=_unknown(
            "no_answer", "the retrieved sources contain no answer-shaped sentence"))
        return
    floor = kb.settings.answer_gate(ex["scored_by"])
    if ex["score"] < floor:
        res.update(answer=NO_ANSWER, grounding=_unknown(
            "answer_gate", f"best sentence scored {ex['score']} < {floor} ({ex['scored_by']})"))
        return
    problems = verify_quotes(ex["quotes"])
    if problems:                                    # an extraction bug, never a user error
        validation_log.error("extractive answer failed verbatim check for %r: %s", question, problems)
        res.update(answer=NO_ANSWER, grounding=_unknown("not_verbatim", "; ".join(problems)))
        return
    card = ex["card"]
    src = {"project_id": card["project_id"], "title": card["title"], "source": card.get("source"),
           "page": ex["page"], "page_label": ex["page_label"], "heading": ex["heading"],
           "score": ex["score"], "chunk_id": ex["chunk_id"]}
    res.update(answer=ex["text"], answer_source=src, grounding={
        "verdict": "verified", "method": "verbatim", "reason": "quoted",
        "detail": f"{len(ex['quotes'])} quoted unit(s), each found verbatim in its chunk; "
                  f"score {ex['score']} >= {floor} ({ex['scored_by']})",
        "evidence": [{"chunk_id": q["chunk_id"], "quote": q["quote"], "source": src["source"],
                      "page_label": src["page_label"]} for q in ex["quotes"]]})


def _llm_answer(kb: KnowledgeBase, query: str, res: dict) -> None:
    """Gemma 4 draft, kept only if its citations AND its claims check out against the
    passages it was shown. Otherwise the (gated, verbatim) extractive answer stands."""
    s = kb.settings
    passages = passages_for(res["citations"])
    try:
        draft = synthesize(query, passages, ollama_url=s.ollama_url, model=s.ollama_model)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("Gemma unavailable, serving extractive answer: %s", exc)
        res["mode"] = "extractive_fallback"
        return
    if is_idk(draft):
        # The model read every passage and found no answer: say so, even if a sentence
        # scored above the extractive gate — the conservative reading wins.
        res.update(mode="llm", answer=NO_ANSWER, answer_source=None, grounding=_unknown(
            "llm_idk", "Gemma found no answer in the retrieved passages"))
        return
    # The allowed set is exactly what Gemma was shown: every (title, page) passage.
    problems = [] if validate_citations(draft, passages) else [
        "a citation is missing or points outside the retrieved passages"]
    problems = problems or check_support(draft, passages)
    if problems:
        validation_log.warning("rejected draft for %r: %s | %s", query, problems, draft[:300])
        res["mode"] = "extractive_fallback"
        res["rejected_draft"] = {"problems": problems}     # the draft text itself is never shown
        return
    cited = {(norm_title(t), p) for t, p in CITE_RE.findall(draft)}
    sources = [p for p in passages if (norm_title(p["title"]), str(p["page"])) in cited]
    res.update(mode="llm", answer=draft, llm_sources=sources, grounding={
        "verdict": "verified", "method": "llm_checked", "reason": "cited_and_supported",
        "detail": "every citation was shown to the model; every sentence's numbers, names "
                  "and wording are found in the passage it cites",
        "evidence": [{"chunk_id": None, "quote": p["snippet"], "source": p.get("source"),
                      "page_label": p.get("page_label")} for p in sources]})


# --------------------------------------------------------------- senior path

def _senior_result(kb: KnowledgeBase, name: str, query: str, normalized: str,
                   route: str) -> dict:
    s = kb.settings
    r = resolve_senior(I.normalize(name), kb.db, kb.client, collection=s.collection,
                       fuzzy_threshold=s.fuzzy_name_threshold)
    if r["status"] == "resolved":
        pf = kb.db.portfolio(r["senior_id"])
        return _result(I.PORTFOLIO, query, normalized, portfolio=pf, senior_id=r["senior_id"],
                       resolution={k: r[k] for k in ("confidence", "via")})
    if r["status"] == "ambiguous":
        return _result(route, query, normalized, status="ambiguous", candidates=r["candidates"])
    return _result(route, query, normalized, status="not_found", closest=r["closest"],
                   note="No senior with that name in the indexed cohort.")


def _senior_topic(kb: KnowledgeBase, routed: dict, facets: dict, page: int) -> dict:
    """§5.5: 'Sok Dara's OCR project' = SENIOR_LOOKUP ∩ TOPIC_SEARCH, neither half dropped."""
    q, n, topic = routed["raw"], routed["normalized"], routed["topic"]
    who = _senior_result(kb, routed["name"], q, n, I.SENIOR_TOPIC)
    if who["status"] == "ambiguous":
        return {**who, "topic": topic}
    if who["status"] == "not_found":
        res = _topic_result(kb, {**routed, "intent": I.TOPIC_SEARCH}, facets, page, False,
                            route=I.SENIOR_TOPIC)
        return {**res, "closest": who["closest"],
                "note": "No senior matched that name; showing topic results only."}

    sid, pf = who["senior_id"], who["portfolio"]
    # Intersection happens inside Qdrant, as a payload filter on senior_ids.
    res = _topic_result(kb, routed, {**facets, "senior_ids": [sid]}, page, False,
                        route=I.SENIOR_TOPIC)
    if res["status"] == "ok" and res["results"]:
        hit_ids = {c["project_id"] for c in res["results"]}
        return {**res, "portfolio": pf, "senior_id": sid,
                "other_projects": [p for p in pf["projects"] if p["project_id"] not in hit_ids]}
    note = (DEGRADED if res["status"] == "degraded"
            else f"None of {pf['name']}'s indexed projects matched '{topic}'.")
    return _result(I.SENIOR_TOPIC, q, n, portfolio=pf, senior_id=sid,
                   other_projects=pf["projects"], note=note)


# ---------------------------------------------------------------- public API

def answer(query: str, kb: KnowledgeBase, *, facets: dict | None = None, page: int = 0,
           enable_llm: bool | None = None) -> dict:
    t0 = time.perf_counter()
    enable_llm = kb.settings.enable_llm if enable_llm is None else enable_llm
    routed = I.route(query, kb.db.known_names(), kb.db.known_surnames())
    intent, facets_m = routed["intent"], merge_facets(routed, facets)

    if intent in (I.SENIOR_LOOKUP, I.PORTFOLIO) and routed["name"]:
        res = _senior_result(kb, routed["name"], query, routed["normalized"], intent)
        if res["status"] == "not_found" and routed["name_heuristic"]:
            # Routed on a surname guess only: a topic search is the safer answer.
            closest = res["closest"]
            res = _topic_result(kb, {**routed, "intent": I.TOPIC_SEARCH}, facets_m, page, enable_llm)
            res["closest"] = closest
    elif intent == I.SENIOR_TOPIC:
        res = _senior_topic(kb, routed, facets_m, page)
    elif intent == I.FACET_BROWSE:
        res = browse(facets_m, kb, page=page)
        res.update(query=query, normalized=routed["normalized"])
    else:
        res = _topic_result(kb, {**routed, "intent": intent if intent != I.PORTFOLIO else I.TOPIC_SEARCH},
                            facets_m, page, enable_llm)
        if res["status"] == "abstain" and not facets_m and len(routed["normalized"].split()) <= 4:
            # A short query nothing matched may be a misspelled name ("sokdaraa").
            # Only a single confident fuzzy match counts; ambiguity keeps the abstain.
            who = resolve_senior(routed["normalized"], kb.db, kb.client,
                                 collection=kb.settings.collection,
                                 fuzzy_threshold=kb.settings.fuzzy_name_threshold)
            if who["status"] == "resolved":
                res = _result(I.PORTFOLIO, query, routed["normalized"],
                              portfolio=kb.db.portfolio(who["senior_id"]),
                              senior_id=who["senior_id"],
                              resolution={k: who[k] for k in ("confidence", "via")},
                              note="No project matched; this looks like a senior's name.")

    res["intent"] = intent
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    res["models"] = kb.models()
    return res


def senior(name: str, kb: KnowledgeBase) -> dict:
    t0 = time.perf_counter()
    res = _senior_result(kb, name, name, I.normalize(name), I.SENIOR_LOOKUP)
    res.update(intent=I.SENIOR_LOOKUP, latency_ms=int((time.perf_counter() - t0) * 1000),
               models=kb.models())
    return res


def portfolio(senior_id: str, kb: KnowledgeBase) -> dict:
    """Served from SQLite only, so it renders even when Qdrant is down."""
    pf = kb.db.portfolio(senior_id)
    if pf is None:
        return _result(I.PORTFOLIO, senior_id, senior_id, status="not_found",
                       note="This portfolio is not available.", intent=I.PORTFOLIO)
    return _result(I.PORTFOLIO, senior_id, senior_id, portfolio=pf, senior_id=senior_id,
                   intent=I.PORTFOLIO)


def _index_project_ids(kb: KnowledgeBase, facets: dict) -> set[str]:
    flt, ids, offset = build_filter(facets), set(), None
    while True:
        pts, offset = kb.client.scroll(kb.settings.collection, scroll_filter=flt, limit=256,
                                       offset=offset, with_payload=["project_id"])
        ids.update(p.payload["project_id"] for p in pts)
        if offset is None:
            return ids


def browse(facets: dict | None, kb: KnowledgeBase, *, page: int = 0, page_size: int = 10) -> dict:
    """Facet browsing: Qdrant payload filter for the list, SQLite GROUP BY for counts."""
    facets = merge_facets(None, facets)
    counts = kb.db.facet_counts(facets)
    label = " ".join(f"{k}={v}" for k, v in facets.items())
    status, note, missing = "ok", None, []
    try:
        projects = kb.db.projects_by_ids(sorted(_index_project_ids(kb, facets)))
        # The list length should equal the SQLite count — the filter-bug detector (Trace 4).
        listed = {p["project_id"] for p in projects}
        missing = [p["project_id"] for p in kb.db.list_projects(facets) if p["project_id"] not in listed]
        if missing:
            note = f"{len(missing)} project(s) are in the metadata store but not in the search index."
    except Exception as exc:
        log.warning("facet browse falling back to SQLite: %s", exc)
        projects, status, note = kb.db.list_projects(facets), "degraded", DEGRADED
    cards = [_project_card(p) for p in projects]
    shown = cards[page * page_size:(page + 1) * page_size]
    return _result(I.FACET_BROWSE, label, label, status=status, results=shown, facets=facets,
                   counts=counts, note=note, missing_from_index=missing, page=page,
                   has_more=len(cards) > (page + 1) * page_size, intent=I.FACET_BROWSE,
                   models=kb.models())


def log_interaction(kb: KnowledgeBase, result: dict, *, surface: str = "cli",
                    user_hash: str | None = None, message_id: int | None = None) -> str:
    """One query_log row per interaction, shaped for Semester-2 training pairs."""
    query_id = str(uuid.uuid4())
    shown = result.get("results") or result.get("weak_matches") or []
    kb.db.log_query(
        query_id=query_id, surface=surface, tg_user_hash=user_hash, tg_message_id=message_id,
        raw_query=result.get("query", ""), normalized_query=result.get("normalized", ""),
        intent=result.get("intent") or result.get("route", ""),
        resolved_senior=result.get("senior_id"),
        retrieved_json=json.dumps([{"project_id": c["project_id"], "rank": i,
                                    "score": c.get("score"), "dense_score": c.get("dense_score")}
                                   for i, c in enumerate(shown, 1)]),
        latency_ms=result.get("latency_ms"),
    )
    return query_id
