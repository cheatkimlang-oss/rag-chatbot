import httpx
import pytest

from answer import service
from answer.validate import validate_citations


@pytest.fixture
def no_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Gemma must not be called here")
    monkeypatch.setattr(service, "synthesize", boom)


def test_topic_search_returns_cards(kb, no_llm):
    res = service.answer("khmer ocr text recognition", kb)
    assert res["status"] == "ok" and res["mode"] == "extractive"
    ids = [c["project_id"] for c in res["results"]]
    assert {"dse2024-07", "dse2023-04"} <= set(ids)
    card = res["results"][0]
    assert card["authors"] and card["page"] and not card["snippet"].startswith("[Project:")


def test_pending_permission_never_surfaces(kb):
    res = service.answer("khmer ocr chatbot permission pending", kb)
    shown = res["results"] + res["weak_matches"]
    assert "ite2023-50" not in {c["project_id"] for c in shown}


@pytest.mark.parametrize("query", ["quantum computing lab schedule 2019",
                                   "quantum computing lab schedule"])   # gate, not just the year filter
def test_negative_query_abstains_without_llm(kb, no_llm, query):
    res = service.answer(query, kb, enable_llm=True)
    assert res["status"] == "abstain"
    assert res["answer"] == service.ABSTAIN
    assert all(c["low_confidence"] for c in res["weak_matches"])


def test_name_query_opens_portfolio_with_consent(kb):
    for q in ("Sok Dara", "សុខ ដារា"):
        res = service.answer(q, kb)
        pf = res["portfolio"]
        assert res["route"] == "PORTFOLIO" and pf["senior_id"] == "s-2024-014"
        assert [p["project_id"] for p in pf["projects"]] == ["dse2024-07", "dse2023-04"]  # newest first
        assert pf["contact"] == {"github": "github.com/sokdara"} and pf["relay_available"]


def test_non_consenting_senior_shows_no_contact(kb):
    pf = service.senior("Chan Sreyneang", kb)["portfolio"]
    assert pf["contact"] is None and pf["contact_notice"] and not pf["relay_available"]


def test_relay_requires_senior_to_have_started_bot(kb):
    pf = service.portfolio("s-2023-005", kb)["portfolio"]
    assert pf["relay_available"] is False


def test_metadata_only_portfolio_without_qdrant(kb):
    kb.client = None                      # portfolio must not touch the vector store
    res = service.portfolio("s-2022-030", kb)
    assert res["status"] == "ok" and res["portfolio"]["metadata_only"] is True


def test_ambiguous_name_returns_candidates(kb):
    kb.db.upsert_senior(dict(senior_id="s-2023-077", full_name_latin="Sok Dara",
                             name_normalized="sok dara", program="ITE", cohort_year=2023))
    res = service.answer("Sok Dara", kb)
    assert res["status"] == "ambiguous" and res["portfolio"] is None
    assert len(res["candidates"]) == 2


def test_misspelled_name_as_plain_text_opens_portfolio(kb):
    res = service.answer("sokdaraa", kb)
    assert res["route"] == "PORTFOLIO" and res["senior_id"] == "s-2024-014"


def test_unrelated_tail_is_dropped_above_gate(kb):
    res = service.answer("rice leaf disease", kb)
    assert [c["project_id"] for c in res["results"]] == ["ite2023-11"]


def test_heuristic_name_miss_falls_back_to_topic(kb):
    res = service.answer("Chan Visal", kb)          # "chan" is a surname; nobody is named this
    assert res["route"] == "TOPIC_SEARCH" and res["closest"]


def test_multi_intent_intersects_on_senior(kb):
    res = service.answer("Sok Dara's OCR project", kb)
    assert res["route"] == "SENIOR_TOPIC" and res["senior_id"] == "s-2024-014"
    assert {c["project_id"] for c in res["results"]} <= {"dse2024-07", "dse2023-04"}


def test_multi_intent_empty_intersection_keeps_both_halves(kb):
    res = service.answer("Kim Lang's ocr project", kb)
    assert res["results"] == [] and res["portfolio"]["senior_id"] == "s-2023-005"
    assert "None of Kim Lang's" in res["note"]


def test_facet_browse_counts_from_sqlite(kb):
    res = service.answer("2023", kb)
    assert res["route"] == "FACET_BROWSE"
    assert res["counts"]["total"] == 2
    assert {c["project_id"] for c in res["results"]} == {"dse2023-04", "ite2023-11"}
    assert res["missing_from_index"] == []


def test_facet_browse_flags_metadata_only_projects(kb):
    res = service.browse({"year": [2022]}, kb)
    assert res["results"] == [] and res["missing_from_index"] == ["ite2022-02"]


def test_explicit_facets_prefilter_topic_search(kb):
    res = service.answer("khmer ocr", kb, facets={"year": [2023]})
    assert {c["year"] for c in res["results"]} == {2023}


def test_qdrant_down_degrades_search_but_not_portfolios(kb):
    class Down:
        def __getattr__(self, name):
            def fail(*a, **k):
                raise ConnectionError("qdrant down")
            return fail
    kb.client = Down()
    assert service.answer("khmer ocr text recognition", kb)["status"] == "degraded"
    assert service.answer("Sok Dara", kb)["portfolio"]["senior_id"] == "s-2024-014"
    assert service.browse({"year": [2023]}, kb)["status"] == "degraded"


QUESTION = "which projects did khmer ocr text recognition?"


def _passage(passages, project_id, page):
    return next(p for p in passages if p["project_id"] == project_id and p["page"] == page)


def test_llm_answer_with_valid_citations(kb, monkeypatch):
    def fake(query, passages, **kw):
        p = _passage(passages, "dse2024-07", 1)          # the passage that says it
        return f"It segments Khmer handwriting. [Project: {p['title']}, p.{p['page']}]"
    monkeypatch.setattr(service, "synthesize", fake)
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["route"] == "ANSWER" and res["mode"] == "llm" and "[Project:" in res["answer"]
    assert res["grounding"]["verdict"] == "verified" and res["grounding"]["method"] == "llm_checked"


def test_llm_real_citation_for_an_unsupported_claim_is_rejected(kb, monkeypatch):
    # A real (title, page) the model WAS shown — but that passage is about a labelling
    # tool and never says this. Citation-only validation used to accept it.
    def fake(query, passages, **kw):
        p = _passage(passages, "dse2023-04", 1)
        return f"It segments Khmer handwriting. [Project: {p['title']}, p.{p['page']}]"
    monkeypatch.setattr(service, "synthesize", fake)
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["mode"] == "extractive_fallback" and "segments" not in (res["answer"] or "")
    assert "wording" in res["rejected_draft"]["problems"][0]


def test_llm_invented_number_is_rejected(kb, monkeypatch):
    def fake(query, passages, **kw):
        p = _passage(passages, "dse2024-07", 1)          # says 81%, not 97%
        return f"The pipeline reaches 97% character accuracy. [Project: {p['title']}, p.1]"
    monkeypatch.setattr(service, "synthesize", fake)
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["mode"] == "extractive_fallback" and "97" not in (res["answer"] or "")
    assert "number" in res["rejected_draft"]["problems"][0]


def test_llm_i_do_not_know_is_passed_through(kb, monkeypatch):
    monkeypatch.setattr(service, "synthesize", lambda *a, **k: "I do not know.")
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["answer"] == service.NO_ANSWER and res["answer"].startswith("I do not know.")
    assert res["grounding"]["reason"] == "llm_idk" and res["answer_source"] is None


def test_llm_invented_citation_falls_back(kb, monkeypatch):
    monkeypatch.setattr(service, "synthesize",
                        lambda *a, **k: "Made up. [Project: Quantum Lab Scheduler, p.9]")
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["mode"] == "extractive_fallback" and res["results"]
    # The rejected draft is never shown; the answer is verbatim retrieved text.
    assert "Quantum" not in res["answer"]
    assert any(res["answer"] in c["snippet"] for c in res["results"])


def test_llm_may_cite_any_passage_it_was_shown_but_nothing_else(kb, monkeypatch):
    seen = {}

    def fake(query, passages, **kw):
        seen["passages"] = passages
        p = next(p for p in passages if p["page"] == 4)            # NOT the best chunk (p.1)
        return f"Lines are segmented first. [Project: {p['title']}, p.4]"
    monkeypatch.setattr(service, "synthesize", fake)
    res = service.answer(QUESTION, kb, enable_llm=True)
    assert res["mode"] == "llm"
    assert [(p["project_id"], p["page"]) for p in res["llm_sources"]] == [("dse2024-07", 4)]
    assert all(p["snippet"] and not p["snippet"].startswith("[Project:") for p in seen["passages"])

    monkeypatch.setattr(service, "synthesize",
                        lambda q, passages, **kw: f"[Project: {passages[0]['title']}, p.99]")
    assert service.answer(QUESTION, kb, enable_llm=True)["mode"] == "extractive_fallback"


def test_ollama_down_falls_back(kb, monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(service, "synthesize", down)
    assert service.answer(QUESTION, kb, enable_llm=True)["mode"] == "extractive_fallback"


def test_validator():
    cites = [{"title": "Khmer OCR, Revisited", "page": 4}]
    assert validate_citations("x [Project: Khmer OCR, Revisited, p.4]", cites)
    assert not validate_citations("x [Project: Khmer OCR, Revisited, p.5]", cites)
    assert not validate_citations("no citations at all", cites)


def test_log_interaction_writes_training_shaped_row(kb):
    res = service.answer("khmer ocr text recognition", kb)
    qid = service.log_interaction(kb, res, surface="cli")
    row = kb.db.conn.execute("SELECT * FROM query_log WHERE query_id = ?", (qid,)).fetchone()
    assert row["intent"] == "TOPIC_SEARCH" and '"rank": 1' in row["retrieved_json"]


def test_health_reports_empty_index_and_model_mismatch(kb):
    assert kb.health() == []
    kb.db.set_meta("embed_revision", "BAAI/bge-m3")
    assert any("--rebuild" in p for p in kb.health())
    kb.client.delete_collection(kb.settings.collection)
    assert any("does not exist" in p for p in kb.health())
