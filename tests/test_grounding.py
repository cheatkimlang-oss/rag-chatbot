"""Step 3 — grounded & verifiable: verified source text, or "I do not know."."""
import dataclasses

import pytest

from answer import service
from answer.validate import IDK, check_support, is_idk, verify_quotes
from cli import render_qa


class StubScorer:
    """Cross-encoder stand-in with a fixed score, to test the gate itself."""
    kind = "cross"

    def __init__(self, value):
        self.value = value

    def score(self, question, texts):
        return [self.value] * len(texts)


def _with(kb, **settings):
    return dataclasses.replace(kb, settings=dataclasses.replace(kb.settings, **settings))


# ------------------------------------------------------------- unit checks

def test_verify_quotes_accepts_rewrapped_text_only():
    chunk = "[Project: X | p.4]\nWe build a pipeline that\nsegments Khmer   handwriting."
    assert verify_quotes([{"quote": "We build a pipeline that segments Khmer handwriting.",
                           "chunk_id": "c", "chunk_text": chunk}]) == []
    assert verify_quotes([{"quote": "We build a pipeline that segments Thai handwriting.",
                           "chunk_id": "c", "chunk_text": chunk}])


PASSAGES = [{"title": "OCR Study", "page": 4,
             "snippet": "Tesseract with custom khm traineddata reaches 81% character accuracy "
                        "on 500 scanned forms collected in Phnom Penh."}]
CITE = "[Project: OCR Study, p.4]"


@pytest.mark.parametrize("draft,problem", [
    (f"Tesseract reaches 81% character accuracy on scanned forms. {CITE}", None),
    # an uncited sentence inherits the citation that closes its paragraph
    (f"The forms were scanned. Tesseract reaches 81% character accuracy. {CITE}", None),
    (f"Tesseract reaches 91% character accuracy on scanned forms. {CITE}", "number"),
    (f"Forms were collected in Battambang with Tesseract traineddata. {CITE}", "name"),
    (f"The project trains a transformer language model for summarising news. {CITE}", "wording"),
    ("Tesseract reaches 81% character accuracy on scanned forms.", "uncited"),
])
def test_check_support(draft, problem):
    problems = check_support(draft, PASSAGES)
    if problem is None:
        assert problems == []
    else:
        assert problems and problem in problems[0]


def test_idk_detection():
    assert is_idk("I do not know.") and is_idk(" i do not know ")
    assert is_idk("No indexed project covers this question.")       # legacy prompt wording
    assert not is_idk("I do not know the exact number, but it is 81%.")


# --------------------------------------------------- service-level contract

def test_nothing_retrieved_says_i_do_not_know(kb):
    res = service.answer("quantum computing lab schedule", kb)
    assert res["status"] == "abstain" and res["answer"].startswith(IDK)
    assert res["grounding"]["reason"] == "retrieval_gate"


def test_related_sources_without_an_answer_say_i_do_not_know(kb):
    kb = dataclasses.replace(kb, scorer=StubScorer(-6.0))         # nothing answers the question
    res = service.answer("what accuracy did the crop disease model reach?", _with(kb, answer_threshold=0.0))
    assert res["status"] == "ok" and res["results"]               # sources found...
    assert res["answer"] == service.NO_ANSWER                     # ...but no answer invented
    assert res["answer_source"] is None
    assert res["grounding"]["verdict"] == "unknown" and res["grounding"]["reason"] == "answer_gate"


def test_verified_answer_carries_verbatim_evidence(kb):
    kb = dataclasses.replace(kb, scorer=StubScorer(6.0))
    res = service.answer("what accuracy does the khmer handwriting pipeline reach?",
                         _with(kb, answer_threshold=0.0))
    g = res["grounding"]
    assert g["verdict"] == "verified" and g["method"] == "verbatim"
    assert res["answer_source"]["chunk_id"] == g["evidence"][0]["chunk_id"]
    snippets = " ".join(" ".join(c["snippet"].split()) for c in res["results"])
    for ev in g["evidence"]:
        assert " ".join(ev["quote"].split()) in snippets           # checkable by anyone


def test_a_non_verbatim_extraction_is_never_served(kb, monkeypatch):
    def broken(question, cards, embedder, scorer=None):          # simulate an extraction bug
        c = cards[0]
        return {"text": "Invented sentence.", "score": 9.0, "scored_by": "cross", "card": c,
                "page": 1, "page_label": "p.1", "heading": None, "chunk_id": "x",
                "quotes": [{"quote": "Invented sentence.", "chunk_id": "x",
                            "chunk_text": c["snippet"]}]}
    monkeypatch.setattr(service, "extract_answer", broken)
    res = service.answer("khmer handwriting accuracy", kb)
    assert res["answer"] == service.NO_ANSWER and res["grounding"]["reason"] == "not_verbatim"


# ------------------------------------------------------------------ CLI view

def test_cli_never_shows_a_related_passage_as_the_answer(kb):
    kb = dataclasses.replace(kb, scorer=StubScorer(-6.0))
    q = "what accuracy did the crop disease model reach?"
    text = render_qa(q, service.answer(q, _with(kb, answer_threshold=0.0)))
    assert "Answer: I do not know." in text and "Related sources" in text
    assert "Rice leaf photographs" not in text                    # no chunk dumped as an answer


def test_cli_shows_how_an_answer_was_verified(kb):
    kb = dataclasses.replace(kb, scorer=StubScorer(6.0))
    q = "what accuracy does the khmer handwriting pipeline reach?"
    text = render_qa(q, service.answer(q, _with(kb, answer_threshold=0.0)))
    assert "Verified: quoted verbatim from dse2024-07__report.pdf::c000" in text
