"""Plane A end to end on a temp corpus: DOCX/MD drop-ins + one manifest project."""
import csv

import docx
import pytest
from qdrant_client import QdrantClient

from answer import service
from answer.service import KnowledgeBase
from config import Settings
from ingestion.chunker import section_of
from ingestion.parsers import parse_file
from ingestion.run import ingest
from store.db import KBStore
from tests.conftest import FakeEmbedder

BODY = ("Practical fluency in software design patterns, algorithmic optimization, data "
        "structures, and database management systems such as PostgreSQL and Qdrant.")


def _docx(path):
    d = docx.Document()
    d.add_paragraph().add_run("PROFILE OF A STUDENT").bold = True
    d.add_paragraph().add_run("1. Core Pillars").bold = True
    d.add_paragraph(BODY)
    d.add_paragraph().add_run("2. Toolkit").bold = True
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Category", "Tools"
    t.cell(1, 0).text, t.cell(1, 1).text = "Languages", "Python, R, SQL, C++, Bash shell scripting for automation"
    d.add_paragraph().add_run("References").bold = True
    d.add_paragraph("[1] Some citation that must never be indexed because references are noise.")
    d.save(path)


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    c = tmp_path / "corpus"
    (c / "dse2024-07").mkdir(parents=True)
    _docx(c / "Student_Profile.docx")
    (c / "notes.md").write_text("# Gardening\n\n## Tomatoes\n\nTomatoes need six hours of direct "
                                "sunlight, deep watering twice a week and well drained soil.\n",
                                encoding="utf-8")
    (c / "photo.jpg").write_bytes(b"\xff\xd8")
    (c / "dse2024-07" / "report.md").write_text(
        "# Abstract\n\nWe segment Khmer handwriting and run Tesseract OCR with custom traineddata, "
        "reaching 81% character accuracy on 500 scanned forms.\n", encoding="utf-8")
    _write_csv(tmp_path / "seniors.csv",
               ["senior_id", "full_name_latin", "full_name_khmer", "aliases", "program", "cohort_year",
                "consent_public_contact", "status"],
               [{"senior_id": "s-1", "full_name_latin": "Sok Dara", "full_name_khmer": "សុខ ដារា",
                 "aliases": "Dara Sok", "program": "DSE", "cohort_year": "2024",
                 "consent_public_contact": "no", "status": "active"}])
    _write_csv(tmp_path / "projects.csv",
               ["project_id", "title_en", "year", "program", "supervisor_id", "senior_ids",
                "abstract", "permission"],
               [{"project_id": "dse2024-07", "title_en": "Khmer Handwriting OCR", "year": "2024",
                 "program": "DSE", "supervisor_id": "sup-3", "senior_ids": "s-1",
                 "abstract": "Khmer OCR.", "permission": "granted"}])
    return tmp_path


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(QdrantClient(":memory:"), FakeEmbedder(), KBStore(tmp_path / "kb.sqlite3"),
                         Settings(abstain_threshold=0.15))


def _run(kb, corpus, **kw):
    return ingest(kb, corpus_dir=corpus / "corpus", projects_csv=corpus / "projects.csv",
                  seniors_csv=corpus / "seniors.csv", **kw)


def test_docx_sections_tables_and_title(tmp_path):
    _docx(tmp_path / "p.docx")
    parsed = parse_file(tmp_path / "p.docx", "d")
    assert parsed.parse_status == "ok" and parsed.title == "Profile Of A Student"
    labels = [(p.metadata["page_label"], p.metadata["heading"]) for p in parsed.pages]
    assert labels == [("§1", "1. Core Pillars"), ("§2", "2. Toolkit"), ("§3", "References")]
    assert "Languages | Python, R, SQL" in parsed.pages[1].page_content


@pytest.mark.parametrize("heading,section", [("1. Introduction", "introduction"),
                                             ("IV. Results", "results"),
                                             ("Chapter 3: Methodology", "methodology"),
                                             ("Tools we used", "body")])
def test_section_of(heading, section):
    assert section_of([heading]) == section


def test_ingest_end_to_end(kb, corpus):
    rep = _run(kb, corpus)
    assert rep["status"] == "ok", rep["errors"]
    assert any("photo.jpg" in w for w in rep["warnings"])            # unsupported is reported
    by_id = {d["doc_id"]: d for d in rep["documents"]}
    assert set(by_id) == {"doc-student-profile__Student_Profile.docx", "doc-notes__notes.md",
                          "dse2024-07__dse2024-07__report.md"}
    texts = [p.payload["text"] for p in kb.client.scroll(kb.settings.collection, limit=100)[0]]
    assert not any("must never be indexed" in t for t in texts)      # references dropped
    assert rep["points_total"] == sum(d["chunks"] for d in rep["documents"])

    res = service.answer("which languages and shell scripting tools are used?", kb)
    top = res["results"][0]
    assert top["title"] == "Profile Of A Student" and top["page_label"] == "§2"
    assert top["source"] == "Student_Profile.docx" and top["authors"] == []

    pf = service.answer("សុខ ដារា", kb)["portfolio"]                 # manifest identity, Khmer alias
    assert pf["projects"][0]["project_id"] == "dse2024-07" and pf["contact"] is None


def test_reingest_unchanged_is_idempotent(kb, corpus):
    first = _run(kb, corpus)
    second = _run(kb, corpus)
    assert second["points_written"] == 0
    assert second["points_total"] == first["points_total"]
    assert all(d.get("skipped") == "unchanged" for d in second["documents"])


def test_removed_file_is_deleted_from_index(kb, corpus):
    _run(kb, corpus)
    (corpus / "corpus" / "notes.md").unlink()
    rep = _run(kb, corpus)
    assert "doc-notes" not in kb.db.project_ids()
    assert rep["status"] == "ok" and service.answer("tomatoes sunlight watering", kb)["status"] == "abstain"


def test_session_picks_up_new_and_deleted_files(corpus, tmp_path, capsys):
    from cli import Refresher

    settings = Settings(abstain_threshold=0.15, corpus_dir=str(corpus / "corpus"),
                        projects_csv=str(corpus / "projects.csv"),
                        seniors_csv=str(corpus / "seniors.csv"))
    kb = KnowledgeBase(QdrantClient(":memory:"), FakeEmbedder(), KBStore(tmp_path / "s.sqlite3"), settings)
    refresh = Refresher(kb)
    refresh()                                              # session start: initial index
    question = "bananas ripen fridge"           # words only the new file contains

    def sources():
        res = service.answer(question, kb)
        return {c["source"] for c in res["results"] + res["weak_matches"]}

    assert "bananas.txt" not in sources()

    new = corpus / "corpus" / "bananas.txt"
    new.write_text("Bananas ripen at room temperature in two to five days; "
                   "keep them away from the fridge until they ripen.", encoding="utf-8")
    refresh()                                              # before the next question
    assert "indexed bananas.txt" in capsys.readouterr().out
    assert service.answer(question, kb)["results"][0]["source"] == "bananas.txt"

    refresh()                                              # nothing changed: no work, no output
    assert capsys.readouterr().out == ""

    new.unlink()
    refresh()
    assert "removed doc-bananas" in capsys.readouterr().out
    assert "bananas.txt" not in sources()


def test_unchanged_files_are_not_reparsed(kb, corpus, monkeypatch):
    import ingestion.run as run

    _run(kb, corpus)
    monkeypatch.setattr(run, "parse_file", lambda *a, **k: pytest.fail("re-parsed an unchanged file"))
    assert _run(kb, corpus)["points_written"] == 0


def test_parsed_documents_carry_fr02_metadata(tmp_path):
    _docx(tmp_path / "p.docx")
    meta = parse_file(tmp_path / "p.docx", "dse-x__p.docx", "dse-x").pages[0].metadata
    assert {"doc_id", "project_id", "page_number", "parser", "parse_status", "sha256"} <= set(meta)
    assert meta["project_id"] == "dse-x" and meta["parse_status"] == "ok" and len(meta["sha256"]) == 64


def test_takedown_removes_project_when_every_author_withdraws(kb, corpus):
    _run(kb, corpus)
    assert "dse2024-07" in kb.db.project_ids()
    with (corpus / "seniors.csv").open(encoding="utf-8-sig") as f:
        text = f.read()
    (corpus / "seniors.csv").write_text(text.replace(",active", ",withdrawn"), encoding="utf-8")
    rep = _run(kb, corpus)
    assert rep["status"] == "ok" and "dse2024-07" in rep["removed"]
    pts = kb.client.scroll(kb.settings.collection, limit=100)[0]
    assert not any(p.payload["project_id"] == "dse2024-07" for p in pts)
    assert service.portfolio("s-1", kb)["status"] == "not_found"


def test_feedback_becomes_a_training_pair(kb, corpus):
    from training_data.build_training_pairs import build_pairs

    _run(kb, corpus)
    res = service.answer("which languages and shell scripting tools are used?", kb)
    qid = service.log_interaction(kb, res, surface="cli")
    assert build_pairs(kb.db.conn) == []                      # no click / rating yet
    kb.db.rate_query(qid, 1, res["results"][0]["project_id"])
    (pair,) = build_pairs(kb.db.conn)
    assert pair["positive_id"] == "doc-student-profile" and pair["schema_version"] == 1
    assert "doc-student-profile" not in pair["hard_negatives"]
    kb.db.rate_query(qid, -1)
    assert build_pairs(kb.db.conn) == []                      # thumbs-down never trains


def test_invalid_manifest_writes_nothing(kb, corpus):
    _write_csv(corpus / "projects.csv", ["project_id", "title_en", "senior_ids", "permission"],
               [{"project_id": "x", "title_en": "X", "senior_ids": "s-404", "permission": "granted"}])
    rep = _run(kb, corpus)
    assert rep["status"] == "failed" and rep["errors"]
    assert not kb.client.collection_exists(kb.settings.collection)
