"""Shared fixtures: an in-memory Qdrant + SQLite KB seeded with a tiny corpus.

FakeEmbedder is a deterministic bag-of-words hasher standing in for bge-m3, so
the tests exercise the real Qdrant hybrid query path without a 2 GB model.
Absolute scores are NOT comparable to bge-m3; thresholds here are test-only.
"""
from __future__ import annotations

import math
import re
import sys
import uuid
import zlib
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer.service import KnowledgeBase  # noqa: E402
from config import Settings  # noqa: E402
from retrieval.intent import normalize  # noqa: E402
from store.db import KBStore  # noqa: E402
from store.vector_schema import COLLECTION, DENSE, LEXICAL, ensure_collection  # noqa: E402

DIM = 1024
NS = uuid.UUID("6f2a1c3e-0000-4000-8000-000000000000")


class FakeEmbedder:
    revision = "fake-bow"
    dim = DIM
    sparse_idf = False
    max_tokens = 512

    def count_tokens(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))

    def encode(self, texts):
        dense, sparse = [], []
        for text in texts:
            toks = re.findall(r"\w+", text.lower())
            vec = [0.0] * DIM
            weights: dict[int, float] = {}
            for t in toks:
                h = zlib.crc32(t.encode())
                vec[h % DIM] += 1.0
                weights[h % 1_000_003] = weights.get(h % 1_000_003, 0.0) + 1.0
            n = math.sqrt(sum(v * v for v in vec)) or 1.0
            dense.append([v / n for v in vec])
            sparse.append((list(weights), list(weights.values())))
        return dense, sparse


SENIORS = [
    dict(senior_id="s-2024-014", full_name_latin="Sok Dara", full_name_khmer="សុខ ដារា",
         program="DSE", cohort_year=2024, supervisor_id="sup-003",
         github="github.com/sokdara", consent_public_contact=True, consent_relay=True,
         telegram_user_hash="abc123"),
    dict(senior_id="s-2024-021", full_name_latin="Chan Sreyneang", program="DSE",
         cohort_year=2024, supervisor_id="sup-003", github="github.com/hidden",
         consent_public_contact=False),
    dict(senior_id="s-2023-005", full_name_latin="Kim Lang", program="ITE", cohort_year=2023,
         supervisor_id="sup-001", consent_relay=True),
    dict(senior_id="s-2022-030", full_name_latin="Long Visal", program="ITE", cohort_year=2022,
         supervisor_id="sup-001"),
    dict(senior_id="s-2023-099", full_name_latin="Meas Bopha", program="ITE", cohort_year=2023,
         status="withdrawn"),
]
ALIASES = {"s-2024-014": [("dara sok", "latin"), ("សុខ ដារា", "khmer")]}

PROJECTS = [
    (dict(project_id="dse2024-07", title_en="Khmer Handwritten Text Recognition with Tesseract",
          year=2024, program="DSE", supervisor_id="sup-003", permission="granted",
          domain="cv", tech_stack=["tesseract", "python"],
          abstract="We segment Khmer handwriting and run Tesseract with custom khm traineddata."),
     [("s-2024-014", "lead"), ("s-2024-021", "member")],
     [("abstract", 1, "We build a pipeline that segments Khmer handwriting and runs Tesseract OCR "
                      "with a custom khm traineddata, reaching 81% character accuracy on 500 scanned forms."),
      ("methodology", 4, "Each scanned page is binarised, lines are segmented, and the text "
                         "recognition model is fine tuned on Khmer handwritten characters.")]),
    (dict(project_id="dse2023-04", title_en="Khmer OCR Dataset Curation Tool", year=2023,
          program="DSE", supervisor_id="sup-003", permission="granted", domain="cv",
          abstract="A labelling tool for Khmer OCR training images."),
     [("s-2024-014", "lead")],
     [("abstract", 1, "A web tool to label, review and curate Khmer OCR training images, "
                      "exporting clean text recognition datasets for model training.")]),
    (dict(project_id="ite2023-11", title_en="Crop Disease Detection with Convolutional Networks",
          year=2023, program="ITE", supervisor_id="sup-001", permission="granted", domain="agri",
          abstract="Rice leaf disease classification with a CNN."),
     [("s-2023-005", "lead")],
     [("abstract", 1, "Rice leaf photographs from Takeo farms are classified into five disease "
                      "classes by a convolutional neural network running on a mobile phone.")]),
    (dict(project_id="ite2022-02", title_en="Library Management Web App", year=2022,
          program="ITE", supervisor_id="sup-001", permission="granted",
          abstract="A web app for managing the faculty library."),
     [("s-2022-030", "lead")],
     []),                                                     # metadata-only project
    (dict(project_id="ite2023-50", title_en="Withdrawn Senior Chatbot", year=2023,
          program="ITE", supervisor_id="sup-001", permission="pending",
          abstract="Should never appear."),
     [("s-2023-099", "lead")],
     [("abstract", 1, "A chatbot about khmer ocr that must never be shown because permission is pending.")]),
]


def seed(db: KBStore, client: QdrantClient, collection: str = COLLECTION) -> None:
    ensure_collection(client, collection, dim=DIM)
    emb = FakeEmbedder()
    for s in SENIORS:
        db.upsert_senior({**s, "name_normalized": normalize(s["full_name_latin"])},
                         ALIASES.get(s["senior_id"], []))
    names = {s["senior_id"]: s for s in SENIORS}
    points = []
    for proj, authors, chunks in PROJECTS:
        db.upsert_project(proj, authors)
        sids = [sid for sid, _ in authors]
        doc_id = f"{proj['project_id']}__report.pdf"
        db.upsert_document(dict(doc_id=doc_id, project_id=proj["project_id"], doc_type="report",
                                file_path=doc_id, sha256="0" * 64, page_count=10,
                                parser="pymupdf", parse_status="ok", chunk_count=len(chunks)))
        for i, (section, page, body) in enumerate(chunks):
            authors_txt = ", ".join(names[sid]["full_name_latin"] for sid in sids)
            crumb = (f"[Project: {proj['title_en']} | Author(s): {authors_txt} | Year: {proj['year']} | "
                     f"Program: {proj['program']} | Section: {section} | p.{page}]")
            text = f"{crumb}\n{body}"
            chunk_id = f"{doc_id}::c{i:03d}"
            (d,), ((idx, val),) = emb.encode([text])
            points.append(models.PointStruct(
                id=str(uuid.uuid5(NS, chunk_id)),
                vector={DENSE: d, LEXICAL: models.SparseVector(indices=idx, values=val)},
                payload=dict(project_id=proj["project_id"], title_en=proj["title_en"],
                             senior_ids=sids,
                             author_names_normalized=" / ".join(normalize(names[s]["full_name_latin"])
                                                                for s in sids),
                             doc_id=doc_id, doc_type="report", year=proj["year"],
                             program=proj["program"], supervisor_id=proj["supervisor_id"],
                             domain=proj.get("domain"), tech_stack=proj.get("tech_stack", []),
                             section=section, page_number=page, chunk_index=i, chunk_id=chunk_id,
                             breadcrumb=crumb, text=text, is_primary_doc=True),
            ))
    client.upsert(collection, points=points, wait=True)


@pytest.fixture
def kb(tmp_path):
    client = QdrantClient(":memory:")
    db = KBStore(tmp_path / "kb.sqlite3")
    seed(db, client)
    # Both gates on the FakeEmbedder's bag-of-words scale (not bge-m3 / cross-encoder).
    settings = Settings(abstain_threshold=0.15, answer_threshold=0.3,
                        sqlite_path=str(tmp_path / "kb.sqlite3"))
    return KnowledgeBase(client=client, embedder=FakeEmbedder(), db=db, settings=settings)
