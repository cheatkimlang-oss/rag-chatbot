from types import SimpleNamespace

from qdrant_client import models

from retrieval.filters import build_filter
from retrieval.hybrid import group_by_project, hybrid_retrieve
from retrieval.langchain_retriever import SeniorProjectKBRetriever


def _pt(pid, score, cid):
    return SimpleNamespace(id=cid, score=score, payload={"project_id": pid, "chunk_id": cid})


def test_grouping_returns_distinct_projects_by_best_score():
    fused = [_pt("a", 0.9, "a1"), _pt("a", 0.8, "a2"), _pt("b", 0.7, "b1"),
             _pt("a", 0.6, "a3"), _pt("c", 0.5, "c1")]
    dense = [_pt("b", 0.61, "b1"), _pt("a", 0.55, "a1")]
    hits = group_by_project(fused, dense, [_pt("a", 3.0, "a1")], top_k=2)
    assert [h.project_id for h in hits] == ["a", "b"]
    assert hits[0].chunk_ids == ["a1", "a2", "a3"]
    assert hits[0].dense_score == 0.55 and hits[0].matched_by == ["dense", "lexical"]
    assert hits[1].matched_by == ["dense"]


def test_hybrid_topic_query_ranks_ocr_projects(kb):
    hits = hybrid_retrieve(kb.client, kb.embedder, "khmer ocr text recognition")
    ids = [h.project_id for h in hits]
    assert len(ids) == len(set(ids))
    assert set(ids[:2]) == {"dse2024-07", "dse2023-04"}
    assert "dense" in hits[0].matched_by and "lexical" in hits[0].matched_by


def test_payload_prefilter(kb):
    flt = build_filter({"year": [2023]})
    hits = hybrid_retrieve(kb.client, kb.embedder, "khmer ocr", flt=flt)
    assert hits and {h.payload["year"] for h in hits} == {2023}


def test_build_filter_shapes():
    assert build_filter({}) is None
    f = build_filter({"year": ["2024"], "program": ["DSE", "ITE"]})
    assert f.must[0].match == models.MatchValue(value=2024)
    assert f.must[1].match == models.MatchAny(any=["DSE", "ITE"])


def test_langchain_retriever_returns_documents(kb):
    retriever = SeniorProjectKBRetriever(client=kb.client, embedder=kb.embedder, top_k=2)
    docs = retriever.invoke("khmer ocr")
    assert len(docs) == 2
    assert len({d.metadata["project_id"] for d in docs}) == 2
    assert docs[0].page_content.startswith("[Project:")
