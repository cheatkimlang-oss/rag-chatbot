"""Configurable retrieval: chunking, distance metric / HNSW, fusion, top-k, search mode."""
import dataclasses

import pytest
from qdrant_client import QdrantClient, models

from answer import service
from answer.service import KnowledgeBase
from config import Settings
from ingestion.chunker import ChunkConfig, build_chunks
from ingestion.parsers import parse_file
from ingestion.run import ingest
from retrieval.hybrid import SearchConfig, hybrid_retrieve
from store.db import KBStore
from store.vector_schema import index_config_of, to_similarity
from tests.conftest import FakeEmbedder

LONG = " ".join(f"Sentence {i} explains gradient descent, learning rates and momentum terms."
                for i in range(40))
TOPICS = {
    "ocr.md": "# OCR\n\nKhmer handwriting recognition with Tesseract and custom traineddata "
              "reaches 81 percent character accuracy on scanned forms.\n",
    "crops.md": "# Crops\n\nRice leaf photographs are classified into disease classes by a "
                "convolutional neural network running on a mobile phone.\n",
    "library.md": "# Library\n\nA web application for managing faculty library loans, "
                  "returns, overdue fines and catalogue search for students.\n",
}


# ---------------------------------------------------------------- settings

@pytest.mark.parametrize("kw,needle", [
    (dict(chunk_overlap=700), "chunk_overlap"),
    (dict(slide_chunk_size=0), "slide_chunk_size"),
    (dict(distance="hamming"), "distance"),
    (dict(chunk_unit="words"), "chunk_unit"),
    (dict(search_mode="semantic"), "search_mode"),
    (dict(fusion="max"), "fusion"),
    (dict(hnsw_ef=0), "hnsw_ef"),
    (dict(display_top_k=0), "display_top_k"),
    (dict(distance="manhattan"), "KB_ABSTAIN_THRESHOLD"),
])
def test_invalid_settings_fail_loudly(kw, needle):
    with pytest.raises(ValueError, match=needle):
        Settings(**kw)


def test_settings_read_from_env(monkeypatch):
    env = {"KB_CHUNK_UNIT": "tokens", "KB_CHUNK_SIZE": "256", "KB_CHUNK_OVERLAP": "32",
           "KB_DISTANCE": "Euclid", "KB_HNSW_M": "32", "KB_HNSW_EF_CONSTRUCT": "400",
           "KB_HNSW_EF": "128", "KB_EXACT_SEARCH": "true", "KB_SEARCH_MODE": "dense",
           "KB_FUSION": "dbsf", "KB_RRF_K": "20", "KB_DISPLAY_TOP_K": "8"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert s.chunk_config()["chunk_size"] == 256 and s.chunk_unit == "tokens"
    assert s.index_config() == {"distance": "euclid", "hnsw_m": 32, "hnsw_ef_construct": 400}
    cfg = SearchConfig.from_settings(s)
    assert (cfg.mode, cfg.fusion, cfg.rrf_k, cfg.hnsw_ef, cfg.exact) == ("dense", "dbsf", 20, 128, True)
    assert s.display_top_k == 8


def test_fusion_and_search_params():
    assert isinstance(SearchConfig().fusion_query(), models.RrfQuery)
    assert SearchConfig().fusion_query().rrf.k == 60                      # the spec's k, not Qdrant's 2
    assert isinstance(SearchConfig(rrf_k=0).fusion_query(), models.FusionQuery)
    assert SearchConfig(fusion="dbsf").fusion_query().fusion == models.Fusion.DBSF
    assert SearchConfig().params() is None
    assert SearchConfig(hnsw_ef=256, exact=True).params() == models.SearchParams(hnsw_ef=256, exact=True)


def test_to_similarity_maps_euclid_onto_cosine():
    # unit vectors at 60 degrees: cosine 0.5, euclidean distance 1.0
    assert to_similarity(1.0, "euclid") == pytest.approx(0.5)
    assert to_similarity(0.5, "cosine") == 0.5 and to_similarity(0.5, "dot") == 0.5
    assert to_similarity(0.0, "manhattan") > to_similarity(3.0, "manhattan")


# ----------------------------------------------------------------- chunking

def _long_doc(tmp_path):
    (tmp_path / "lecture.md").write_text(f"# Optimisation\n\n{LONG}\n", encoding="utf-8")
    return parse_file(tmp_path / "lecture.md", "d")


def _sizes(chunks):
    """Body length of each chunk, breadcrumb line excluded."""
    return [len(c.text.split("\n", 1)[1]) for c in chunks]


def test_chunk_size_and_overlap_are_honoured(tmp_path):
    parsed = _long_doc(tmp_path)
    project = {"project_id": "p", "title_en": "T", "year": 2024, "program": "DOC"}
    small = build_chunks(parsed, project, [], "report", ChunkConfig(size=300, overlap=0))
    large = build_chunks(parsed, project, [], "report", ChunkConfig(size=1200, overlap=0))
    assert len(small) > len(large) > 1
    assert max(_sizes(small)) <= 300 and max(_sizes(large)) <= 1200

    overlapped = build_chunks(parsed, project, [], "report", ChunkConfig(size=300, overlap=120))
    a, b = (c.text.split("\n", 1)[1] for c in overlapped[:2])
    assert a[-40:] in b                                             # the overlap is really shared


def test_token_mode_uses_the_embedder_tokenizer(tmp_path):
    emb = FakeEmbedder()
    parsed = _long_doc(tmp_path)
    cfg = ChunkConfig.from_settings(Settings(chunk_unit="tokens", chunk_size=60, chunk_overlap=10), emb)
    chunks = build_chunks(parsed, {"project_id": "p", "title_en": "T", "year": 2024, "program": "DOC"},
                          [], "report", cfg)
    assert chunks and all(emb.count_tokens(c.text.split("\n", 1)[1]) <= 60 for c in chunks)


def test_token_chunks_the_embedder_would_truncate_are_refused():
    with pytest.raises(ValueError, match="silently truncated"):
        ChunkConfig.from_settings(Settings(chunk_unit="tokens", chunk_size=500, chunk_overlap=50),
                                  FakeEmbedder())                    # 500 + breadcrumb > 512
    with pytest.raises(ValueError, match="tokenizer"):
        ChunkConfig.from_settings(Settings(chunk_unit="tokens", chunk_size=200), object())


# ------------------------------------------------- ingestion + vector index

def _corpus(tmp_path):
    c = tmp_path / "corpus"
    c.mkdir(exist_ok=True)
    for name, text in TOPICS.items():
        (c / name).write_text(text, encoding="utf-8")
    (c / "lecture.md").write_text(f"# Optimisation\n\n{LONG}\n", encoding="utf-8")
    return c


def _kb(tmp_path, name="kb", client=None, **kw):
    s = Settings(abstain_threshold=0.15, **kw)
    return KnowledgeBase(client or QdrantClient(":memory:"), FakeEmbedder(),
                         KBStore(tmp_path / f"{name}.sqlite3"), s)


def _ingest(kb, corpus, **kw):
    none = corpus.parent / "none.csv"
    return ingest(kb, corpus_dir=corpus, projects_csv=none, seniors_csv=none, **kw)


def test_changing_chunk_size_rechunks_on_next_run(tmp_path):
    corpus, client = _corpus(tmp_path), QdrantClient(":memory:")
    first = _ingest(_kb(tmp_path, client=client), corpus)
    assert _ingest(_kb(tmp_path, client=client), corpus)["points_written"] == 0
    rechunked = _ingest(_kb(tmp_path, client=client, chunk_size=300, chunk_overlap=30), corpus)
    assert rechunked["status"] == "ok" and rechunked["points_written"] > 0
    assert rechunked["points_total"] > first["points_total"]        # smaller chunks, more points
    assert rechunked["config"]["chunking"]["chunk_size"] == 300


@pytest.mark.parametrize("distance", ["cosine", "dot", "euclid"])
def test_every_metric_ranks_alike_and_gates_on_cosine_scale(tmp_path, distance):
    corpus = _corpus(tmp_path)
    base, other = _kb(tmp_path, "cos"), _kb(tmp_path, distance, distance=distance, hnsw_m=32,
                                             hnsw_ef_construct=128)
    assert _ingest(base, corpus)["status"] == "ok"
    rep = _ingest(other, corpus)
    assert rep["status"] == "ok", rep["errors"]
    assert index_config_of(other.client, other.settings.collection) == {
        "distance": distance, "hnsw_m": 32, "hnsw_ef_construct": 128}

    q = "rice leaf disease convolutional network"
    a, b = service.answer(q, base), service.answer(q, other)
    assert [c["project_id"] for c in a["results"]] == [c["project_id"] for c in b["results"]]
    assert b["gate_score"] == pytest.approx(a["gate_score"], abs=1e-4)   # same cosine scale


def test_changing_distance_needs_rebuild(tmp_path):
    corpus, client = _corpus(tmp_path), QdrantClient(":memory:")
    _ingest(_kb(tmp_path, client=client), corpus)
    euclid = _kb(tmp_path, client=client, distance="euclid")
    rep = _ingest(euclid, corpus)
    assert rep["status"] == "failed" and "--rebuild" in rep["errors"][0] and "euclid" in rep["errors"][0]
    assert any("--rebuild" in p for p in euclid.health())
    rebuilt = _ingest(euclid, corpus, rebuild=True)
    assert rebuilt["status"] == "ok" and euclid.health() == []


# ------------------------------------------------------- search-time knobs

@pytest.fixture
def indexed(tmp_path):
    kb = _kb(tmp_path)
    assert _ingest(kb, _corpus(tmp_path))["status"] == "ok"
    return kb


def _search(kb, q, **cfg):
    return hybrid_retrieve(kb.client, kb.embedder, q, collection=kb.settings.collection,
                           search=SearchConfig(**cfg))


def test_rrf_uses_k_60_not_qdrants_default_2(indexed):
    top = _search(indexed, "khmer handwriting tesseract")[0]
    assert top.score <= 2 / 60 + 1e-9                     # k=2 would score ~0.67
    assert _search(indexed, "khmer handwriting tesseract", rrf_k=0)[0].score > 0.3


@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_search_modes_agree_on_an_easy_query(indexed, mode):
    hits = _search(indexed, "library loans overdue fines", mode=mode)
    assert hits[0].project_id == "doc-library"
    assert hits[0].dense_score is not None                  # the gate works in every mode


def test_lexical_mode_with_no_keywords_returns_nothing(indexed, monkeypatch):
    monkeypatch.setattr(indexed.embedder, "encode_query",
                        lambda texts: ([[1.0] + [0.0] * 1023], [([], [])]), raising=False)
    assert _search(indexed, "anything", mode="lexical") == []


def test_exact_and_hnsw_ef_return_the_same_ranking(indexed):
    q = "momentum learning rate gradient descent"
    default = [h.project_id for h in _search(indexed, q)]
    assert [h.project_id for h in _search(indexed, q, exact=True)] == default
    assert [h.project_id for h in _search(indexed, q, hnsw_ef=512)] == default


def test_display_top_k_controls_page_size(tmp_path):
    kb = _kb(tmp_path, display_top_k=1)
    _ingest(kb, _corpus(tmp_path))
    res = service.answer("khmer handwriting rice leaf library loans gradient", kb)
    assert len(res["results"]) == 1 and res["has_more"]
    wide = dataclasses.replace(kb, settings=dataclasses.replace(kb.settings, display_top_k=4))
    assert len(service.answer("khmer handwriting rice leaf library loans gradient", wide)["results"]) > 1
