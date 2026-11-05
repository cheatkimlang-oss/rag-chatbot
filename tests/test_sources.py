"""Web pages + video transcripts join PDFs/DOCX/PPTX/notes in the one unified index."""
import httpx
import pytest
from qdrant_client import QdrantClient

from answer import service
from answer.service import KnowledgeBase
from config import Settings
from ingestion.fetch_web import fetch, read_url_list, snapshot_name
from ingestion.parsers import parse_file, read_cues
from ingestion.run import ingest
from store.db import KBStore
from tests.conftest import FakeEmbedder

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>Admissions | RUPP</title><script>var tracking = "never index me";</script></head>
<body>
<header><nav><a href="/">Home</a> <a href="/about">About</a></nav></header>
<main>
  <article>
    <header><h1>Faculty of Engineering Admissions</h1></header>
    <p>Applicants to the Data Science and Engineering program sit a mathematics entrance
       examination in August and submit their high school transcript.</p>
    <h2>Scholarships</h2>
    <ul><li>Merit scholarships cover full tuition for the top ten applicants.</li>
        <li>Need-based grants are reviewed by the faculty committee each semester.</li></ul>
    <table><tr><th>Program</th><th>Seats</th></tr><tr><td>DSE</td><td>120</td></tr></table>
  </article>
</main>
<footer>Copyright RUPP. Follow us on social media for updates.</footer>
</body></html>"""

VTT = """WEBVTT
NOTE source-url: https://youtu.be/abc123

00:00:01.000 --> 00:00:04.000
<v Lecturer>Welcome to lecture three on gradient descent.

00:00:04.000 --> 00:00:08.000
Welcome to lecture three on gradient descent.
The learning rate controls the size of each update step.

00:02:05.000 --> 00:02:09.500
Momentum accumulates past gradients to dampen oscillation in narrow valleys
of the loss surface, which speeds up convergence.
"""

SRT = """1
00:00:00,500 --> 00:00:03,000
Backpropagation applies the chain rule layer by layer.

2
00:00:03,000 --> 00:00:06,000
Each weight receives the gradient of the loss with respect to it.
"""


def test_html_keeps_content_drops_chrome(tmp_path):
    (tmp_path / "p.html").write_bytes(PAGE)
    parsed = parse_file(tmp_path / "p.html", "d")
    assert parsed.parse_status == "ok" and parsed.title == "Admissions | RUPP"
    text = "\n".join(p.page_content for p in parsed.pages)
    assert "mathematics entrance" in text and "Merit scholarships" in text
    assert "Program | Seats" in text and "DSE | 120" in text          # tables survive
    for chrome in ("never index me", "Home", "Copyright", "social media"):
        assert chrome not in text
    assert [p.metadata["heading"] for p in parsed.pages] == [
        "Faculty of Engineering Admissions", "Scholarships"]


def test_html_source_url_comes_from_meta(tmp_path):
    (tmp_path / "p.html").write_bytes(PAGE.replace(
        b"<head>", b'<head><meta name="source-url" content="https://www.rupp.edu.kh/fe">'))
    parsed = parse_file(tmp_path / "p.html", "d")
    assert parsed.source_url == "https://www.rupp.edu.kh/fe"
    assert parsed.pages[0].metadata["source_url"] == "https://www.rupp.edu.kh/fe"


def test_vtt_rolling_captions_are_deduplicated_and_windowed(tmp_path):
    (tmp_path / "lec3.vtt").write_text(VTT, encoding="utf-8")
    parsed = parse_file(tmp_path / "lec3.vtt", "d")
    assert parsed.parse_status == "ok" and parsed.source_url == "https://youtu.be/abc123"
    first, second = parsed.pages                               # 0–2 min, 2–4 min windows
    assert first.page_content.count("Welcome to lecture three") == 1
    assert "<v" not in first.page_content
    assert first.metadata["page_label"] == "00:00:01–00:00:08"
    assert second.metadata["page_label"].startswith("00:02:05")
    assert second.metadata["start_seconds"] == 125


def test_srt_parses():
    cues = read_cues(SRT)
    assert [c[0] for c in cues] == [0.5, 3.0] and cues[1][2].startswith("Each weight")


# ------------------------------------------------------------------ fetcher

def _mock(pages: dict[str, tuple[int, str, bytes]]):
    def handler(req: httpx.Request):
        status, ctype, body = pages.get(str(req.url), (404, "text/plain", b""))
        return httpx.Response(status, headers={"content-type": ctype}, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_snapshots_respects_robots_and_is_idempotent(tmp_path):
    site = {
        "https://uni.test/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /private"),
        "https://uni.test/fe": (200, "text/html; charset=utf-8", PAGE),
        "https://uni.test/handbook.pdf": (200, "application/pdf", b"%PDF"),
    }
    urls = ["https://uni.test/fe", "https://uni.test/private/x", "https://uni.test/handbook.pdf"]
    with _mock(site) as client:
        rows = fetch(urls, tmp_path, client=client, delay=0)
    assert [r["status"].split(":")[0] for r in rows] == ["new", "skipped", "skipped"]
    saved = tmp_path / snapshot_name("https://uni.test/fe")
    assert parse_file(saved, "d").source_url == "https://uni.test/fe"

    # A markup-only change (e.g. a rotating token in <head>) is not a content change.
    site["https://uni.test/fe"] = (200, "text/html", PAGE.replace(b"<title>", b"<meta name='csrf' content='z'><title>"))
    with _mock(site) as client:
        assert fetch(urls[:1], tmp_path, client=client, delay=0)[0]["status"] == "unchanged"
    site["https://uni.test/fe"] = (200, "text/html", PAGE.replace(b"August", b"July"))
    with _mock(site) as client:
        assert fetch(urls[:1], tmp_path, client=client, delay=0)[0]["status"] == "updated"


def test_url_list_skips_comments_and_duplicates(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("# header\nhttps://a.test/x  # inline\n\nhttps://a.test/x\nhttps://b.test/\n", encoding="utf-8")
    assert read_url_list(f) == ["https://a.test/x", "https://b.test/"]


# ------------------------------------------------------------- unified index

@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(QdrantClient(":memory:"), FakeEmbedder(), KBStore(tmp_path / "kb.sqlite3"),
                         Settings(abstain_threshold=0.15))


def test_all_source_types_land_in_one_index(kb, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "web-rupp-fe.html").write_bytes(PAGE.replace(
        b"<head>", b'<head><meta name="source-url" content="https://www.rupp.edu.kh/fe">'))
    (corpus / "lecture3.vtt").write_text(VTT, encoding="utf-8")
    (corpus / "lecture4.srt").write_text(SRT, encoding="utf-8")
    (corpus / "notes.md").write_text("# Week 1\n\nRaw lecture notes: linear regression minimises "
                                     "the squared error between predictions and labels.\n",
                                     encoding="utf-8")
    rep = ingest(kb, corpus_dir=corpus, projects_csv=tmp_path / "none.csv",
                 seniors_csv=tmp_path / "none.csv")
    assert rep["status"] == "ok", rep["errors"]
    assert all(d["chunks"] > 0 for d in rep["documents"]), rep["documents"]
    types = {p.payload["doc_type"] for p in kb.client.scroll(kb.settings.collection, limit=100)[0]}
    assert types == {"webpage", "transcript", "notes"}

    web = service.answer("merit scholarships tuition applicants", kb)["results"][0]
    assert web["source"] == "https://www.rupp.edu.kh/fe"              # citation = the URL
    video = service.answer("momentum past gradients oscillation valleys", kb)["results"][0]
    assert video["source"] == "https://youtu.be/abc123" and video["page_label"].startswith("00:02:05")

    again = ingest(kb, corpus_dir=corpus, projects_csv=tmp_path / "none.csv",
                   seniors_csv=tmp_path / "none.csv")
    assert again["points_written"] == 0                               # idempotent
