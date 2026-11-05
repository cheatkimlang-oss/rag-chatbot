"""Stage 01 — document loading into page-anchored LangChain `Document`s (FR-02).

Every parser returns a `ParsedDocument` whose `pages` are LangChain Documents:
    page_content  the text of one anchor unit (PDF page, DOCX/MD/HTML section,
                  slide, transcript time window)
    metadata      page_number, page_label ("p.4" | "§2" | "slide 3" |
                  "00:04:00–00:06:00"), heading, headings

Source types -> one unified index:
    PDF reports / theses          .pdf            page anchors (+ OCR fallback)
    Word documents                .docx           heading-delimited sections
    slide presentations           .pptx           one anchor per slide (+ notes)
    raw lecture notes             .md .txt        markdown header sections
    university web pages          .html .htm      heading sections; the source URL
                                                  travels with every chunk
    video / lecture transcripts   .vtt .srt       fixed time windows

Anchors are non-negotiable: without them there are no citations. A document that
yields no text is `parse_status = "failed"` — reported, never indexed as empty.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".pptx", ".md", ".markdown", ".txt",
             ".html", ".htm", ".vtt", ".srt"}
# Bump when parsing/chunking output changes: unchanged files are then re-indexed.
PARSER_VERSION = 3
OCR_MIN_CHARS = 20
BULLET_LINE = re.compile(r"^[ \t]*([•▪◦●○■□➢►\-–*])[ \t]*\n[ \t]*", re.M)


@dataclass
class ParsedDocument:
    doc_id: str
    path: Path
    pages: list[Document]
    parse_status: str             # ok | ocr | failed
    parser: str
    page_count: int
    sha256: str
    title: str | None = None      # best-effort title from the file itself
    errors: list[str] = field(default_factory=list)
    source_url: str | None = None  # web page / video URL, when the file records one


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _page(text: str, number: int, label: str, heading: str | None = None,
          headings: list[str] | None = None) -> Document:
    return Document(page_content=text, metadata={
        "page_number": number, "page_label": label, "heading": heading,
        "headings": headings or ([heading] if heading else []),
    })


# --------------------------------------------------------------------- PDF

def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(img, lang="eng+khm")
    except Exception as exc:                  # tesseract binary / khm traineddata missing
        log.warning("OCR failed: %s", exc)
        return ""


def _pdf_headings(page) -> list[str]:
    """Cheap heuristic: lines noticeably larger than the body font."""
    blocks = page.get_text("dict")["blocks"]
    sizes = [s["size"] for b in blocks for ln in b.get("lines", []) for s in ln["spans"]]
    if not sizes:
        return []
    body = max(set(sizes), key=sizes.count)
    return [" ".join(s["text"] for s in ln["spans"]).strip()
            for b in blocks for ln in b.get("lines", [])
            if ln["spans"] and ln["spans"][0]["size"] > body * 1.15
            and " ".join(s["text"] for s in ln["spans"]).strip()]


def parse_pdf(path: Path) -> tuple[list[Document], str, int, str | None]:
    import pymupdf

    doc = pymupdf.open(path)
    pages, used_ocr, title = [], False, (doc.metadata or {}).get("title") or None
    for idx, page in enumerate(doc, start=1):
        # PyMuPDF emits a bullet glyph as its own line; glue it back onto its item.
        text = BULLET_LINE.sub(r"\1 ", page.get_text("text")).strip()
        if len(text) < OCR_MIN_CHARS:          # scanned page -> OCR
            text = _ocr_page(page).strip()
            used_ocr = used_ocr or bool(text)
        if not text:
            continue                           # blank page: never emit empty text
        heads = _pdf_headings(page)
        pages.append(_page(text, idx, f"p.{idx}", heads[0] if heads else None, heads))
    return pages, "pymupdf+ocr" if used_ocr else "pymupdf", doc.page_count, title


# -------------------------------------------------------------------- DOCX

def _is_docx_heading(par) -> bool:
    style = (par.style.name or "").lower() if par.style is not None else ""
    if style.startswith(("heading", "title")):
        return True
    text = par.text.strip()
    runs = [r for r in par.runs if r.text.strip()]
    # Many reports fake headings as short, fully-bold Normal paragraphs.
    return (bool(runs) and len(text) <= 120 and "list" not in style
            and all(r.bold for r in runs) and not text.endswith((".", ":")))


def _docx_blocks(d):
    """Paragraphs and tables in body order (python-docx lists them separately)."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, d)
        elif tag == "tbl":
            yield Table(child, d)


def parse_docx(path: Path) -> tuple[list[Document], str, int, str | None]:
    """One anchor per heading-delimited section, labelled §N (DOCX has no stable pages)."""
    import docx
    from docx.table import Table

    d = docx.Document(str(path))
    title = (d.core_properties.title or "").strip() or None
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for block in _docx_blocks(d):
        if isinstance(block, Table):
            rows = []
            for row in block.rows:
                cells = []
                for c in row.cells:                    # merged cells repeat; keep one copy
                    t = " ".join(c.text.split())
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    rows.append(" | ".join(cells))
            sections[-1][1].append("\n".join(rows))
            continue
        text = block.text.strip()
        if not text:
            continue
        if _is_docx_heading(block):
            if title is None:
                title = text                          # first heading = document title
            sections.append((text, []))
        else:
            style = (block.style.name or "").lower() if block.style is not None else ""
            sections[-1][1].append(f"- {text}" if "list" in style else text)

    pages, n = [], 0
    for heading, body in sections:
        text = "\n\n".join(body).strip()
        if not text:
            continue
        n += 1
        full = f"{heading}\n\n{text}" if heading else text
        pages.append(_page(full, n, f"§{n}", heading))
    return pages, "python-docx", n, title


# -------------------------------------------------------------------- PPTX

def parse_pptx(path: Path) -> tuple[list[Document], str, int, str | None]:
    from pptx import Presentation

    prs, pages, title = Presentation(str(path)), [], None
    for idx, slide in enumerate(prs.slides, start=1):
        parts, heading = [], None
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            heading = slide.shapes.title.text_frame.text.strip() or None
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
            elif getattr(shape, "has_table", False) and shape.has_table:
                parts += [" | ".join(c.text.strip() for c in r.cells if c.text.strip())
                          for r in shape.table.rows]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append(slide.notes_slide.notes_text_frame.text.strip())
        text = "\n".join(p for p in parts if p)
        if text:
            title = title or heading
            pages.append(_page(text, idx, f"slide {idx}", heading))
    return pages, "python-pptx", len(prs.slides), title


# --------------------------------------------------------------- MD / TXT

def parse_text(path: Path) -> tuple[list[Document], str, int, str | None]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".txt":
        return ([_page(raw.strip(), 1, "p.1")] if raw.strip() else []), "text", 1, None

    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    pages, title = [], None
    for n, part in enumerate((p for p in splitter.split_text(raw) if p.page_content.strip()), 1):
        heads = [part.metadata[k] for k in ("h1", "h2", "h3") if k in part.metadata]
        title = title or part.metadata.get("h1")
        pages.append(_page(part.page_content.strip(), n, f"§{n}", heads[-1] if heads else None, heads))
    return pages, "markdown", len(pages), title


# -------------------------------------------------------------------- HTML
# University pages are saved to disk first (ingestion/fetch_web.py), so
# ingestion stays offline, hashable and idempotent. Page chrome (menus, footers,
# scripts, cookie banners) is dropped; when the page marks its content with
# <main>/<article>, only that region is kept.

_HTML_SKIP = {"script", "style", "noscript", "template", "svg", "nav", "footer",
              "aside", "form", "iframe", "button", "select", "dialog"}
_HTML_MAIN = {"main", "article"}
_HTML_HEADINGS = {"h1", "h2", "h3"}
_HTML_BLOCKS = {"p", "div", "section", "li", "tr", "table", "ul", "ol", "blockquote",
                "pre", "dd", "dt", "figcaption", "h4", "h5", "h6", "br", "hr", "header"}


class _HTMLText(HTMLParser):
    """Flattens HTML into ordered (kind, text, in_main) items: kind = heading | text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, str, bool]] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self._buf: list[str] = []
        self._skip = 0           # depth inside page chrome
        self._header = 0         # depth inside <header> outside main content
        self._main = 0           # depth inside <main>/<article>
        self._heading = False
        self._in_title = False

    def _flush(self, kind: str = "text") -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text and text != "|":
            self.items.append((kind, text.strip(" |"), self._main > 0))

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("name") or a.get("property") or "").lower()
            if key and a.get("content"):
                self.meta[key] = a["content"].strip()
            return
        if tag == "link" and "canonical" in a.get("rel", "").lower() and a.get("href"):
            self.meta.setdefault("canonical", a["href"].strip())
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _HTML_SKIP:
            self._skip += 1
            return
        if tag == "header" and not self._main:      # site banner; an article's own header stays
            self._header += 1
            return
        if self._skip or self._header:
            return
        if tag in _HTML_MAIN:
            self._flush()
            self._main += 1
        elif tag in _HTML_HEADINGS:
            self._flush()
            self._heading = True
        elif tag in _HTML_BLOCKS:
            self._flush()
            if tag == "li":
                self._buf.append("- ")
        elif tag in ("td", "th"):
            self._buf.append(" | ")
        elif tag == "img" and a.get("alt", "").strip():
            self._buf.append(f" {a['alt'].strip()} ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in _HTML_SKIP:                        # <iframe/> etc. never get an end tag
            self._skip -= 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            return
        if tag in _HTML_SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "header" and self._header:
            self._header -= 1
            return
        if self._skip or self._header:
            return
        if tag in _HTML_HEADINGS and self._heading:
            self._flush("heading")
            self._heading = False
        elif tag in _HTML_MAIN:
            self._flush()
            self._main = max(0, self._main - 1)
        elif tag in _HTML_BLOCKS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and not self._header:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush("heading" if self._heading else "text")


def _decode_html(raw: bytes) -> str:
    m = re.search(rb"""<meta[^>]+charset=["']?([\w-]+)""", raw[:4096], re.I)
    for enc in ([m.group(1).decode("ascii", "ignore")] if m else []) + ["utf-8"]:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def read_html(path: Path) -> tuple[list[tuple[str, list[str]]], str | None, str | None]:
    """(sections [(heading, paragraphs)], title, source_url) of a saved web page."""
    return read_html_bytes(path.read_bytes())


def read_html_bytes(raw: bytes) -> tuple[list[tuple[str, list[str]]], str | None, str | None]:
    p = _HTMLText()
    p.feed(_decode_html(raw))
    p.close()
    items = p.items
    if any(in_main for _, _, in_main in items):      # page marks its content: keep only that
        items = [it for it in items if it[2]]
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    first_h1 = None
    for kind, text, _ in items:
        if kind == "heading":
            first_h1 = first_h1 or text
            sections.append((text, []))
        elif not sections[-1][1] or sections[-1][1][-1] != text:   # repeated menu lines
            sections[-1][1].append(text)
    title = (p.meta.get("og:title") or " ".join(p.title.split()) or first_h1 or None)
    url = p.meta.get("source-url") or p.meta.get("og:url") or p.meta.get("canonical")
    return [s for s in sections if s[1]], title, url


def parse_html(path: Path) -> tuple[list[Document], str, int, str | None]:
    sections, title, _ = read_html(path)
    pages = []
    for n, (heading, body) in enumerate(sections, 1):
        text = "\n".join(body)
        pages.append(_page(f"{heading}\n\n{text}" if heading else text, n, f"§{n}", heading))
    return pages, "html", len(pages), title


# -------------------------------------------------------------- transcripts
# WebVTT / SubRip captions (YouTube, Zoom, Teams, Whisper output). Cues are merged
# into fixed time windows: one window = one citable anchor ("00:04:00–00:06:00").

TRANSCRIPT_WINDOW_SECONDS = 120
_CUE_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
                       r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_TAG = re.compile(r"<[^>]+>")                    # <v Speaker>, <c>, <00:00:01.000>
_VTT_URL = re.compile(r"^NOTE\s+source[-_ ]?url\s*:\s*(\S+)", re.I | re.M)


def _secs(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def read_cues(raw: str) -> list[tuple[float, float, str]]:
    """(start, end, line) per caption line, with rolling auto-caption repeats removed."""
    cues, last = [], None
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = block.strip().split("\n")
        at = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        m = _CUE_TIME.search(lines[at]) if at is not None else None
        if not m:
            continue                                  # WEBVTT header, NOTE, STYLE, REGION
        start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
        for ln in lines[at + 1:]:
            text = " ".join(_CUE_TAG.sub("", ln).split())
            if text and text != last:                 # YouTube repeats the previous line
                cues.append((start, end, text))
                last = text
    return cues


def parse_transcript(path: Path) -> tuple[list[Document], str, int, str | None]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    windows: dict[int, list[tuple[float, float, str]]] = {}
    for cue in read_cues(raw):
        windows.setdefault(int(cue[0] // TRANSCRIPT_WINDOW_SECONDS), []).append(cue)
    pages = []
    for n, (_, cues) in enumerate(sorted(windows.items()), 1):
        start, end = cues[0][0], max(c[1] for c in cues)
        page = _page(" ".join(c[2] for c in cues), n, f"{_clock(start)}–{_clock(end)}")
        page.metadata["start_seconds"] = int(start)   # lets a card deep-link into the video
        pages.append(page)
    return pages, "vtt" if path.suffix.lower() == ".vtt" else "srt", len(pages), None


def _source_url(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return read_html(path)[2]
    if suffix == ".vtt":
        m = _VTT_URL.search(path.read_text(encoding="utf-8-sig", errors="replace")[:4096])
        return m.group(1) if m else None
    return None


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".pptx": parse_pptx,
           ".md": parse_text, ".markdown": parse_text, ".txt": parse_text,
           ".html": parse_html, ".htm": parse_html,
           ".vtt": parse_transcript, ".srt": parse_transcript}


def parse_file(path: Path, doc_id: str, project_id: str | None = None) -> ParsedDocument:
    digest = sha256_of(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        return ParsedDocument(doc_id, path, [], "failed", "none", 0, digest,
                              errors=[f"unsupported file type {path.suffix!r}"])
    try:
        pages, name, count, title = parser(path)
        url = _source_url(path)
    except Exception as exc:                   # corrupt / password-protected file
        log.warning("parse failed for %s: %s", path, exc)
        return ParsedDocument(doc_id, path, [], "failed", parser.__name__, 0, digest,
                              errors=[f"{type(exc).__name__}: {exc}"])
    status = "failed" if not pages else ("ocr" if name.endswith("+ocr") else "ok")
    errors = [] if pages else ["no extractable text"]
    for page in pages:                          # FR-02: the standard Document metadata
        page.metadata.update(doc_id=doc_id, project_id=project_id, parser=name,
                             parse_status=status, sha256=digest, source=path.name,
                             source_url=url)
    return ParsedDocument(doc_id, path, pages, status, name, count, digest,
                          title=_clean_title(title), errors=errors, source_url=url)


def _clean_title(t: str | None) -> str | None:
    if not t:
        return None
    t = " ".join(t.split())
    return t.title() if t.isupper() else t
