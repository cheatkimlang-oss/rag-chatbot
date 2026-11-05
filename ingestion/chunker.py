"""Stage 02 — structure-aware chunking with LangChain splitters (FR-03).

Rules: the abstract is one atomic chunk; references/bibliography/appendix are
dropped; slides split smaller and are non-primary; chunks < `min_chunk_chars` are
dropped; every chunk carries a breadcrumb so it is self-describing on its own.

Sizes come from `ChunkConfig` (Settings: KB_CHUNK_*), measured in characters
(default 700/100, slides 300/50) or in tokens of the embedder's own tokenizer.
Token mode refuses a chunk_size the embedder would silently truncate.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BODY_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
MIN_CHUNK_CHARS = 80
BREADCRUMB_TOKEN_BUDGET = 64       # "[Project: … | Author(s): … | p.4]" prefix, token mode


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 700
    overlap: int = 100
    slide_size: int = 300
    slide_overlap: int = 50
    min_chars: int = MIN_CHUNK_CHARS
    unit: str = "chars"                                   # chars | tokens
    length_function: Callable[[str], int] | None = None   # set for unit == "tokens"

    @classmethod
    def from_settings(cls, s, embedder=None) -> "ChunkConfig":
        length = None
        if s.chunk_unit == "tokens":
            count = getattr(embedder, "count_tokens", None)
            if count is None:
                raise ValueError(f"KB_CHUNK_UNIT=tokens needs an embedder with a tokenizer; "
                                 f"{type(embedder).__name__} has none")
            limit = getattr(embedder, "max_tokens", None)
            if limit and max(s.chunk_size, s.slide_chunk_size) + BREADCRUMB_TOKEN_BUDGET > limit:
                raise ValueError(
                    f"chunk_size {s.chunk_size} tokens + ~{BREADCRUMB_TOKEN_BUDGET} breadcrumb "
                    f"tokens exceeds the embedder's {limit}-token limit: the tail would be "
                    f"silently truncated. Use KB_CHUNK_SIZE <= {limit - BREADCRUMB_TOKEN_BUDGET}.")
            length = count
        return cls(size=s.chunk_size, overlap=s.chunk_overlap, slide_size=s.slide_chunk_size,
                   slide_overlap=s.slide_chunk_overlap, min_chars=s.min_chunk_chars,
                   unit=s.chunk_unit, length_function=length)

    def splitters(self) -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
        extra = {"length_function": self.length_function} if self.length_function else {}
        body = RecursiveCharacterTextSplitter(chunk_size=self.size, chunk_overlap=self.overlap,
                                              separators=BODY_SEPARATORS, **extra)
        slides = RecursiveCharacterTextSplitter(chunk_size=self.slide_size,
                                                chunk_overlap=self.slide_overlap, **extra)
        return body, slides
SECTIONS = ("abstract", "introduction", "related work", "literature review", "methodology",
            "methods", "results", "discussion", "conclusion", "references", "bibliography",
            "appendix")
DROP_SECTIONS = {"references", "bibliography", "appendix"}
# "3.", "2.1 ", "IV. " — roman numerals only when followed by "." or ")", so
# "introduction" keeps its leading "i".
_LEADING_NUM = re.compile(r"^(?:[\d.\s:)\-–]+|[ivx]+[.)]\s*)+", re.I)


@dataclass
class Chunk:
    chunk_id: str
    text: str          # breadcrumb + body
    metadata: dict

    def to_langchain_document(self) -> Document:
        return Document(page_content=self.text, metadata=self.metadata)


def section_of(headings: list[str]) -> str:
    """Canonical section name, or 'body'. 'Chapter 3: Methodology' -> 'methodology'."""
    for h in headings:
        hl = _LEADING_NUM.sub("", h.lower()).strip()
        hl = re.sub(r"^chapter\s+\w+[:.\s]*", "", hl)
        for s in SECTIONS:
            if hl.startswith(s):
                return s
    return "body"


def build_chunks(parsed, project: dict, authors: list[dict], doc_type: str,
                 config: ChunkConfig | None = None) -> list[Chunk]:
    """`project` = SQLite-shaped project row; `authors` = [{senior_id, full_name_latin, name_normalized}]."""
    config = config or ChunkConfig()
    body_splitter, slide_splitter = config.splitters()
    author_txt = ", ".join(a["full_name_latin"] for a in authors)
    names_norm = " / ".join(a["name_normalized"] for a in authors)
    senior_ids = [a["senior_id"] for a in authors]
    chunks: list[Chunk] = []
    i = 0
    section = "body"
    for page in parsed.pages:
        meta = page.metadata
        detected = section_of(meta.get("headings") or [])
        # PDF pages without a heading continue the previous section (a long
        # bibliography spans pages); DOCX/MD anchors always carry their own.
        if detected != "body" or meta.get("heading"):
            section = detected
        if section in DROP_SECTIONS:
            continue                                         # citation noise

        if section == "abstract":
            bodies = [page.page_content]                     # atomic
        elif doc_type == "slides":
            bodies = slide_splitter.split_text(page.page_content)
        else:
            bodies = body_splitter.split_text(page.page_content)

        for body in bodies:
            body = body.strip()
            if len(body) < config.min_chars:
                continue
            parts = [f"Project: {project['title_en']}"]
            if author_txt:
                parts += [f"Author(s): {author_txt}", f"Year: {project['year']}",
                          f"Program: {project['program']}"]
            else:
                parts.append(f"Source: {parsed.source_url or parsed.path.name}")
            parts += [f"Section: {meta.get('heading') or section}", meta["page_label"]]
            breadcrumb = "[" + " | ".join(parts) + "]"
            chunk_id = f"{parsed.doc_id}::c{i:03d}"
            chunks.append(Chunk(chunk_id=chunk_id, text=f"{breadcrumb}\n{body}", metadata={
                "project_id": project["project_id"],
                "title_en": project["title_en"],
                "senior_ids": senior_ids,
                "author_names_normalized": names_norm,
                "doc_id": parsed.doc_id,
                "doc_type": doc_type,
                "year": int(project["year"]),
                "program": project["program"],
                "supervisor_id": project.get("supervisor_id"),
                "domain": project.get("domain"),
                "tech_stack": project.get("tech_stack") or [],
                "section": section,
                "heading": meta.get("heading"),
                "page_number": meta["page_number"],
                "page_label": meta["page_label"],
                "chunk_index": i,
                "chunk_id": chunk_id,
                "breadcrumb": breadcrumb,
                "source_file": parsed.path.name,
                "source_url": parsed.source_url,              # web page / video, else None
                "start_seconds": meta.get("start_seconds"),   # transcripts only
                "is_primary_doc": doc_type == "report",
            }))
            i += 1
    return chunks
