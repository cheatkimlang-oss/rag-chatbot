"""Stage 00 — manifest validation (FR-01) and source discovery.

Two kinds of sources:
  * manifest projects — a row in projects.csv + files in data/corpus/<project_id>/.
    Identity (authors, year, supervisor) comes from the CSV, never from the PDF.
  * drop-in documents — any supported file placed directly in data/corpus/.
    Searchable by topic; no authors, so no portfolio links.

`validate()` is pure: if the report has errors, ingestion writes nothing.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ingestion.parsers import SUPPORTED
from retrieval.intent import contains_khmer, normalize

REQUIRED_PROJECT = ["project_id", "title_en", "year", "program", "supervisor_id",
                    "senior_ids", "abstract", "permission"]
REQUIRED_SENIOR = ["senior_id", "full_name_latin", "program", "cohort_year"]
DROP_IN_PROGRAM = "DOC"
DOC_TYPES = {"slides": "slides", "poster": "poster", "notes": "notes"}


@dataclass
class Source:
    project: dict                       # SQLite-shaped project row
    authors: list[tuple[str, str | None]]
    files: list[tuple[Path, str]]       # (path, doc_type)
    drop_in: bool = False


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corpus_hash: str = ""
    seniors: list[dict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # utf-8-sig: Excel and PowerShell's Out-File both write a BOM.
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def _split(value: str) -> list[str]:
    return [s.strip() for s in (value or "").split(";") if s.strip()]


def _bool(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def corpus_hash(corpus_dir: Path) -> str:
    """Order-independent hash of file contents -> re-runs are provably idempotent."""
    h = hashlib.sha256()
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(corpus_dir).as_posix().encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def doc_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        return "slides"
    if suffix in (".html", ".htm"):
        return "webpage"
    if suffix in (".vtt", ".srt"):
        return "transcript"
    for key, value in DOC_TYPES.items():
        if key in path.stem.lower():
            return value
    return "report"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "doc"


def senior_row(r: dict) -> tuple[dict, list[tuple[str, str]]]:
    row = {
        "senior_id": r["senior_id"], "full_name_latin": r["full_name_latin"],
        "full_name_khmer": r.get("full_name_khmer") or None,
        "name_normalized": normalize(r["full_name_latin"]),
        "program": r["program"].upper(), "cohort_year": int(r["cohort_year"]),
        "supervisor_id": r.get("supervisor_id") or None,
        "contact_email": r.get("contact_email") or None, "github": r.get("github") or None,
        "linkedin": r.get("linkedin") or None, "website": r.get("website") or None,
        "consent_public_contact": _bool(r.get("consent_public_contact")),
        "consent_relay": _bool(r.get("consent_relay")),
        "status": (r.get("status") or "active").lower(),
    }
    aliases = {normalize(a) for a in _split(r.get("aliases", ""))}
    if row["full_name_khmer"]:
        aliases.add(normalize(row["full_name_khmer"]))
    aliases.discard(row["name_normalized"])
    return row, [(a, "khmer" if contains_khmer(a) else "latin") for a in sorted(aliases)]


def validate(projects_csv: Path, seniors_csv: Path, corpus_dir: Path) -> ValidationReport:
    rep = ValidationReport(corpus_hash=corpus_hash(corpus_dir) if corpus_dir.exists() else "")
    if not corpus_dir.exists():
        rep.errors.append(f"corpus folder {corpus_dir} does not exist")
        return rep

    seniors: dict[str, dict] = {}
    for line, r in enumerate(_read_csv(seniors_csv), start=2):
        missing = [c for c in REQUIRED_SENIOR if not r.get(c)]
        if missing:
            rep.errors.append(f"seniors.csv line {line}: missing {', '.join(missing)}")
            continue
        if r["senior_id"] in seniors:
            rep.errors.append(f"seniors.csv line {line}: duplicate senior_id '{r['senior_id']}'")
            continue
        try:
            rep.seniors.append(senior_row(r))
            seniors[r["senior_id"]] = r
        except ValueError:
            rep.errors.append(f"seniors.csv line {line}: cohort_year must be a number")

    seen: set[str] = set()
    for line, r in enumerate(_read_csv(projects_csv), start=2):
        pid = r.get("project_id", "")
        where = f"projects.csv line {line}"
        missing = [c for c in REQUIRED_PROJECT if not r.get(c)]
        if missing:
            rep.errors.append(f"{where}: missing {', '.join(missing)}")
        if pid in seen:
            rep.errors.append(f"{where}: duplicate project_id '{pid}'")
        seen.add(pid)
        if missing or not pid:
            continue
        if r["permission"].lower() != "granted":
            rep.warnings.append(f"{where}: '{pid}' permission != granted -> SKIPPED")
            continue

        authors, row_ok, withdrawn = [], True, 0
        for sid in _split(r["senior_ids"]):
            if sid not in seniors:
                rep.errors.append(f"{where}: unknown senior_id '{sid}'")
                row_ok = False
            elif seniors[sid].get("status", "active").lower() == "withdrawn":
                rep.warnings.append(f"{where}: '{sid}' withdrawn -> not linked")
                withdrawn += 1
            else:
                authors.append((sid, "lead" if not authors else "member"))
        if row_ok and withdrawn and not authors:
            # Takedown (PRD §8): every author withdrew -> the project and its chunks go.
            rep.warnings.append(f"{where}: '{pid}' all authors withdrawn -> SKIPPED (removed from index)")
            continue
        folder = corpus_dir / pid
        if not folder.is_dir():
            rep.errors.append(f"{where}: corpus folder missing: {folder}")
            row_ok = False
        try:
            year = int(r["year"])
        except ValueError:
            rep.errors.append(f"{where}: year must be a number")
            row_ok = False
        if not row_ok:
            continue

        files = sorted(p for p in folder.rglob("*")
                       if p.is_file() and not p.name.startswith(("~$", ".")))   # Office lock files
        for p in files:
            if p.suffix.lower() not in SUPPORTED:
                rep.warnings.append(f"{pid}: skipped unsupported file {p.name}")
        project = {
            "project_id": pid, "title_en": r["title_en"], "title_khmer": r.get("title_khmer") or None,
            "abstract": r["abstract"], "year": year, "term": r.get("term") or None,
            "program": r["program"].upper(), "domain": r.get("domain") or None,
            "tech_stack": _split(r.get("tech_stack", "")), "keywords": _split(r.get("keywords", "")),
            "repo_url": r.get("repo_url") or None, "demo_url": r.get("demo_url") or None,
            "supervisor_id": r["supervisor_id"], "permission": "granted",
        }
        rep.sources.append(Source(project, authors,
                                  [(p, doc_type_for(p)) for p in files if p.suffix.lower() in SUPPORTED]))

    # Drop-in documents: supported files directly in the corpus root.
    for p in sorted(corpus_dir.iterdir()):
        if not p.is_file() or p.name.startswith(("~$", ".")):   # skip Office lock files
            continue
        if p.suffix.lower() not in SUPPORTED:
            rep.warnings.append(f"drop-in: skipped unsupported file {p.name}")
            continue
        pid = f"doc-{slugify(p.stem)}"
        if pid in seen:
            rep.errors.append(f"drop-in {p.name}: id '{pid}' collides with a manifest project")
            continue
        seen.add(pid)
        rep.sources.append(Source(
            project={"project_id": pid, "title_en": p.stem.replace("_", " "), "abstract": "",
                     "year": datetime.fromtimestamp(p.stat().st_mtime).year,
                     "program": DROP_IN_PROGRAM, "supervisor_id": None, "permission": "granted",
                     "tech_stack": [], "keywords": []},
            authors=[], files=[(p, doc_type_for(p))], drop_in=True))
    return rep
