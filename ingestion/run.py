"""Plane A — ingestion: validate -> parse -> chunk -> embed -> upsert -> verify.

    python -m ingestion.run              incremental: only new/changed files are re-embedded
    python -m ingestion.run --rebuild    drop the index + catalog and rebuild from scratch

Idempotent: point ids are uuid5(chunk_id) and unchanged files are skipped, so a
re-run on an unchanged corpus writes 0 points. Writes logs/ingest_<ts>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from qdrant_client import models

from ingestion.chunker import ChunkConfig, build_chunks
from ingestion.manifest import validate
from ingestion.parsers import PARSER_VERSION, parse_file, sha256_of
from store.vector_schema import DENSE, LEXICAL, REQUIRED_PAYLOAD, ensure_collection, index_mismatch

log = logging.getLogger("ingestion")
NS = uuid.UUID("6f2a1c3e-0000-4000-8000-000000000000")   # fixed -> stable point ids
ABSTRACT_CHARS = 600


def _match(key: str, value) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])


def _delete_points(client, collection: str, key: str, value: str) -> None:
    client.delete(collection, points_selector=models.FilterSelector(filter=_match(key, value)), wait=True)


def _fingerprint(sha: str, project: dict, authors: list[dict], revision: str,
                 chunking: dict | None = None) -> str:
    """A file is unchanged only if its bytes, its project row, the model, the parser
    AND the chunking settings are — so changing KB_CHUNK_SIZE re-chunks everything."""
    blob = json.dumps([sha, project, authors, revision, PARSER_VERSION, chunking or {}],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _drop_in_abstract(parsed) -> str:
    for page in parsed.pages:
        text = page.page_content
        heading = page.metadata.get("heading")
        if heading and text.startswith(heading):
            text = text[len(heading):]
        text = " ".join(text.split())
        if len(text) >= 80:
            return text[:ABSTRACT_CHARS]
    return ""


def ingest(kb, *, rebuild: bool = False, corpus_dir: Path | None = None,
           projects_csv: Path | None = None, seniors_csv: Path | None = None) -> dict:
    s, client, db, emb = kb.settings, kb.client, kb.db, kb.embedder
    corpus_dir = Path(corpus_dir or s.corpus_dir)
    t0 = time.perf_counter()
    rep = validate(Path(projects_csv or s.projects_csv), Path(seniors_csv or s.seniors_csv), corpus_dir)
    report = {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "failed",
              "embedder": emb.revision, "collection": s.collection, "corpus_hash": rep.corpus_hash,
              "errors": list(rep.errors), "warnings": list(rep.warnings), "documents": [], "removed": [],
              "points_written": 0, "points_total": 0}
    if not rep.ok:
        return report                                   # fail fast: nothing written

    report["config"] = {"chunking": s.chunk_config(), "index": s.index_config()}
    try:
        chunking = ChunkConfig.from_settings(s, emb)
    except ValueError as exc:                           # e.g. token chunks the model would truncate
        report["errors"].append(str(exc))
        return report
    indexed = db.get_meta("embed_revision")
    if indexed and indexed != emb.revision and not rebuild:
        report["errors"].append(f"index was built with {indexed!r}, embedder is {emb.revision!r}: "
                                "re-run with --rebuild")
        return report
    if not rebuild and (why := index_mismatch(client, s.collection, s.index_config())):
        report["errors"].append(why)                    # distance / HNSW cannot change in place
        return report
    if rebuild:
        if client.collection_exists(s.collection):
            client.delete_collection(s.collection)
        db.reset_catalog()
    ensure_collection(client, s.collection, dim=emb.dim, sparse_idf=emb.sparse_idf,
                      index=s.index_config())
    db.set_meta("embed_revision", emb.revision)

    seniors = {}
    for row, aliases in rep.seniors:
        db.upsert_senior(row, aliases)
        seniors[row["senior_id"]] = row

    # Projects removed from the manifest / corpus: drop their points and rows.
    current = {src.project["project_id"] for src in rep.sources}
    for pid in db.project_ids() - current:
        _delete_points(client, s.collection, "project_id", pid)
        db.delete_project(pid)
        report["removed"].append(pid)

    known_docs = db.documents()
    known_projects = db.project_ids()
    for src in rep.sources:
        pid = src.project["project_id"]
        authors = [{"senior_id": sid, "full_name_latin": seniors[sid]["full_name_latin"],
                    "name_normalized": seniors[sid]["name_normalized"]} for sid, _ in src.authors]
        # Fingerprint BEFORE parsing (bytes + manifest row + model): an unchanged
        # file is never re-read, so a refresh before every question stays cheap.
        docs = []
        for path, doc_type in src.files:
            doc_id = f"{pid}__{path.relative_to(corpus_dir).as_posix().replace('/', '__')}"
            fp = _fingerprint(sha256_of(path), src.project, authors, emb.revision,
                              s.chunk_config())
            old = known_docs.get(doc_id)
            unchanged = bool(old) and db.get_meta(f"fp:{doc_id}") == fp
            docs.append((doc_id, path, doc_type, fp, old, unchanged))
        keep = {d[0] for d in docs}

        if pid in known_projects and all(d[5] for d in docs):
            for doc_id, path, _, _, old, _ in docs:
                report["documents"].append({
                    "doc_id": doc_id, "file": str(path), "parser": old["parser"],
                    "parse_status": old["parse_status"], "pages": old["page_count"] or 0,
                    "chunks": old["chunk_count"], "errors": [], "skipped": "unchanged"})
        else:
            parsed_docs = [(doc_id, doc_type, fp, parse_file(path, doc_id, pid))
                           for doc_id, path, doc_type, fp, _, _ in docs]
            project = dict(src.project)
            if src.drop_in and parsed_docs:             # identity comes from the file itself
                parsed = parsed_docs[0][3]
                project["title_en"] = parsed.title or project["title_en"]
                project["abstract"] = _drop_in_abstract(parsed)
            db.upsert_project(project, src.authors)
            for doc_id, doc_type, fp, parsed in parsed_docs:
                report["points_written"] += _index_document(kb, report, pid, doc_id, doc_type,
                                                            fp, parsed, project, authors, chunking)

        for doc_id, old in known_docs.items():          # files removed from a project folder
            if old["project_id"] == pid and doc_id not in keep:
                _delete_points(client, s.collection, "doc_id", doc_id)
                db.delete_document(doc_id)
                report["removed"].append(doc_id)

    report["errors"] += verify(kb)
    report["points_total"] = client.count(s.collection, exact=True).count
    report["status"] = "ok" if not report["errors"] else "failed"
    db.set_meta("corpus_hash", rep.corpus_hash)
    report["seconds"] = round(time.perf_counter() - t0, 2)
    return report


def _index_document(kb, report, pid, doc_id, doc_type, fp, parsed, project, authors,
                    chunking: ChunkConfig | None = None) -> int:
    s, client, db, emb = kb.settings, kb.client, kb.db, kb.embedder
    entry = {"doc_id": doc_id, "file": str(parsed.path), "parser": parsed.parser,
             "parse_status": parsed.parse_status, "pages": len(parsed.pages),
             "chunks": 0, "errors": parsed.errors}
    report["documents"].append(entry)
    _delete_points(client, s.collection, "doc_id", doc_id)    # chunk count may shrink
    chunks = build_chunks(parsed, project, authors, doc_type, chunking) if parsed.pages else []
    if parsed.parse_status != "failed" and not chunks:
        parsed.parse_status = "failed"
        entry.update(parse_status="failed", errors=["text too short to index"])
    if chunks:
        dense, sparse = emb.encode([c.text for c in chunks])
        client.upsert(s.collection, wait=True, points=[
            models.PointStruct(
                id=str(uuid.uuid5(NS, c.chunk_id)),
                vector={DENSE: d, LEXICAL: models.SparseVector(indices=sv[0], values=sv[1])},
                payload={**c.metadata, "text": c.text},
            ) for c, d, sv in zip(chunks, dense, sparse)])
    entry["chunks"] = len(chunks)
    db.upsert_document({"doc_id": doc_id, "project_id": pid, "doc_type": doc_type,
                        "file_path": str(parsed.path), "sha256": parsed.sha256,
                        "page_count": parsed.page_count, "parser": parsed.parser,
                        "parse_status": parsed.parse_status, "chunk_count": len(chunks)})
    # Recorded for failed files too: a broken file is reported once, not re-parsed
    # before every question. Changing the file changes its fingerprint.
    db.set_meta(f"fp:{doc_id}", fp)
    if parsed.parse_status == "failed":
        report["warnings"].append(f"{doc_id}: FAILED ({'; '.join(entry['errors'])}) -> excluded")
    return len(chunks)


def corpus_signature(settings) -> tuple:
    """Names, sizes and mtimes of every source file — a stat-only change detector."""
    paths = [Path(settings.projects_csv), Path(settings.seniors_csv)]
    corpus = Path(settings.corpus_dir)
    if corpus.exists():
        paths += sorted(p for p in corpus.rglob("*") if p.is_file())
    out = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p), st.st_size, st.st_mtime_ns))
    return tuple(out)


def verify(kb) -> list[str]:
    """Post-run integrity assertions — code, not discipline (Spec §8.3)."""
    s, client, db = kb.settings, kb.client, kb.db
    errors = []
    expected = sum(d["chunk_count"] or 0 for d in db.documents().values())
    count = client.count(s.collection, exact=True).count
    if count != expected:
        errors.append(f"points {count} != sum(chunk_count) {expected}")
    offset = None
    while True:
        pts, offset = client.scroll(s.collection, limit=256, offset=offset, with_payload=True)
        for p in pts:
            missing = [k for k in REQUIRED_PAYLOAD if k not in p.payload]
            if missing:
                errors.append(f"{p.payload.get('chunk_id', p.id)}: payload missing {missing}")
            for sid in p.payload.get("senior_ids", []):
                if not db.senior_exists(sid):
                    errors.append(f"{p.payload.get('chunk_id')}: inactive/unknown senior_id {sid}")
        if offset is None or len(errors) > 20:
            break
    return errors


def main(argv: list[str] | None = None) -> int:
    from answer.service import KnowledgeBase
    from config import Settings

    ap = argparse.ArgumentParser(description="RUPP-SPKB ingestion")
    ap.add_argument("--rebuild", action="store_true", help="drop index + catalog first")
    ap.add_argument("--corpus"), ap.add_argument("--projects"), ap.add_argument("--seniors")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        kb = KnowledgeBase.from_settings(Settings.from_env())
    except RuntimeError as exc:                 # e.g. local index locked by a cli session
        print(f"Cannot start: {exc}")
        return 1
    try:
        report = ingest(kb, rebuild=args.rebuild, corpus_dir=args.corpus,
                        projects_csv=args.projects, seniors_csv=args.seniors)
    finally:
        kb.close()

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"ingest_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for d in report["documents"]:
        note = d.get("skipped") or d["parse_status"]
        print(f"  {d['doc_id']:<60} {d['pages']:>3} anchors  {d['chunks']:>3} chunks  [{note}]")
    for doc_id in report["removed"]:
        print(f"  removed: {doc_id}")
    for w in report["warnings"]:
        print(f"  warning: {w}")
    for e in report["errors"]:
        print(f"  ERROR: {e}")
    print(f"\n{report['status'].upper()}: {report['points_written']} points written, "
          f"{report['points_total']} in index · {report['embedder']} · report: {out}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
