"""Stage 00 on its own: validate the manifest without writing anything (FR-01).

    python -m ingestion.validate_manifest [--manifest data/projects.csv] [--seniors data/seniors.csv]

Exit code 1 if any row is invalid. The diff section lists what the next
ingestion run would add, update or remove compared with the current catalog.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingestion.manifest import validate


def manifest_diff(report, db) -> dict[str, list[str]]:
    current = {s.project["project_id"]: s for s in report.sources}
    known = db.project_ids()
    updated = []
    for pid in sorted(set(current) & known):
        if current[pid].drop_in:          # identity comes from the file, not the manifest
            continue
        row = db.project(pid) or {}
        src = current[pid].project
        if any(str(row.get(k) or "") != str(src.get(k) or "")
               for k in ("title_en", "year", "program", "supervisor_id")) \
                or sorted(row.get("senior_ids") or []) != sorted(sid for sid, _ in current[pid].authors):
            updated.append(pid)
    return {"added": sorted(set(current) - known), "updated": updated, "removed": sorted(known - set(current))}


def main(argv: list[str] | None = None) -> int:
    from config import Settings
    from store.db import KBStore

    s = Settings.from_env()
    ap = argparse.ArgumentParser(description="Validate projects.csv + seniors.csv + corpus folders")
    ap.add_argument("--manifest", default=s.projects_csv)
    ap.add_argument("--seniors", default=s.seniors_csv)
    ap.add_argument("--corpus", default=s.corpus_dir)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rep = validate(Path(args.manifest), Path(args.seniors), Path(args.corpus))
    manifest = [x for x in rep.sources if not x.drop_in]
    print(f"{len(rep.seniors)} seniors · {len(manifest)} manifest projects · "
          f"{len(rep.sources) - len(manifest)} drop-in documents · corpus hash {rep.corpus_hash[:12]}")
    for e in rep.errors:
        print(f"  ERROR: {e}")
    for w in rep.warnings:
        print(f"  warning: {w}")
    if rep.ok and Path(s.sqlite_path).exists():
        for kind, ids in manifest_diff(rep, KBStore(s.sqlite_path)).items():
            if ids:
                print(f"  {kind}: {', '.join(ids)}")
    print("OK" if rep.ok else f"FAILED: {len(rep.errors)} error(s) — nothing would be indexed")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
