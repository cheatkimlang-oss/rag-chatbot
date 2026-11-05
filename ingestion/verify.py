"""Post-run integrity assertions on their own (Architecture Spec §8.3, Pipeline §14 step 3).

    python -m ingestion.verify

Checks: points == sum(chunk_count); every point carries the required payload;
every senior_id is an active senior; the index was built with the embedder
this process would query with. Exit code 1 on any failure.
"""
from __future__ import annotations

import sys

from ingestion.run import verify


def main() -> int:
    from answer.service import KnowledgeBase
    from config import Settings

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        kb = KnowledgeBase.from_settings(Settings.from_env(), load_embedder=False)
    except RuntimeError as exc:
        print(f"Cannot start: {exc}")
        return 1
    try:
        s = kb.settings
        problems = list(kb.health())
        if not problems:
            problems += verify(kb)
        indexed = kb.db.get_meta("embed_revision")
        expected = s.embed_model if s.embed_backend == "bge-m3" else f"fastembed:{s.fastembed_dense}+{s.fastembed_sparse}"
        if indexed and indexed != expected:
            problems.append(f"index built with {indexed!r}, settings expect {expected!r} — run ingestion --rebuild")
        if not problems:
            docs = kb.db.documents().values()
            print(f"OK: {kb.client.count(s.collection, exact=True).count} points · {len(docs)} documents · "
                  f"{sum(1 for d in docs if d['parse_status'] == 'failed')} failed · embedder {indexed}")
    finally:
        kb.close()
    for p in problems:
        print(f"  FAIL: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
