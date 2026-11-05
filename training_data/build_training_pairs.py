"""Query log -> contrastive training pairs for Semester 2 (FR-11, Architecture Spec §12.2).

    python -m training_data.build_training_pairs [--out training_data/training_pairs_v1.jsonl]

(Not named `datasets/`: that would shadow HuggingFace `datasets`, which FlagEmbedding imports.)

positive  : the project the user clicked (bot) or confirmed with /good (cli)
negatives : projects ranked above or around it but not clicked  -> hard negatives
guard     : drop rating == -1; require exactly one positive
Reproducible from query_log alone: same log -> byte-identical file.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCHEMA_VERSION = 1
MAX_HARD_NEGATIVES = 5


def build_pairs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT query_id, raw_query, normalized_query, intent, retrieved_json,
                  clicked_project, rating, created_at
           FROM query_log
           WHERE clicked_project IS NOT NULL AND COALESCE(rating, 0) != -1
           ORDER BY created_at, query_id""").fetchall()
    pairs = []
    for query_id, raw, norm, intent, retrieved, clicked, rating, created in rows:
        try:
            ranked = [r["project_id"] for r in sorted(json.loads(retrieved or "[]"), key=lambda r: r["rank"])]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        negatives = [pid for pid in ranked if pid != clicked][:MAX_HARD_NEGATIVES]
        pairs.append({"query": raw, "normalized_query": norm, "positive_id": clicked,
                      "hard_negatives": negatives, "intent": intent, "rating": rating,
                      "query_id": query_id, "created_at": created, "schema_version": SCHEMA_VERSION})
    return pairs


def main(argv: list[str] | None = None) -> int:
    from config import Settings

    ap = argparse.ArgumentParser(description="Export training pairs from query_log")
    ap.add_argument("--db", default=Settings.from_env().sqlite_path)
    ap.add_argument("--out", default="training_data/training_pairs_v1.jsonl")
    args = ap.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    pairs = build_pairs(conn)
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in pairs),
                   encoding="utf-8")
    print(f"{len(pairs)} training pairs from {total} logged queries -> {out} (schema v{SCHEMA_VERSION})")
    if not pairs:
        print("  no positives yet: rate answers with /good or /bad in `python -m cli`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
