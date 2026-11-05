"""Golden-set evaluation by slice (Architecture Spec §13, PRD §9–10).

    python -m eval.run_eval                          metrics per slice at the configured gate
    python -m eval.run_eval --sweep-threshold        + the abstain-threshold sweep table

Calls answer.service.answer() directly — the same code path the bot and CLI use,
never Telegram. Each query runs ONCE with the gate disabled; the gate is then
replayed offline for every threshold, so the sweep costs no extra searches.

The threshold rule (Pipeline §13): the SMALLEST threshold reaching abstention
precision >= 0.95 on the negative slice without dropping topic Recall@5 below 0.90.
Every run writes logs/eval_<ts>.json with the model tags used (FR-14).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

TARGET_ABSTENTION = 0.95
TARGET_RECALL = 0.90
SWEEP = [round(0.30 + 0.025 * i, 3) for i in range(21)]          # 0.300 … 0.800


def slice_kind(name: str) -> str:
    if name.startswith("name"):
        return "name"
    if name.startswith("negative"):
        return "negative"
    if name.startswith("facet"):
        return "facet"
    return "topic"                     # topic_en, topic_km, answer_* …


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    for r in rows:
        r["expected"] = {p.strip() for p in (r.get("expected_project_ids") or "").split("|") if p.strip()}
        r["kind"] = slice_kind(r["slice"])
    return rows


def run_queries(kb, rows: list[dict]) -> list[dict]:
    from answer import service

    records = []
    for r in rows:
        res = service.answer(r["query"], kb, enable_llm=False)
        cards = [{"project_id": c["project_id"], "dense": c.get("dense_score"),
                  "lexical": "lexical" in (c.get("matched") or "")} for c in res["results"]]
        if not cards and res.get("portfolio"):           # name query -> the senior's projects
            cards = [{"project_id": p["project_id"], "dense": None, "lexical": True}
                     for p in res["portfolio"]["projects"]]
        records.append({**{k: r[k] for k in ("query_id", "query", "slice", "kind")},
                        "expected": sorted(r["expected"]),
                        "expected_senior_id": (r.get("expected_senior_id") or "").strip() or None,
                        "route": res["route"], "gate": res.get("gate_score"),
                        "senior_id": res.get("senior_id"), "cards": cards,
                        "latency_ms": res.get("latency_ms")})
    return records


def ranked_at(rec: dict, threshold: float) -> list[str]:
    """Replay the service's gate + tail filter at `threshold` (answer/service.py)."""
    if rec["gate"] is None:                              # name / facet paths are not gated
        return [c["project_id"] for c in rec["cards"]]
    if rec["gate"] < threshold:
        return []                                        # abstained
    return [c["project_id"] for c in rec["cards"]
            if c["lexical"] or (c["dense"] or 0.0) >= threshold]


def score(records: list[dict], threshold: float) -> dict[str, dict]:
    by_slice: dict[str, list[dict]] = {}
    for rec in records:
        by_slice.setdefault(rec["slice"], []).append(rec)
    out = {}
    for name, recs in sorted(by_slice.items()):
        kind = recs[0]["kind"]
        row: dict = {"n": len(recs), "kind": kind}
        if kind in ("topic", "facet"):
            ranked = [(ranked_at(r, threshold), set(r["expected"])) for r in recs]
            row["recall@5"] = mean(recall_at_k(rk, ex, 5) for rk, ex in ranked)
            row["mrr@10"] = mean(reciprocal_rank(rk, ex, 10) for rk, ex in ranked)
            row["ndcg@10"] = mean(ndcg_at_k(rk, ex, 10) for rk, ex in ranked)
        elif kind == "name":
            row["top1"] = mean(float(r["senior_id"] == r["expected_senior_id"]) for r in recs)
        elif kind == "negative":
            row["abstention"] = mean(float(not ranked_at(r, threshold)) for r in recs)
        row["p50_ms"] = sorted(r["latency_ms"] or 0 for r in recs)[len(recs) // 2]
        out[name] = row
    return out


def sweep(records: list[dict]) -> tuple[list[dict], float | None]:
    table = []
    for t in SWEEP:
        s = score(records, t)
        neg = [v["abstention"] for v in s.values() if v["kind"] == "negative"]
        topic = [r for r in records if r["kind"] == "topic"]
        recall = mean(recall_at_k(ranked_at(r, t), set(r["expected"]), 5) for r in topic)
        table.append({"threshold": t, "abstention": mean(neg) if neg else None, "topic_recall@5": recall})
    chosen = next((row["threshold"] for row in table
                   if (row["abstention"] or 0) >= TARGET_ABSTENTION
                   and (row["topic_recall@5"] or 0) >= TARGET_RECALL), None)
    return table, chosen


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_slices(scores: dict[str, dict]) -> None:
    cols = ["n", "recall@5", "mrr@10", "ndcg@10", "top1", "abstention", "p50_ms"]
    print(f"  {'slice':<16}" + "".join(f"{c:>12}" for c in cols))
    for name, row in scores.items():
        print(f"  {name:<16}" + "".join(f"{_fmt(row.get(c)):>12}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    from answer.service import KnowledgeBase
    from config import Settings

    ap = argparse.ArgumentParser(description="RUPP-SPKB golden-set evaluation")
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--threshold", type=float, help="gate to report at (default: configured)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = load_golden(Path(args.golden))
    base = Settings.from_env()
    threshold = args.threshold if args.threshold is not None else base.gate
    # Gate off + 10 results per query: the gate is replayed offline in score().
    settings = dataclasses.replace(base, abstain_threshold=0.0, display_top_k=10, answer_scorer="")
    try:
        kb = KnowledgeBase.from_settings(settings)
    except RuntimeError as exc:
        print(f"Cannot start: {exc}")
        return 1
    try:
        for problem in kb.health():
            print(f"  health: {problem}")
        records = run_queries(kb, rows)
        models = {**kb.models(), "indexed_with": kb.db.get_meta("embed_revision")}
    finally:
        kb.close()

    scores = score(records, threshold)
    print(f"\nGolden set {args.golden}: {len(rows)} queries · embedder {models['embed']} · gate {threshold}\n")
    print_slices(scores)
    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "golden": args.golden,
              "models": models, "threshold": threshold, "slices": scores, "records": records}

    if args.sweep_threshold:
        table, chosen = sweep(records)
        print(f"\nAbstain-threshold sweep (targets: abstention >= {TARGET_ABSTENTION}, "
              f"topic Recall@5 >= {TARGET_RECALL})\n  {'threshold':>9} {'abstention':>11} {'recall@5':>9}")
        for row in table:
            mark = "  <- chosen" if row["threshold"] == chosen else ""
            print(f"  {row['threshold']:>9.3f} {_fmt(row['abstention']):>11} {_fmt(row['topic_recall@5']):>9}{mark}")
        print(f"\nchosen threshold: {chosen if chosen is not None else 'none meets both targets'}"
              + (f"  ->  set KB_ABSTAIN_THRESHOLD={chosen}" if chosen is not None else ""))
        report.update(sweep=table, chosen_threshold=chosen)

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
