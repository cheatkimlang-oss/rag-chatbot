"""Answer-level evaluation: does the system give the RIGHT answer, or honestly abstain?

    python -m eval.run_qa                  run eval/qa_set.csv, print failures + a summary
    python -m eval.run_qa --all            print every question, not just failures
    python -m eval.run_qa --only dse       one slice

eval/run_eval.py measures retrieval (was the right document found?). This one checks the
served answer, the same way the CLI shows it:

  answerable question  PASS when the answer is verified AND contains the expected text.
                       `must_contain` is "a|b|c": any one of them, case-insensitive.
  negative question    PASS when the system says "I do not know" (abstains or refuses
                       to serve an unverified sentence).

Writes logs/qa_<timestamp>.json with every answer, so runs can be compared.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from answer import service
from config import Settings

SET = Path(__file__).with_name("qa_set.csv")


def served(res: dict) -> tuple[str, str | None, bool]:
    """(answer text, source doc id, verified?) exactly as the CLI would present it."""
    g = res.get("grounding") or {}
    verified = res["status"] == "ok" and bool(res.get("results")) and g.get("verdict") == "verified"
    src = None
    if res.get("llm_sources"):
        src = res["llm_sources"][0]["project_id"]
    elif res.get("answer_source"):
        src = res["answer_source"]["project_id"]
    return res.get("answer") or "", src, verified


def judge(row: dict, res: dict) -> tuple[bool, str]:
    text, src, verified = served(res)
    if row["slice"].startswith("negative"):
        ok = not verified
        return ok, "abstained" if ok else f"answered anyway: {text[:120]!r}"
    if not verified:
        return False, f"no verified answer ({res['status']}; {(res.get('grounding') or {}).get('detail', '')})"
    wanted = [w.strip() for w in row["must_contain"].split("|") if w.strip()]
    if not any(w.lower() in text.lower() for w in wanted):
        return False, f"wrong answer from {src}: {text[:160]!r}"
    docs = [d for d in row["expected_doc"].split("|") if d]
    if docs and not any((src or "").startswith(d) for d in docs):
        return True, f"right text, but from {src}"          # counted as pass, flagged
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="answer-level QA evaluation")
    ap.add_argument("--set", default=str(SET))
    ap.add_argument("--only", help="run one slice (about_me, dse, ite, negative, ...)")
    ap.add_argument("--all", action="store_true", help="print passes too")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)

    rows = list(csv.DictReader(open(args.set, encoding="utf-8")))
    if args.only:
        rows = [r for r in rows if r["slice"].startswith(args.only)]
    kb = service.KnowledgeBase.from_settings(Settings.from_env(), load_embedder=True)
    per_slice, records = defaultdict(lambda: [0, 0]), []
    try:
        for row in rows:
            res = service.answer(row["question"], kb)
            ok, why = judge(row, res)
            per_slice[row["slice"]][0] += ok
            per_slice[row["slice"]][1] += 1
            text, src, _ = served(res)
            score = re.search(r"score (-?[\d.]+)", (res.get("grounding") or {}).get("detail") or "")
            records.append({**row, "pass": ok, "why": why, "answer": text, "source": src,
                            "answer_score": float(score.group(1)) if score else None})
            if args.all or not ok or why != "ok":
                print(f"{'PASS' if ok else 'FAIL'} {row['id']:<4} {row['question']}\n      {why}")
    finally:
        kb.close()

    total = sum(p for p, _ in per_slice.values()), sum(n for _, n in per_slice.values())
    print("\nslice                 pass / total")
    for name, (p, n) in per_slice.items():
        print(f"  {name:<20} {p:>4} / {n}")
    print(f"  {'ALL':<20} {total[0]:>4} / {total[1]}  ({100 * total[0] / max(total[1], 1):.0f}%)")
    out = Path("logs") / f"qa_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"details: {out}")
    return 0 if total[0] == total[1] else 1


if __name__ == "__main__":
    sys.exit(main())
