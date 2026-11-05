"""Offline front door to the retrieval plane (and the demo fallback if Telegram is blocked).

    python -m cli --query "khmer ocr"
    python -m cli --senior "Sok Dara"
    python -m cli --portfolio s-2024-014
    python -m cli --year 2024 --program DSE
    python -m cli --query "who built a khmer ocr system?" --llm --json

Calls answer.service exactly like the bot will — nothing here ranks or filters.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from answer import service
from config import Settings


def _card(c: dict, i: int) -> str:
    label = c.get("page_label") or (f"p.{c['page']}" if c.get("page") else None)
    where = f", {label}" if label else ""
    if c.get("heading"):
        where += f" ({c['heading']})"
    if c["authors"]:
        byline = f"   {', '.join(c['authors'])} · {c['year']} · {c['program']} · sup. {c['supervisor_id'] or '—'}"
    else:                                   # drop-in document: no people attached
        byline = f"   source: {c.get('source') or c['title']}"
    snippet = " ".join(c["snippet"].split())
    lines = [f"{i}. {c['title']}", byline,
             f"   \"{snippet[:400]}{'…' if len(snippet) > 400 else ''}\"",
             f"   — {c['title']}{where}"]
    meta = f"   matched: {c['matched']}"
    if c.get("score") is not None:
        meta += f" · rrf {c['score']} · cos {c['dense_score']}"
    if c.get("low_confidence"):
        meta += " · LOW CONFIDENCE"
    return "\n".join(lines + [meta])


def _portfolio(pf: dict) -> str:
    head = f"{pf['name']}" + (f"  ·  {pf['name_khmer']}" if pf.get("name_khmer") else "")
    out = [head, f"{pf['program']} · Cohort {pf['cohort_year']} · Supervisor {pf['supervisor_id'] or '—'}",
           "", f"Projects ({len(pf['projects'])})"]
    for i, p in enumerate(pf["projects"], 1):
        tag = "" if p["has_documents"] else "  [metadata only]"
        out.append(f"{i}. {p['title_en']} ({p['year']}){tag}")
    out.append("")
    if pf["contact"]:
        out.append("Connect: " + " · ".join(f"{k}: {v}" for k, v in pf["contact"].items()))
    else:
        out.append(pf["contact_notice"])
    if pf["relay_available"]:
        out.append("[Send a message via the bot]")
    return "\n".join(out)


def render(res: dict) -> str:
    out = []
    if res.get("note"):
        out.append(f"note: {res['note']}")
    if res["status"] == "ambiguous":
        out.append("Several seniors match that name. Which one?")
        out += [f"  [{c['name']} — {c['program']} {c['cohort_year']}]  ({c['senior_id']})"
                for c in res["candidates"]]
    if res["status"] == "not_found" and res.get("closest"):
        out.append("Closest matches (not confirmed):")
        out += [f"  · {c['name']} — {c['program']} {c['cohort_year']}   (similarity {c['score']:.0f})"
                for c in res["closest"]]
    if res["status"] == "abstain":
        out.append(res["answer"])
        if res["weak_matches"]:
            out.append("\nWeak matches (low confidence):")
            out += [_card(c, i) for i, c in enumerate(res["weak_matches"], 1)]
    if res.get("counts"):
        c = res["counts"]
        out.append(f"{c['total']} projects  ·  " + "  ".join(f"[{k} {v}]" for k, v in c["program"].items()))
    if res.get("answer") and res["status"] == "ok":
        out.append(res["answer"])
    if res["results"]:
        out += [_card(c, i) for i, c in enumerate(res["results"], 1 + res["page"] * 5)]
        if res["has_more"]:
            out.append("[ Next ▶ ]")
    if res.get("portfolio"):
        out.append(("\n" if out else "") + _portfolio(res["portfolio"]))
    if res.get("mode"):
        out.append(f"\n(mode: {res['mode']} · route: {res['route']} · {res.get('latency_ms', '?')} ms)")
    return "\n".join(out) or "(no output)"


def _source(c: dict) -> str:
    where = c.get("source") or c["title"]
    label = c.get("page_label") or (f"p.{c['page']}" if c.get("page") else "")
    heading = f" ({c['heading']})" if c.get("heading") else ""
    return f"{where}, {label}{heading}".rstrip(", ")


def render_qa(question: str, res: dict) -> str:
    """Question / Answer / Source layout for free-text questions."""
    out = [f"Question: {question}"]
    if res["status"] == "abstain":
        out.append(f"Answer: {res['answer']}")
        if res["weak_matches"]:
            out.append(f"Closest (low confidence): {_source(res['weak_matches'][0])}")
    elif res["results"]:
        g = res.get("grounding") or {}
        out.append(f"Answer: {res['answer']}")
        if g.get("verdict") != "verified":
            # I do not know: never present a merely related passage as the answer.
            out.append("Related sources (they do not state the answer): "
                       + "; ".join(_source(c) for c in res["results"]))
            if g.get("detail"):
                out.append(f"Why: {g['detail']}")
            return "\n".join(out)
        if g.get("method") == "llm_checked":
            seen, srcs = set(), []
            for p in res.get("llm_sources") or []:
                label = _source(p)
                if label not in seen:
                    seen.add(label)
                    srcs.append(label)
            out.append("Sources: " + "; ".join(srcs))
            out.append("Verified: every sentence checked against the passage it cites")
            used = {p["project_id"] for p in res.get("llm_sources") or []}
        else:
            src = res["answer_source"]
            out.append(f"Source: {_source(src)}")
            out.append(f"Verified: quoted verbatim from {src.get('chunk_id') or 'the source'}")
            used = {src["project_id"]}
        others = [_source(c) for c in res["results"] if c["project_id"] not in used]
        if others:
            out.append("Also relevant: " + "; ".join(others))
    else:                                   # portfolio, name disambiguation, browse, degraded
        out.append("Answer:")
        out.append(render(res))
    return "\n".join(out)


HELP = """Ask a question or type a topic / a senior's name. Commands:
  /senior <name>   /portfolio <senior_id>   /year <yyyy>   /reload   /help   /quit
  /good or /bad    rate the last answer (saved as training data for next semester)
Files added to data/corpus are indexed automatically before your next question."""


class Refresher:
    """Re-indexes data/corpus when its files change — checked before each question.

    Runs inside this process (same Qdrant handle), so there is no lock conflict
    with a second program, and unchanged files are never re-read.
    """

    def __init__(self, kb):
        self.kb, self.signature = kb, None

    def __call__(self, force: bool = False) -> None:
        from ingestion.run import corpus_signature, ingest

        sig = corpus_signature(self.kb.settings)
        if sig == self.signature and not force:
            return
        indexed = self.kb.db.get_meta("embed_revision")
        rebuild = bool(indexed) and indexed != self.kb.embedder.revision
        if rebuild:            # vectors from another model are not comparable: start over
            print(f"  ↻ search model changed ({indexed} → {self.kb.embedder.revision}): rebuilding index …")
        report = ingest(self.kb, rebuild=rebuild)
        self.signature = sig
        changed = [d for d in report["documents"] if not d.get("skipped")]
        for d in changed:
            status = "FAILED" if d["parse_status"] == "failed" else f"{d['chunks']} chunks"
            print(f"  ↻ indexed {Path(d['file']).name}: {status}")
        for doc_id in report["removed"]:
            print(f"  ↻ removed {doc_id}")
        for e in report["errors"]:
            print(f"  ! {e}")
        if force and not changed and not report["removed"] and not report["errors"]:
            print(f"  index is up to date ({report['points_total']} chunks)")


CODE_DIRS = ("answer", "retrieval", "ingestion", "store")


def _code_stamp() -> float:
    """Newest modification time of this program's own .py files."""
    root = Path(__file__).resolve().parent
    files = [root / "cli.py", root / "config.py"] + [p for d in CODE_DIRS for p in (root / d).glob("*.py")]
    return max((p.stat().st_mtime for p in files if p.exists()), default=0.0)


def _answer_project(res: dict) -> str | None:
    """The project the shown answer came from — what a /good confirms."""
    if res.get("llm_sources"):
        return res["llm_sources"][0]["project_id"]
    if res.get("answer_source"):
        return res["answer_source"]["project_id"]
    if res.get("results"):
        return res["results"][0]["project_id"]
    return None


def repl(kb, *, log_queries: bool = True, enable_llm: bool | None = None,
         auto_refresh: bool = True) -> None:
    """Interactive prompt: one loaded model, many questions."""
    refresh = Refresher(kb) if auto_refresh else (lambda force=False: None)
    refresh()
    print(HELP)
    started, warned, last = _code_stamp(), False, None
    while True:
        if not warned and _code_stamp() > started:
            # Python loaded the code at startup; edits only apply after a restart.
            print("\n  ! The program was updated since this session started. "
                  "Type /quit and run it again to use the new version.")
            warned = True
        try:
            q = input("\n> ").replace("﻿", "").strip()   # piped input may carry a BOM
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        cmd, _, arg = q.partition(" ")
        if cmd in ("/quit", "/exit"):
            return
        if cmd == "/help":
            print(HELP)
            continue
        if cmd == "/reload":
            refresh(force=True)
            continue
        if cmd in ("/good", "/bad"):
            # The CLI's 👍 / 👎 (FR-11): rating + the confirmed project become a training pair.
            if not last:
                print("  nothing to rate yet: ask a question first")
            else:
                good = cmd == "/good"
                kb.db.rate_query(last["query_id"], 1 if good else -1,
                                 last["project_id"] if good else None)
                print(f"  thanks — recorded as {'helpful' if good else 'not helpful'}")
            continue
        refresh()
        if cmd == "/senior" and arg:
            res = service.senior(arg, kb)
        elif cmd == "/portfolio" and arg:
            res = service.portfolio(arg.strip(), kb)
        elif cmd == "/year" and arg.strip().isdigit():
            res = service.browse({"year": [int(arg)]}, kb)
        else:
            res = service.answer(q, kb, enable_llm=enable_llm)
        query_id = service.log_interaction(kb, res, surface="cli") if log_queries else None
        last = {"query_id": query_id, "project_id": _answer_project(res)} if query_id else None
        print(render_qa(q, res) if not cmd.startswith("/") else render(res))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RUPP-SPKB retrieval CLI (no arguments = interactive)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--query", "-q")
    g.add_argument("--senior")
    g.add_argument("--portfolio", metavar="SENIOR_ID")
    g.add_argument("--browse", action="store_true", help="facet browse with --year/--program")
    ap.add_argument("--year", type=int, action="append")
    ap.add_argument("--program", action="append")
    ap.add_argument("--page", type=int, default=0)
    llm = ap.add_mutually_exclusive_group()
    llm.add_argument("--llm", action="store_true",
                     help="Gemma 4 writes answers to questions (same as ENABLE_LLM=true; default off)")
    llm.add_argument("--no-llm", action="store_true", help="quoted answers only, never call Gemma")
    ap.add_argument("--json", action="store_true", help="print the raw service result")
    ap.add_argument("--no-log", action="store_true", help="do not write to query_log")
    ap.add_argument("--no-refresh", action="store_true",
                    help="do not auto-index changes in data/corpus before answering")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")       # Khmer on a Windows console
        sys.stdin.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    facets = {k: v for k, v in {"year": args.year, "program": args.program}.items() if v}
    if facets and not (args.query or args.browse):
        ap.error("--year/--program apply to --query or --browse")

    settings = Settings.from_env()
    interactive = not (args.query or args.senior or args.portfolio or args.browse)
    try:
        kb = service.KnowledgeBase.from_settings(settings, load_embedder=bool(args.query) or interactive)
    except RuntimeError as exc:                 # e.g. local index locked by another window
        print(f"Cannot start: {exc}")
        return 1
    for problem in kb.health():
        logging.warning("health: %s", problem)

    args.llm_on = _llm_mode(kb, args) if (args.query or interactive) else False
    try:
        return _dispatch(kb, args, facets, interactive)
    finally:
        kb.close()


def _llm_mode(kb, args) -> bool:
    """Gemma only when asked: --llm or ENABLE_LLM=true (spec default: off, extractive)."""
    from answer.prompts import model_available, warm_up

    s = kb.settings
    if args.no_llm:
        return False
    wanted = args.llm or s.enable_llm
    ready = model_available(ollama_url=s.ollama_url, model=s.ollama_model)
    if not wanted:
        if ready:
            print(f"  tip: {s.ollama_model} is installed — start with --llm (or set ENABLE_LLM=true) "
                  "for written answers.")
        return False
    if not ready:
        print(f"  ! Gemma ({s.ollama_model}) is not available in Ollama — quoted answers only.")
        return False
    print(f"  loading {s.ollama_model} … (first time can take a minute)")
    try:
        warm_up(ollama_url=s.ollama_url, model=s.ollama_model)
    except Exception as exc:
        print(f"  ! could not load Gemma ({exc}) — quoted answers only.")
        return False
    print("  Gemma ready: questions get written answers with checked citations.")
    return True


def _dispatch(kb, args, facets: dict, interactive: bool) -> int:
    if interactive:
        logging.getLogger().setLevel(logging.WARNING)
        repl(kb, log_queries=not args.no_log, enable_llm=args.llm_on,
             auto_refresh=not args.no_refresh)
        return 0
    if args.query and not args.no_refresh:
        Refresher(kb)()
    if args.query:
        res = service.answer(args.query, kb, facets=facets, page=args.page,
                             enable_llm=args.llm_on)
    elif args.senior:
        res = service.senior(args.senior, kb)
    elif args.portfolio:
        res = service.portfolio(args.portfolio, kb)
    else:
        res = service.browse(facets, kb, page=args.page)

    if not args.no_log:
        service.log_interaction(kb, res, surface="cli")
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    elif args.query:
        print(render_qa(args.query, res))
    else:
        print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
