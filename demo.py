r"""Interactive query demo on the built-in sample corpus — no Qdrant server, no bge-m3.

    .\.venv\Scripts\python.exe demo.py

Runs the real retrieval plane (answer.service) against an in-memory Qdrant and
the fake bag-of-words embedder from the tests. Scores are NOT bge-m3 scores;
this is for trying the query paths before ingestion exists.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

from qdrant_client import QdrantClient

from answer import service
from cli import render
from config import Settings
from store.db import KBStore
from tests.conftest import FakeEmbedder, seed

HELP = """Type a query, or:
  /senior <name>      e.g. /senior sokdaraa
  /portfolio <id>     e.g. /portfolio s-2024-014
  /year <yyyy>        e.g. /year 2023
  /quit
Try: khmer ocr text recognition | Sok Dara | សុខ ដារា | Sok Dara's OCR project | quantum computing"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    db_path = Path(tempfile.mkdtemp()) / "demo.sqlite3"
    client, db = QdrantClient(":memory:"), KBStore(db_path)
    seed(db, client)
    kb = service.KnowledgeBase(client, FakeEmbedder(), db,
                               Settings(abstain_threshold=0.15, sqlite_path=str(db_path)))
    print(HELP)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        cmd, _, arg = q.partition(" ")
        if cmd == "/quit":
            break
        elif cmd == "/senior":
            res = service.senior(arg, kb)
        elif cmd == "/portfolio":
            res = service.portfolio(arg, kb)
        elif cmd == "/year" and arg.isdigit():
            res = service.browse({"year": [int(arg)]}, kb)
        elif cmd == "/help":
            print(HELP)
            continue
        else:
            res = service.answer(q, kb)
        print(render(res))


if __name__ == "__main__":
    main()
