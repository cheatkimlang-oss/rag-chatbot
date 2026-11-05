"""Snapshot university web pages into the corpus, so they ingest like any file.

    python -m ingestion.fetch_web                       URLs from data/web_sources.txt
    python -m ingestion.fetch_web --url https://...     one page
    python -m ingestion.run                             then index as usual

Why snapshot instead of fetching during ingestion: the index must be rebuildable
and idempotent from what is on disk (corpus hash, deterministic point ids). Each
page is saved as data/corpus/web-<slug>.html with its URL recorded in a
<meta name="source-url"> tag, which the HTML parser turns into the citation.

A page is rewritten only when its visible text changed, so re-fetching an unchanged
site causes 0 re-embedded chunks. robots.txt is honoured and requests are spaced
out; only pages the department is allowed to republish belong in the list.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ingestion.manifest import slugify
from ingestion.parsers import read_html_bytes

USER_AGENT = "RUPP-SPKB/2.0 (academic knowledge-base project)"
DEFAULT_LIST = Path("data/web_sources.txt")
DEFAULT_OUT = Path("data/corpus")
_HEAD = re.compile(rb"<head[^>]*>", re.I)


def read_url_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    urls = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return list(dict.fromkeys(urls))                    # de-duplicate, keep order


def snapshot_name(url: str) -> str:
    parts = urlsplit(url)
    return f"web-{slugify(parts.netloc + parts.path + ('-' + parts.query if parts.query else ''))}.html"


def with_source_meta(body: bytes, url: str) -> bytes:
    tag = f'<meta name="source-url" content="{html.escape(url, quote=True)}">'.encode()
    m = _HEAD.search(body)
    return body[:m.end()] + tag + body[m.end():] if m else tag + body


class _Robots:
    def __init__(self, client: httpx.Client):
        self.client, self.cache = client, {}

    def allowed(self, url: str) -> bool:
        p = urlsplit(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self.cache:
            rp = RobotFileParser()
            try:
                r = self.client.get(f"{root}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])                            # unreachable robots.txt: no rules
            self.cache[root] = rp
        return self.cache[root].can_fetch(USER_AGENT, url)


def fetch(urls: list[str], out_dir: Path = DEFAULT_OUT, *, client: httpx.Client | None = None,
          delay: float = 1.0) -> list[dict]:
    """Fetch each URL to out_dir. Returns one {url, file, status} row per URL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20,
                                    follow_redirects=True)
    robots, rows = _Robots(client), []
    try:
        for i, url in enumerate(urls):
            row = {"url": url, "file": None, "status": ""}
            rows.append(row)
            if urlsplit(url).scheme not in ("http", "https"):
                row["status"] = "error: not an http(s) URL"
                continue
            if not robots.allowed(url):
                row["status"] = "skipped: disallowed by robots.txt"
                continue
            if i and delay:
                time.sleep(delay)                       # be polite to university servers
            try:
                r = client.get(url)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                row["status"] = f"error: {exc}"
                continue
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype:
                row["status"] = f"skipped: content-type {ctype or 'unknown'} (save PDFs into the corpus directly)"
                continue
            target = out_dir / snapshot_name(url)
            data = with_source_meta(r.content, url)
            row["file"] = str(target)
            # Compare extracted text, not bytes: rotating banners / CSRF tokens in
            # the markup must not force a re-embed of an unchanged page.
            if target.exists() and read_html_bytes(target.read_bytes()) == read_html_bytes(data):
                row["status"] = "unchanged"
            else:
                row["status"] = "updated" if target.exists() else "new"
                target.write_bytes(data)
    finally:
        if own:
            client.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snapshot web pages into the corpus")
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST, help="file with one URL per line")
    ap.add_argument("--url", action="append", default=[], help="fetch this URL (repeatable)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    urls = args.url or read_url_list(args.list)
    if not urls:
        print(f"No URLs: add them to {args.list} or pass --url")
        return 1
    rows = fetch(urls, args.out, delay=args.delay)
    for r in rows:
        print(f"  [{r['status']}] {r['url']}" + (f" -> {r['file']}" if r["file"] else ""))
    failed = [r for r in rows if r["status"].startswith("error")]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} ok. Next: python -m ingestion.run")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
