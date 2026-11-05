"""Serve the RUPP student assistant on this computer."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from answer import DEFAULT_MODEL, EXAMPLE_QUESTIONS, build_answer
from curriculum import load_curriculum
from hybrid import load_hybrid
from retrieve import COLLECTIONS, PROJECT_DIR, load_passages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        rows = load_passages(COLLECTIONS["official"])
        snapshot = load_curriculum()
        page = (PROJECT_DIR / "web" / "index.html").read_bytes()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if snapshot is None:
        parser.error("Refresh the official curriculum before starting web chat.")

    search = load_hybrid(rows)
    known_urls = {snapshot["source_url"]}
    known_urls.update(program["source_url"] for program in snapshot["programs"])
    known_urls.update(row["url"] for row in rows if row.get("url"))
    known_urls = {url for url in known_urls if urlsplit(url).scheme == "https"}

    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self):
            port = self.server.server_port
            if self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                return True
            self.send_error(403)
            return False

        def reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, status, payload):
            self.reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if not self.allowed_host():
                return
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/examples":
                self.reply_json(200, {"examples": EXAMPLE_QUESTIONS})
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.allowed_host():
                return
            if self.path != "/api/ask":
                self.send_error(404)
                return

            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers['Host']}":
                self.reply_json(403, {"error": "Open the local assistant page to ask questions."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get_content_type() != "application/json":
                    raise ValueError
                payload = json.loads(self.rfile.read(length))
                question = payload["question"].strip()
                if not 0 < len(question) <= 500:
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError):
                self.reply_json(400, {"error": "Enter a question of up to 500 characters."})
                return

            reply = build_answer(question, rows, args.model, snapshot, official=True, search=search)
            body, sources = [], []
            for line in reply.splitlines():
                url = line.split(" | ")[-1].removeprefix("Source: ").strip()
                if line.startswith("Source: ") and url in known_urls:
                    if url not in sources:
                        sources.append(url)
                else:
                    body.append(line)
            self.reply_json(200, {"answer": "\n".join(body).strip(), "sources": sources})

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Open http://127.0.0.1:{server.server_port} in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
