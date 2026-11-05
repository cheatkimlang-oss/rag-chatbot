"""SQLite metadata store (Architecture Spec §7.2).

The authoritative operational view of seniors, projects, authorship and consent.
Portfolios and facet counts are served from here, so they keep working when
Qdrant is down. Consent lives here and only here — never in Qdrant payloads.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS senior (
  senior_id              TEXT PRIMARY KEY,
  full_name_latin        TEXT NOT NULL,
  full_name_khmer        TEXT,
  name_normalized        TEXT NOT NULL,
  program                TEXT NOT NULL,
  cohort_year            INTEGER NOT NULL,
  supervisor_id          TEXT,
  contact_email          TEXT,
  github                 TEXT,
  linkedin               TEXT,
  website                TEXT,
  consent_public_contact INTEGER NOT NULL DEFAULT 0,
  consent_relay          INTEGER NOT NULL DEFAULT 0,
  telegram_user_hash     TEXT,
  status                 TEXT NOT NULL DEFAULT 'active',
  created_at             TEXT NOT NULL,
  updated_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS senior_alias (
  senior_id        TEXT NOT NULL REFERENCES senior(senior_id),
  alias_normalized TEXT NOT NULL,
  script           TEXT NOT NULL,
  PRIMARY KEY (senior_id, alias_normalized)
);
CREATE INDEX IF NOT EXISTS idx_alias ON senior_alias(alias_normalized);

CREATE TABLE IF NOT EXISTS project (
  project_id    TEXT PRIMARY KEY,
  title_en      TEXT NOT NULL,
  title_khmer   TEXT,
  abstract      TEXT NOT NULL,
  year          INTEGER NOT NULL,
  term          TEXT,
  program       TEXT NOT NULL,
  domain        TEXT,
  tech_stack    TEXT,
  keywords      TEXT,
  repo_url      TEXT,
  demo_url      TEXT,
  supervisor_id TEXT,
  permission    TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authorship (
  project_id TEXT NOT NULL REFERENCES project(project_id),
  senior_id  TEXT NOT NULL REFERENCES senior(senior_id),
  role       TEXT,
  PRIMARY KEY (project_id, senior_id)
);

CREATE TABLE IF NOT EXISTS document (
  doc_id       TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES project(project_id),
  doc_type     TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  page_count   INTEGER,
  parser       TEXT,
  parse_status TEXT NOT NULL,
  chunk_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_log (
  query_id         TEXT PRIMARY KEY,
  surface          TEXT NOT NULL DEFAULT 'telegram',
  tg_user_hash     TEXT,
  tg_message_id    INTEGER,
  raw_query        TEXT NOT NULL,
  normalized_query TEXT NOT NULL,
  intent           TEXT NOT NULL,
  resolved_senior  TEXT,
  retrieved_json   TEXT NOT NULL,
  clicked_project  TEXT,
  rating           INTEGER,
  latency_ms       INTEGER,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qlog_time ON query_log(created_at);

CREATE TABLE IF NOT EXISTS kb_meta (               -- index identity (embedder revision, ...)
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relay_message (
  relay_id   TEXT PRIMARY KEY,
  senior_id  TEXT NOT NULL REFERENCES senior(senior_id),
  from_hash  TEXT NOT NULL,
  body       TEXT NOT NULL,
  status     TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

CONTACT_FIELDS = ("github", "linkedin", "website", "contact_email")
NO_CONSENT_NOTICE = ("This senior has not opted in to public contact. "
                     "Ask your department secretariat to forward a message.")
FACET_COLUMNS = {"year": "p.year", "program": "p.program",
                 "supervisor_id": "p.supervisor_id", "domain": "p.domain"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


class KBStore:
    def __init__(self, path: str | Path = "data/kb.sqlite3"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes
    # Used by ingestion (and test fixtures). Rows arrive already validated.

    def upsert_senior(self, row: dict, aliases: list[tuple[str, str]] = ()) -> None:
        now = _now()
        cols = ["senior_id", "full_name_latin", "full_name_khmer", "name_normalized",
                "program", "cohort_year", "supervisor_id", "contact_email", "github",
                "linkedin", "website", "consent_public_contact", "consent_relay",
                "telegram_user_hash", "status"]
        values = {c: row.get(c) for c in cols}
        values["consent_public_contact"] = int(bool(row.get("consent_public_contact")))
        values["consent_relay"] = int(bool(row.get("consent_relay")))
        values["status"] = row.get("status") or "active"
        values["created_at"] = values["updated_at"] = now
        names = list(values)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c != "created_at")
        self.conn.execute(
            f"INSERT INTO senior ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
            f"ON CONFLICT(senior_id) DO UPDATE SET {updates}",
            [values[c] for c in names],
        )
        self.conn.execute("DELETE FROM senior_alias WHERE senior_id = ?", (row["senior_id"],))
        self.conn.executemany(
            "INSERT OR IGNORE INTO senior_alias (senior_id, alias_normalized, script) VALUES (?, ?, ?)",
            [(row["senior_id"], alias, script) for alias, script in aliases],
        )
        self.conn.commit()

    def upsert_project(self, row: dict, authors: list[tuple[str, str | None]]) -> None:
        cols = ["project_id", "title_en", "title_khmer", "abstract", "year", "term",
                "program", "domain", "tech_stack", "keywords", "repo_url", "demo_url",
                "supervisor_id", "permission"]
        values = {c: row.get(c) for c in cols}
        for c in ("tech_stack", "keywords"):
            if isinstance(values[c], (list, tuple)):
                values[c] = json.dumps(list(values[c]), ensure_ascii=False)
        values["created_at"] = _now()
        names = list(values)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c != "created_at")
        self.conn.execute(
            f"INSERT INTO project ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
            f"ON CONFLICT(project_id) DO UPDATE SET {updates}",
            [values[c] for c in names],
        )
        self.conn.execute("DELETE FROM authorship WHERE project_id = ?", (row["project_id"],))
        self.conn.executemany(
            "INSERT INTO authorship (project_id, senior_id, role) VALUES (?, ?, ?)",
            [(row["project_id"], sid, role) for sid, role in authors],
        )
        self.conn.commit()

    def upsert_document(self, row: dict) -> None:
        cols = ["doc_id", "project_id", "doc_type", "file_path", "sha256", "page_count",
                "parser", "parse_status", "chunk_count"]
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "doc_id")
        self.conn.execute(
            f"INSERT INTO document ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT(doc_id) DO UPDATE SET {updates}",
            [row.get(c) for c in cols],
        )
        self.conn.commit()

    def documents(self) -> dict[str, dict]:
        return {r["doc_id"]: dict(r) for r in self.conn.execute("SELECT * FROM document")}

    def project_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT project_id FROM project")}

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM document WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def delete_project(self, project_id: str) -> None:
        for table in ("document", "authorship", "project"):
            self.conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
        self.conn.commit()

    def reset_catalog(self) -> None:
        """--rebuild: drop the derived catalog. query_log and relay_message are kept."""
        for table in ("document", "authorship", "project", "senior_alias", "senior", "kb_meta"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        r = self.conn.execute("SELECT value FROM kb_meta WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO kb_meta (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------- name lookup

    def lookup_alias(self, normalized: str) -> list[str]:
        """Exact match on curated aliases or the canonical normalized name.

        Returns a list on purpose: two seniors may share a name, and merging
        them is the one mistake this product must never make.
        """
        rows = self.conn.execute(
            """SELECT DISTINCT s.senior_id FROM senior s
               LEFT JOIN senior_alias a ON a.senior_id = s.senior_id
               WHERE s.status = 'active'
                 AND (s.name_normalized = ? OR a.alias_normalized = ?)
               ORDER BY s.senior_id""",
            (normalized, normalized),
        ).fetchall()
        return [r["senior_id"] for r in rows]

    def name_directory(self) -> list[tuple[str, str]]:
        """(senior_id, name) for every canonical name and alias of active seniors."""
        rows = self.conn.execute(
            """SELECT senior_id, name_normalized AS name FROM senior WHERE status = 'active'
               UNION
               SELECT a.senior_id, a.alias_normalized FROM senior_alias a
               JOIN senior s ON s.senior_id = a.senior_id WHERE s.status = 'active'"""
        ).fetchall()
        return [(r["senior_id"], r["name"]) for r in rows]

    def known_names(self) -> set[str]:
        return {name for _, name in self.name_directory()}

    def known_surnames(self) -> set[str]:
        """First and last Latin tokens (Khmer order is surname-first; romanized
        variants like 'DARA Sok' flip it)."""
        out: set[str] = set()
        for _, name in self.name_directory():
            tokens = name.split()
            if len(tokens) >= 2 and tokens[0].isascii():
                out.update({tokens[0], tokens[-1]})
        return out

    def all_seniors(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM senior WHERE status = 'active'").fetchall()
        return [dict(r) for r in rows]

    def senior_exists(self, senior_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM senior WHERE senior_id = ? AND status = 'active'", (senior_id,)
        ).fetchone() is not None

    def senior_card(self, senior_id: str) -> dict:
        """Disambiguation row: enough to tell two same-named people apart."""
        r = self.conn.execute(
            "SELECT full_name_latin, full_name_khmer, program, cohort_year FROM senior WHERE senior_id = ?",
            (senior_id,),
        ).fetchone()
        if r is None:
            return {}
        return {"name": r["full_name_latin"], "name_khmer": r["full_name_khmer"],
                "program": r["program"], "cohort_year": r["cohort_year"]}

    def author_names(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.senior_id, s.full_name_latin FROM authorship a
               JOIN senior s ON s.senior_id = a.senior_id
               WHERE a.project_id = ? AND s.status = 'active'
               ORDER BY CASE a.role WHEN 'lead' THEN 0 ELSE 1 END, s.full_name_latin""",
            (project_id,),
        ).fetchall()
        return [{"senior_id": r["senior_id"], "name": r["full_name_latin"]} for r in rows]

    # ---------------------------------------------------------------- projects

    def _project_dict(self, r: sqlite3.Row) -> dict:
        p = dict(r)
        for c in ("tech_stack", "keywords"):
            try:
                p[c] = json.loads(p[c]) if p.get(c) else []
            except json.JSONDecodeError:
                p[c] = [t.strip() for t in str(p[c]).split(";") if t.strip()]
        p["authors"] = self.author_names(p["project_id"])
        p["senior_ids"] = [a["senior_id"] for a in p["authors"]]
        p["has_documents"] = self.conn.execute(
            "SELECT 1 FROM document WHERE project_id = ? AND parse_status != 'failed' AND chunk_count > 0",
            (p["project_id"],),
        ).fetchone() is not None
        return p

    def project(self, project_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM project WHERE project_id = ? AND permission = 'granted'", (project_id,)
        ).fetchone()
        return self._project_dict(r) if r else None

    def projects_by_ids(self, project_ids: list[str]) -> list[dict]:
        out = [self.project(pid) for pid in project_ids]
        return sorted((p for p in out if p), key=lambda p: (-p["year"], p["title_en"]))

    def projects_for_senior(self, senior_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.* FROM project p JOIN authorship a ON a.project_id = p.project_id
               WHERE a.senior_id = ? AND p.permission = 'granted'
               ORDER BY p.year DESC, p.title_en""",
            (senior_id,),
        ).fetchall()
        return [self._project_dict(r) for r in rows]

    # --------------------------------------------------------------- portfolio

    def portfolio(self, senior_id: str) -> dict | None:
        """Everything about one senior, consent-gated. None if unknown/withdrawn."""
        r = self.conn.execute(
            "SELECT * FROM senior WHERE senior_id = ? AND status = 'active'", (senior_id,)
        ).fetchone()
        if r is None:
            return None
        s = dict(r)
        projects = self.projects_for_senior(senior_id)
        consent = bool(s["consent_public_contact"])
        contact = {k: s[k] for k in CONTACT_FIELDS if s.get(k)} if consent else None
        return {
            "senior_id": senior_id,
            "name": s["full_name_latin"],
            "name_khmer": s["full_name_khmer"],
            "program": s["program"],
            "cohort_year": s["cohort_year"],
            "supervisor_id": s["supervisor_id"],
            "projects": projects,
            "metadata_only": not any(p["has_documents"] for p in projects),
            "contact": contact,
            "contact_notice": None if consent else NO_CONSENT_NOTICE,
            # Relay needs consent AND the senior having started the bot.
            "relay_available": bool(s["consent_relay"]) and bool(s["telegram_user_hash"]),
        }

    # ------------------------------------------------------------------ facets

    def _facet_where(self, facets: dict) -> tuple[str, list]:
        clauses, params = ["p.permission = 'granted'"], []
        for key, col in FACET_COLUMNS.items():
            values = _as_list(facets.get(key))
            if values:
                clauses.append(f"{col} IN ({', '.join('?' * len(values))})")
                params.extend(values)
        for tech in _as_list(facets.get("tech_stack")):
            clauses.append("EXISTS (SELECT 1 FROM json_each(p.tech_stack) WHERE lower(value) = lower(?))")
            params.append(tech)
        return " AND ".join(clauses), params

    def list_projects(self, facets: dict) -> list[dict]:
        where, params = self._facet_where(facets)
        rows = self.conn.execute(
            f"SELECT p.* FROM project p WHERE {where} ORDER BY p.year DESC, p.title_en", params
        ).fetchall()
        return [self._project_dict(r) for r in rows]

    def facet_counts(self, facets: dict) -> dict:
        """Counts describe the corpus under the active filters, not the result page."""
        where, params = self._facet_where(facets)
        total = self.conn.execute(f"SELECT COUNT(*) FROM project p WHERE {where}", params).fetchone()[0]
        out = {"total": total}
        for key in ("year", "program"):
            rows = self.conn.execute(
                f"SELECT {FACET_COLUMNS[key]} AS v, COUNT(*) AS n FROM project p WHERE {where} "
                f"GROUP BY v ORDER BY v DESC", params,
            ).fetchall()
            out[key] = {r["v"]: r["n"] for r in rows}
        return out

    # ----------------------------------------------------------------- logging

    def rate_query(self, query_id: str, rating: int, clicked_project: str | None = None) -> None:
        """Feedback on a logged answer: rating -1 | 0 | 1, optionally the project it confirms."""
        self.conn.execute(
            "UPDATE query_log SET rating = ?, clicked_project = COALESCE(?, clicked_project) "
            "WHERE query_id = ?", (rating, clicked_project, query_id))
        self.conn.commit()

    def log_query(self, **row) -> None:
        row.setdefault("created_at", _now())
        cols = list(row)
        self.conn.execute(
            f"INSERT INTO query_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        self.conn.commit()
