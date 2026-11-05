"""Gemma 4 synthesis via Ollama (FR-09). Only ever called above the relevance gate.

The prompt receives only the retrieved passages (top projects x top chunks,
~2–3K tokens) — Gemma's long window is not an excuse to dump the corpus.
Contact details and other personal data are never placed in this prompt.
"""
from __future__ import annotations

import html

import httpx

SYSTEM = """You answer questions about RUPP documents and past senior projects using ONLY
the passages inside <retrieved_projects>.

Rules:
1. Use only the provided passages. Do not add outside knowledge.
2. Cite every factual sentence as [Project: <title>, p.<page>], copying the title
   and page number exactly from the passage you used.
3. If the passages do not answer the question, output exactly:
   I do not know.
   Do not guess, and do not answer a different question instead.
4. Never invent a senior name, supervisor, year, number, or repository link: every
   number and name you write must appear in the passage you cite.
5. Keep the answer under 900 characters — it is delivered as a Telegram message.
6. Answer directly. When a passage is written in the first person ("I am ..."),
   answer the user in the second person ("You are ...")."""

PASSAGES_PER_PROJECT = 3


def passages_for(cards: list[dict], per_project: int = PASSAGES_PER_PROJECT) -> list[dict]:
    """The evidence Gemma may use — and therefore the only (title, page) pairs it may cite."""
    out = []
    for c in cards:
        for ev in (c.get("evidence") or [])[:per_project]:
            text, crumb = ev.get("text", ""), ev.get("breadcrumb")
            if crumb and text.startswith(crumb):
                text = text[len(crumb):]
            out.append({"project_id": c["project_id"], "title": c["title"], "source": c.get("source"),
                        "page": ev.get("page_number"), "page_label": ev.get("page_label"),
                        "heading": ev.get("heading"), "snippet": text.strip()})
    return out


def build_context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"<project title=\"{html.escape(p['title'])}\" page=\"{p['page']}\">\n{p['snippet']}\n</project>"
        for p in passages
    )


def _options() -> dict:
    return {
        "temperature": 0.0,
        "top_p": 0.9,
        "num_ctx": 8192,      # MUST be explicit; Ollama's default truncates evidence
        "num_predict": 600,   # keeps the reply inside one Telegram message
    }


def synthesize(query: str, passages: list[dict], *, ollama_url: str, model: str,
               timeout: float = 180.0) -> str:
    r = httpx.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"<retrieved_projects>\n{build_context(passages)}\n</retrieved_projects>\n\n{query}"},
            ],
            "stream": False,
            "think": False,           # no hidden reasoning tokens: faster, deterministic
            "keep_alive": "30m",
            "options": _options(),
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def model_available(*, ollama_url: str, model: str, timeout: float = 3.0) -> bool:
    """True if Ollama is running and the pinned tag is pulled (never pulls by itself)."""
    try:
        r = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError:
        return False
    return any(m.get("name") == model or m.get("model") == model for m in r.json().get("models", []))


def warm_up(*, ollama_url: str, model: str, timeout: float = 300.0) -> None:
    """Load the model into memory before the first question (pre-demo checklist)."""
    httpx.post(f"{ollama_url.rstrip('/')}/api/generate",
               json={"model": model, "prompt": "", "keep_alive": "30m", "options": _options()},
               timeout=timeout).raise_for_status()
