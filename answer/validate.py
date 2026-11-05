"""Deterministic grounding checks — deliberately dumb, and not another model.

Two answer paths, two checks:

  extractive  `verify_quotes`   every quoted unit must occur verbatim (modulo
                                whitespace) in the chunk it is attributed to.
  Gemma       `validate_citations` + `check_support`
                                every cited (title, page) must be a passage Gemma
                                was shown, every factual sentence must carry (or
                                inherit) a citation, and each sentence must be
                                backed by its cited passage: all its numbers and
                                proper names appear there, and most of its content
                                words do. A draft that fails is discarded.

None of this proves an answer is *true*; it proves every claim is *in the
source text* — which is exactly the promise "answers restricted to verified
source text" makes. Anything else becomes "I do not know."
"""
from __future__ import annotations

import re

CITE_RE = re.compile(r"\[Project:\s*(.+?),\s*p\.\s*(\d+)\]")
IDK = "I do not know."
# Older prompt wording; still recognised so a cached model reply is not misread.
LEGACY_NOT_COVERED = "No indexed project covers this question."

SUPPORT_MIN_COVERAGE = 0.6      # share of a sentence's content words found in its passage
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(•\-])|\n+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[a-z0-9]+")
_PROPER_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_STOP = set("""a an the and or of to in on at for from by with about as is are was were be been being
am do does did i me my we our you your he she it its they them their this that these those what which
who whom whose when where why how can could should would will shall may might must there here also so
than then too very just not no yes into over under up down out has have had uses used use using based
both each such other only more most some any all one two""".split())


def norm_title(t: str) -> str:
    return " ".join(t.split()).casefold()


def _squash(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def _stems(text: str) -> set[str]:
    return {re.sub(r"(ies|es|s|ing|ed|ly)$", "", w) or w
            for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1}


# ------------------------------------------------------------- extractive

def verify_quotes(quotes: list[dict]) -> list[str]:
    """[] if every quote is verbatim in its chunk, else one problem per quote that is not."""
    problems = []
    for q in quotes:
        if _squash(q["quote"]) not in _squash(q["chunk_text"]):
            problems.append(f"not verbatim in {q.get('chunk_id')}: {q['quote'][:80]!r}")
    return problems


# ----------------------------------------------------------------- Gemma

def is_idk(text: str) -> bool:
    t = (text or "").strip().rstrip(".").casefold()
    return t in (IDK.rstrip(".").casefold(), LEGACY_NOT_COVERED.rstrip(".").casefold())


def validate_citations(text: str, citations: list[dict]) -> bool:
    allowed = {(norm_title(c["title"]), str(c["page"])) for c in citations}
    found = {(norm_title(t), p) for t, p in CITE_RE.findall(text or "")}
    if not found:
        return False
    return found <= allowed      # no invented sources, no invented pages


def _sentences(draft: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """(claim text, its citations). An uncited sentence inherits the next citation in
    its paragraph ("A. B [Project: X, p.2]" -> both from X p.2), else the previous one."""
    out = []
    for para in re.split(r"\n\s*\n", draft.strip()):
        parts = [p.strip() for p in _SENTENCE_RE.split(para) if p and p.strip()]
        rows = [(CITE_RE.sub("", p).strip(" •-"), [(norm_title(t), n) for t, n in CITE_RE.findall(p)])
                for p in parts]
        for i, (claim, cites) in enumerate(rows):
            if not cites:
                later = next((c for _, c in rows[i + 1:] if c), None)
                earlier = next((c for _, c in reversed(rows[:i]) if c), None)
                cites = later or earlier or []
            out.append((claim, cites))
    return out


def check_support(draft: str, passages: list[dict]) -> list[str]:
    """[] if every factual sentence is backed by the passage(s) it cites."""
    by_key: dict[tuple[str, str], str] = {}
    for p in passages:
        key = (norm_title(p["title"]), str(p["page"]))
        by_key[key] = by_key.get(key, "") + "\n" + p["snippet"]
    problems = []
    for claim, cites in _sentences(draft):
        content = _stems(claim)
        if len(content) < 2:                  # connective / heading-like fragment, no claim
            continue
        if not cites:
            problems.append(f"uncited claim: {claim[:80]!r}")
            continue
        source = "\n".join(by_key.get(c, "") for c in cites)
        if not source.strip():
            problems.append(f"cites a passage it was not shown: {claim[:80]!r}")
            continue
        src_squash, src_stems = _squash(source), _stems(source)
        missing_nums = [n for n in _NUMBER_RE.findall(claim) if n not in source]
        if missing_nums:
            problems.append(f"number(s) {missing_nums} not in the cited passage: {claim[:80]!r}")
        missing_names = [w for w in _PROPER_RE.findall(claim)
                         if w.casefold() not in src_squash and w.casefold() not in {"you", "your"}]
        if missing_names:
            problems.append(f"name(s) {missing_names} not in the cited passage: {claim[:80]!r}")
        coverage = len(content & src_stems) / len(content)
        if coverage < SUPPORT_MIN_COVERAGE:
            problems.append(f"only {coverage:.0%} of the wording is in the cited passage: {claim[:80]!r}")
    return problems
