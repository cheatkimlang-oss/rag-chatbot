"""Stage 04 — query understanding: normalize, route to an intent, extract entities.

Rules first, no LLM. When in doubt the router chooses TOPIC_SEARCH: a wrong
topic search still surfaces the right projects lexically, while a wrong
SENIOR_LOOKUP shows a confident portfolio for the wrong person.
"""
from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

TOPIC_SEARCH = "TOPIC_SEARCH"
SENIOR_LOOKUP = "SENIOR_LOOKUP"
PORTFOLIO = "PORTFOLIO"
FACET_BROWSE = "FACET_BROWSE"
ANSWER = "ANSWER"
SENIOR_TOPIC = "SENIOR_TOPIC"        # multi-intent (§5.5): "Sok Dara's OCR project"

YEAR_RE = re.compile(r"\b(20\d{2})\b")
PROGRAM_RE = re.compile(r"\b(DSE|ITE|TEE|BE|FTCE)\b", re.IGNORECASE)
QUESTION_RE = re.compile(r"^(who|what|which|when|where|why|how|has anyone|is there|are there)\b", re.I)
PORTFOLIO_RE = re.compile(r"\b(portfolio(?: of)?|projects? (?:by|of)|works? (?:by|of))\b", re.I)
POSSESSIVE_RE = re.compile(r"['’]s\b")

# Words that make a 2–4 token query look like a topic rather than a name.
TOPIC_WORDS = {"project", "projects", "thesis", "system", "using", "with", "about",
               "for", "based", "app", "application", "model", "detection", "recognition",
               "analysis", "prediction", "learning", "network", "data", "web", "mobile"}
# Removed when checking whether anything besides a name / facet remains.
FILLER_WORDS = {"project", "projects", "thesis", "by", "of", "the", "a", "an", "on", "in",
                "about", "from", "year", "work", "works", "did", "does", "do", "build",
                "built", "make", "made", "what", "who", "which", "is", "are", "show", "me",
                "all", "list", "program", "cohort", "s"}

KHMER_RANGE = ("ក", "៿")
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), " ")


def is_khmer_char(ch: str) -> bool:
    return KHMER_RANGE[0] <= ch <= KHMER_RANGE[1] or "᧠" <= ch <= "᧿"


def contains_khmer(text: str) -> bool:
    return any(is_khmer_char(ch) for ch in text)


def normalize(text: str) -> str:
    """NFKC, casefold, strip Latin diacritics, collapse whitespace — Khmer preserved.

    Diacritic stripping is applied per character to non-Khmer text only: Khmer
    vowel signs are combining marks (Mn), and stripping them would destroy the name.
    """
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).casefold()
    out: list[str] = []
    for ch in t:
        if is_khmer_char(ch):
            out.append(ch)
        else:
            out.extend(c for c in unicodedata.normalize("NFD", ch)
                       if unicodedata.category(c) != "Mn")
    return " ".join(unicodedata.normalize("NFC", "".join(out)).split())


def segment_for_lexical(text: str) -> str:
    """Khmer has no spaces; the sparse encoder needs word boundaries."""
    if not contains_khmer(text):
        return text
    try:
        from khmernltk import word_tokenize
    except ImportError:
        log.warning("khmer-nltk not installed; Khmer lexical matching will be weak")
        return text
    return " ".join(tok for tok in word_tokenize(text) if tok.strip())


def find_name(norm: str, known_names: set[str]) -> str | None:
    """Longest known name/alias that occurs as a whole-word span of the query."""
    padded = f" {norm} "
    best = None
    for name in known_names:
        if f" {name} " in padded and (best is None or len(name) > len(best)):
            best = name
    return best


def _strip_punct(text: str) -> str:
    # Category-based, not `[^\w]`: Khmer vowel signs are not `\w` in Python's re.
    return " ".join("".join(" " if unicodedata.category(ch)[0] in "PS" else ch
                            for ch in text).split())


def _residual(tokens: list[str], drop: set[str]) -> list[str]:
    return [t for t in tokens if t not in drop and t not in FILLER_WORDS]


def route(query: str, known_names: set[str], known_surnames: set[str]) -> dict:
    norm = normalize(query)
    years = [int(y) for y in YEAR_RE.findall(norm)]
    programs = [p.upper() for p in PROGRAM_RE.findall(norm)]
    # Possessives and punctuation are noise for name matching ("sok dara's" -> "sok dara").
    name_text = _strip_punct(POSSESSIVE_RE.sub("", norm))
    tokens = name_text.split()
    facet_tokens = {str(y) for y in years} | {p.lower() for p in programs}

    intent, name, topic, heuristic = TOPIC_SEARCH, None, None, False

    if PORTFOLIO_RE.search(norm):
        intent = PORTFOLIO
        name = " ".join(t for t in PORTFOLIO_RE.sub(" ", name_text).split()
                        if t not in FILLER_WORDS) or None
    elif name_text in known_names:
        intent, name = SENIOR_LOOKUP, name_text
    elif (found := find_name(name_text, known_names)):
        rest = _residual(name_text.replace(found, " ").split(), facet_tokens)
        rest = [t for t in rest if not QUESTION_RE.match(t)]
        name = found
        if rest:
            intent, topic = SENIOR_TOPIC, " ".join(rest)
        else:
            intent = SENIOR_LOOKUP
    elif QUESTION_RE.search(norm):
        intent = ANSWER
    elif (years or programs) and not _residual(tokens, facet_tokens):
        intent = FACET_BROWSE
    elif 2 <= len(tokens) <= 4 and not (set(tokens) & TOPIC_WORDS) and not facet_tokens:
        if set(tokens) & known_surnames:
            intent, name, heuristic = SENIOR_LOOKUP, name_text, True

    topic_text = topic if topic is not None else query
    return {
        "raw": query,
        "normalized": norm,
        "intent": intent,
        "name": name,
        "name_heuristic": heuristic,   # routed by surname guess, not an exact directory hit
        "topic": topic,
        "lexical": segment_for_lexical(topic_text),
        "years": years,
        "programs": programs,
    }
