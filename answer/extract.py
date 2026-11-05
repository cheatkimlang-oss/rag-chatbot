"""Extractive answer selection: the sentence(s) in the retrieved chunks that answer the question.

Retrieval returns ~700-char chunks; a student asking "which university do I
study at?" wants the one sentence that says so. This picks it without any
generation, so every word shown is verbatim from the source (FR-09 default
mode, no hallucination surface):

  1. split the top chunks into units — sentences, bullet items, table rows
     (PDF hard-wrapped lines are re-joined first; headings are kept only as labels);
  2. score each unit against the question — with a cross-encoder trained on
     question/answer pairs when available (CrossEncoderScorer), otherwise
         score = dense cosine(question, unit) + LEXICAL_WEIGHT * keyword overlap;
  3. answer = best unit, widened to its whole bullet list or to adjacent
     sentences of the same paragraph that score almost as well.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

LEXICAL_WEIGHT = 0.20
MIN_UNIT_CHARS = 25
MAX_BULLETS = 8
# An adjacent sentence joins the answer if it scores within this of the best
# (cross-encoder logits vs. cosine — different scales).
NEIGHBOUR_MARGIN = {"cross": 1.0, "dense": 0.04}
# Bullet answers: a question naming the list's heading ("what are my goals?" vs
# "Goals and Interests") gets the whole list; otherwise only the bullets scoring
# within this of the best ("what exam am I preparing for?" -> just the IELTS one).
LIST_MARGIN = {"cross": 2.0, "dense": 0.05}
MIN_SENTENCE_WORDS = 6        # shorter "sentences" are titles / TOC entries, not answers
MIN_BULLET_WORDS = 3
CHUNKS_PER_CARD = 10          # all fused chunks: the answer may sit in a project's 5th-best chunk
CARDS_SEARCHED = 3


class CrossEncoderScorer:
    """Question->passage relevance (MS MARCO cross-encoder via fastembed, ~80 MB, CPU)."""
    kind = "cross"

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from ingestion.embedder import FASTEMBED_CACHE

        self.model = TextCrossEncoder(model, cache_dir=FASTEMBED_CACHE)
        self.revision = model

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(question, texts)]


def make_scorer(model: str | None):
    if not model:
        return None
    try:
        return CrossEncoderScorer(model)
    except Exception as exc:          # not installed / offline on first use
        log.warning("answer scorer %s unavailable, using dense+keyword scoring: %s", model, exc)
        return None

_LEADING_NUM = re.compile(r"^\s*\d+(\.\d+)*[.)]?\s+")
BULLET_RE = re.compile(r"^\s*([•▪◦●○■□➢►*\-–]|\d{1,2}[.)])\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an the and or of to in on at for from by with about as is are was were be been
being am do does did i me my we our you your he she it its they them their this that these those
what which who whom whose when where why how can could should would will shall may might must
there here tell give show please any some all much many more most also so than then too very
just not no yes into over under up down out""".split())


@dataclass
class Unit:
    text: str             # as displayed (bullet marker normalized to "•")
    kind: str             # sentence | bullet | table | heading
    block: int            # paragraph / bullet-run index inside its chunk
    order: int            # global order inside its chunk
    card: dict
    chunk: dict
    context: str | None = None   # nearest heading above it — "Goals and Interests" makes
                                 # "Improve my English" an answer to "what are my goals?"

    @property
    def scored_text(self) -> str:
        return f"{self.context}: {self.text}" if self.context else self.text


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def _stems(text: str) -> set[str]:
    """Content words with plural/verb endings cut: 'goals' ~ 'goal', 'skills' ~ 'skill'."""
    return {re.sub(r"(ies|es|s|ing|ed)$", "", w) or w for w in _content_words(text)}


def _blocks(text: str) -> list[tuple[str, str]]:
    """Re-join hard-wrapped lines into (kind, text) blocks."""
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines: list[str] = []
    for ln in raw:                                  # a lone "•" belongs to the next line
        if lines and BULLET_RE.match(lines[-1] + " x") and len(lines[-1]) <= 3:
            lines[-1] = f"{lines[-1]} {ln}"
        else:
            lines.append(ln)
    out: list[list] = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if BULLET_RE.match(ln):
            out.append(["bullet", BULLET_RE.sub("", ln)])
        elif " | " in ln:                           # table rows: one table = one unit
            if out and out[-1][0] == "table":
                out[-1][1] += "\n" + ln
            else:
                out.append(["table", ln])
        elif (len(ln) < 60 and not ln.endswith((".", "!", "?", ",", ";", ":"))
              and (not nxt or nxt[:1].isupper() or BULLET_RE.match(nxt))):
            out.append(["heading", ln])
        elif out and out[-1][0] in ("sentence", "bullet") and not out[-1][1].endswith((".", "!", "?", ":")):
            out[-1][1] += " " + ln          # continuation of a wrapped line
        else:
            out.append(["sentence", ln])
    return [(k, "\n".join(" ".join(row.split()) for row in t.splitlines()) if k == "table"
             else " ".join(t.split())) for k, t in out]


def _strip_breadcrumb(chunk: dict) -> str:
    text, crumb = chunk.get("text", ""), chunk.get("breadcrumb")
    if crumb and text.startswith(crumb):
        text = text[len(crumb):]
    return text.strip()


def units_of(card: dict, chunk: dict) -> list[Unit]:
    units, order = [], 0
    text = _strip_breadcrumb(chunk)
    heading = chunk.get("heading")
    if heading and text.startswith(heading):        # "1. Executive Summary" is a label, not a bullet
        text = text[len(heading):].strip()
    # Context = page/section title + nearest sub-heading: "Our Goal: The RUPP Student
    # Chatbot — THE SOLUTION" lets that slide's bullet answer "what is the RUPP chatbot?".
    title = _LEADING_NUM.sub("", heading).strip() if heading else None
    local = None
    for b, (kind, text) in enumerate(_blocks(text)):
        if kind == "heading":
            local = text
        context = " — ".join(dict.fromkeys(c for c in (title, local) if c)) or None
        pieces = SENTENCE_RE.split(text) if kind == "sentence" else [text]
        for piece in pieces:
            piece = piece.strip()
            if piece:
                units.append(Unit(piece, kind, b, order, card, chunk,
                                  context=None if kind == "heading" else context))
                order += 1
    return units


def _answerable(u: Unit) -> bool:
    words = len(u.text.split())
    if (u.kind == "sentence" and words < MIN_SENTENCE_WORDS) or (u.kind == "bullet" and words < MIN_BULLET_WORDS):
        return False
    return (u.kind != "heading" and (len(u.text) >= MIN_UNIT_CHARS or u.kind == "bullet")
            and not u.text.endswith("?")                        # a slide title asking the question
            and u.text != (u.chunk.get("heading") or ""))


def _score_units(question: str, units: list[Unit], embedder, scorer) -> tuple[list[float], str]:
    if scorer is not None:
        try:
            return scorer.score(question, [u.scored_text for u in units]), "cross"
        except Exception as exc:
            log.warning("answer scorer failed, falling back: %s", exc)
    encode_q = getattr(embedder, "encode_query", embedder.encode)
    qv = encode_q([question])[0][0]
    vecs = embedder.encode([u.scored_text for u in units])[0]
    q_words = _content_words(question)
    out = []
    for u, v in zip(units, vecs):
        cos = sum(a * b for a, b in zip(qv, v))               # both L2-normalized
        overlap = len(q_words & _content_words(u.scored_text)) / len(q_words) if q_words else 0.0
        out.append(cos + LEXICAL_WEIGHT * overlap)
    return out, "dense"


def extract_answer(question: str, cards: list[dict], embedder, scorer=None) -> dict | None:
    """Best verbatim answer from the top cards, or None if nothing is answerable."""
    all_units: list[Unit] = []
    seen: set[str] = set()
    for card in cards[:CARDS_SEARCHED]:
        for chunk in (card.get("evidence") or [])[:CHUNKS_PER_CARD]:
            for u in units_of(card, chunk):
                key = u.text.lower()
                if key not in seen:                             # chunk overlap repeats text
                    seen.add(key)
                    all_units.append(u)
    candidates = [u for u in all_units if _answerable(u)]
    if not candidates:
        return None

    raw, how = _score_units(question, candidates, embedder, scorer)
    scores = {id(u): s for u, s in zip(candidates, raw)}
    best = max(candidates, key=lambda u: scores[id(u)])
    best_score = scores[id(best)]
    margin = NEIGHBOUR_MARGIN[how]

    same_chunk = [u for u in all_units if u.chunk is best.chunk]
    if best.kind == "bullet":
        # Only the bullets that answer the question, led by their heading.
        run = [u for u in same_chunk if u.kind == "bullet" and _contiguous_bullets(same_chunk, best, u)]
        lead_unit = _heading_before(same_chunk, run[0]) if run else None
        lead = lead_unit.text if lead_unit else None
        asks_for_list = bool(lead and _stems(question) & _stems(lead))
        keep = [u for u in run if asks_for_list or (
            id(u) in scores and scores[id(u)] >= best_score - LIST_MARGIN[how])][:MAX_BULLETS]
        if len(keep) == 1:
            keep = [best]
            text = f"{lead}: {best.text}" if lead else best.text
        else:
            text = (f"{lead}:\n" if lead else "") + "\n".join(f"• {u.text}" for u in keep)
        used = ([lead_unit] if lead_unit else []) + keep
    else:
        picked = [best]
        for u in same_chunk:                                   # adjacent, nearly as good
            if (u is not best and u.block == best.block and abs(u.order - best.order) == 1
                    and id(u) in scores and scores[id(u)] >= best_score - margin):
                picked.append(u)
        used = sorted(picked, key=lambda u: u.order)
        text = " ".join(u.text for u in used)

    return {"text": text, "score": round(best_score, 4), "scored_by": how, "card": best.card,
            "page_label": best.chunk.get("page_label") or f"p.{best.chunk.get('page_number')}",
            "page": best.chunk.get("page_number"), "heading": best.chunk.get("heading"),
            "chunk_id": best.chunk.get("chunk_id"),
            # Every unit the answer is assembled from, with the chunk it must occur in:
            # the input to answer/validate.verify_quotes.
            "quotes": [{"quote": u.text, "chunk_id": u.chunk.get("chunk_id"),
                        "chunk_text": u.chunk.get("text", "")} for u in used]}


def _contiguous_bullets(units: list[Unit], anchor: Unit, u: Unit) -> bool:
    """True if `u` is in the same unbroken run of bullets as `anchor`."""
    lo, hi = sorted((anchor.order, u.order))
    return all(x.kind == "bullet" for x in units if lo <= x.order <= hi)


def _heading_before(units: list[Unit], first: Unit) -> Unit | None:
    prev = [u for u in units if u.order == first.order - 1]
    return prev[0] if prev and prev[0].kind == "heading" else None
