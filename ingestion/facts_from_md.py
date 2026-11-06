"""Turn a degree-specification Markdown file into plain sentences the answerer can quote.

    python -m ingestion.facts_from_md data/corpus/data_science_engineering_degree_specification.md --short DSE

Writes <stem>_facts.md next to the input. The answer layer serves ONE verbatim
sentence, so table rows ("| **I** | Artificial Intelligence | 3.0 | (3-0) |") and bare
list items ("* Data Analyst") make poor answers. This reads the file's own tables and
lists and restates each fact as a sentence that names the program. Nothing is added
that the source file does not say, except credit totals summed from its tables.

Expected source layout (as in the DSE specification):
    # <program name>
    ### <Year heading>  + table | Semester | Subject | Credits | Hours |
    a "Career" heading followed by "* job" items
    a "Research" heading followed by "* **Domain:** description" items
    a grading table | Score Range | GPA | Letter Grade | Standing |
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

YEAR_WORDS = {"foundation": "first year (Foundation Year)", "second": "second year",
              "third": "third year", "fourth": "fourth year"}
SEM_NUM = {"I": "1", "II": "2"}


def _clean(cell: str) -> str:
    return re.sub(r"\*\*|__", "", cell).strip()


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        if not ln.strip().startswith("|"):
            continue
        cells = [_clean(c) for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):      # separator row
            continue
        rows.append(cells)
    return rows[1:]                                                 # drop the header row


def _sections(text: str) -> list[tuple[int, str, list[str]]]:
    """(level, heading, body lines) for every #-heading."""
    out, cur = [], None
    for ln in text.splitlines():
        m = re.match(r"(#{1,6})\s+(.*)", ln)
        if m:
            cur = (len(m.group(1)), _clean(m.group(2)), [])
            out.append(cur)
        elif cur:
            cur[2].append(ln)
    return out


def _n(x: float) -> str:
    return f"{x:g}"


def _credits(c: float) -> str:
    return f"{_n(c)} credit{'' if c == 1 else 's'}"


def build(text: str, short: str) -> str:
    secs = _sections(text)
    full = next(h for lvl, h, _ in secs if lvl == 1)
    L = [f"# {full}: facts", "",
         f"Generated from the {full} specification. {short} stands for "
         f"{re.sub(r'^(Bachelor|Master) of ', '', full.split('(')[0].strip())}.", ""]

    for _, head, body in secs:
        bullets = [_clean(re.sub(r"^\s*[*-]\s+", "", b)) for b in body if re.match(r"\s*[*-]\s+", b)]
        low = head.lower()
        if "career" in low and bullets:
            L += [f"## {short} careers and jobs", "",
                  f"{short} graduates can get jobs as: {', '.join(bullets)}.",
                  f"Graduates of the {full} are qualified for roles including {', '.join(bullets)}.", ""]
        elif "research" in low and bullets:
            names = [b.split(":")[0].strip() for b in bullets]
            L += [f"## {short} research domains", "",
                  f"The research domains of the {full} are: {', '.join(names)}."]
            L += [f"In {short}, the research domain {b.split(':')[0].strip()} covers "
                  f"{b.split(':', 1)[1].strip().rstrip('.')}." for b in bullets if ":" in b]
            L.append("")
        elif bullets and any(":" in b for b in bullets):  # "* **Label:** text" overview items
            L += [f"## {short} {head.split('.', 1)[-1].strip()}", ""]
            for b in bullets:
                if ":" in b:
                    label, desc = (p.strip() for p in b.split(":", 1))
                    L.append(f"{short} {label.lower()}: {desc}")
            L.append("")

        numbered = [_clean(re.sub(r"^\s*\d+\.\s+", "", b)) for b in body if re.match(r"\s*\d+\.\s+", b)]
        if "track" in low and numbered:
            names = [n.split(":")[0].strip() for n in numbered]
            L += [f"## {short} final semester tracks", "",
                  f"In the final semester (fourth year, Semester II) of the {short} program, students "
                  f"choose one of {len(names)} tracks: {', '.join(names)}.",
                  f"The {short} final semester tracks are {', '.join(names)}."]
            L += [f"The {short} {n.split(':')[0].strip()} is {n.split(':', 1)[1].strip()[0].lower()}"
                  f"{n.split(':', 1)[1].strip()[1:]}" for n in numbered if ":" in n]
            if not re.search(r"GPA\s*[><≥]|GPA of|GPA above", "\n".join(body)):
                L.append(f"The {short} specification does not state a GPA requirement for any of "
                         f"the final semester tracks.")
            L.append("")

    for _, head, body in secs:                       # grading table
        rows = _table_rows(body)
        if rows and len(rows[0]) == 4 and "grad" in head.lower():
            L += [f"## {short} grading and GPA", ""]
            for score, gpa, letter, meaning in rows:
                L.append(f"In the {short} program, a score of {score.replace(' ', '')} earns grade "
                         f"{letter} ({meaning}), worth {gpa} grade points.")
            L.append("")

    total, year_lines = 0.0, []
    where: dict[str, list[str]] = {}
    credit_of: dict[str, float] = {}
    for _, head, body in secs:                       # curriculum tables
        key = head.split()[0].lower()
        if key not in YEAR_WORDS:
            continue
        rows = _table_rows(body)
        by_sem: dict[str, list[tuple[str, float]]] = {}
        for sem, subject, credit, _hours in rows:
            by_sem.setdefault(sem, []).append((subject, float(credit)))
        year_lines += [f"## {short} {head} courses", ""]
        for sem, courses in by_sem.items():
            sub = sum(c for _, c in courses)
            total += sub
            listing = "; ".join(f"{s} ({_credits(c)})" for s, c in courses)
            year = YEAR_WORDS[key]
            year_lines.append(f"{short} {year} semester {SEM_NUM.get(sem, sem)} courses: in Semester {sem} "
                              f"of the {year} of the {full}, students take {len(courses)} courses worth "
                              f"{_n(sub)} credits: {listing}.")
            for s, c in courses:
                where.setdefault(s, []).append(f"Semester {sem} of the {year}")
                credit_of[s] = c
                if c >= 5:                            # big items people ask about directly
                    year_lines.append(f"In the {short} program, {s} is worth {_credits(c)} "
                                      f"(Semester {sem} of the {year}).")
        year_lines.append("")
    L += year_lines
    if where:                                         # "when do I study X?"
        L += [f"## When {short} courses are taught", ""]
        L += [f"In the {short} degree, {s} is taught in {', '.join(p)}, worth {_credits(credit_of[s])}."
              for s, p in where.items()]
        L.append("")
    L += [f"## {short} total credits", "",
          f"The {short} degree totals {_n(total)} credits across all listed courses "
          f"(summed from the curriculum tables).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("--short", required=True, help="program abbreviation, e.g. DSE")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    src = Path(args.source)
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_facts.md")
    out.write_text(build(src.read_text(encoding="utf-8"), args.short), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
