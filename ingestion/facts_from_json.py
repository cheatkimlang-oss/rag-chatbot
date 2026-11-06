"""Turn rupp_ite_programs.json into plain sentences the extractive answerer can quote.

    python -m ingestion.facts_from_json [--json rupp_ite_programs.json] [--out data/corpus/rupp_ite_facts.md]

The answer layer serves ONE verbatim sentence from the corpus. Curriculum tables chunk
into rows like "| Thesis (GPA>3.5) | 14 | - |", which is a poor answer; this writes each
fact as a self-contained sentence under a topic heading instead. Every sentence is
generated from the JSON, so re-run this after re-scraping and nothing drifts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

YEARS = {"Foundation Year": "first year (Foundation Year)", "Second Year": "second year",
         "Third Year": "third year", "Fourth Year": "fourth year"}


def _n(x: float) -> str:
    return f"{x:g}"


def _courses(rows: list[dict]) -> str:
    return "; ".join(f"{r['subject']} ({_n(r['credit'])} credit{'' if r['credit'] == 1 else 's'})"
                     for r in rows)


def _curriculum(short: str, full: str, curr: dict) -> list[str]:
    out = []
    for year, sems in curr.items():
        out.append(f"\n## {short} {year} courses\n")
        for sem, rows in sems.items():
            if isinstance(rows, dict):                      # optional final-semester tracks
                for opt, opt_rows in rows.items():
                    total = sum(r["credit"] for r in opt_rows)
                    out.append(f"In {sem} of the {YEARS[year]} of the {full}, {opt} consists of: "
                               f"{_courses(opt_rows)}, for {_n(total)} credits in total.")
                continue
            total = sum(r["credit"] for r in rows)
            num = {"Semester I": "1", "Semester II": "2"}[sem]
            out.append(f"{short} {YEARS[year]} semester {num} courses: in {sem} of the {YEARS[year]} "
                       f"of the {full}, students take {len(rows)} courses worth {_n(total)} credits: "
                       f"{_courses(rows)}.")
    return out


def _course_index(short: str, curr: dict) -> list[str]:
    """One sentence per subject: answers "when do I study X?" / "how many credits is X?"."""
    where: dict[str, list[str]] = {}
    credit: dict[str, float] = {}
    for year, sems in curr.items():
        for sem, rows in sems.items():
            groups = rows.items() if isinstance(rows, dict) else [(None, rows)]
            for opt, rs in groups:
                for r in rs:
                    label = f"{sem} of the {YEARS[year]}" + (f" ({opt})" if opt else "")
                    where.setdefault(r["subject"], []).append(label)
                    credit[r["subject"]] = r["credit"]
    out = [f"\n## When {short} courses are taught\n"]
    for subject, places in where.items():
        c = credit[subject]
        out.append(f"In the {short} degree, {subject} is taught in {', '.join(places)}, "
                   f"worth {_n(c)} credit{'' if c == 1 else 's'}.")
    return out


def build(data: dict) -> str:
    d, ite, aie = data["department"], data["degrees"]["ITE"], data["degrees"]["AIE"]
    c = d["contact"]
    bachelors = [p["name"] for p in d["degrees_listed"]["bachelor"]]
    mite = d["degrees_listed"]["master"][0]
    L = ["# RUPP Information Technology Engineering Department: facts",
         f"\nGenerated from {data['source']} web pages retrieved {data['retrieved']}.",
         "\n## Department of Information Technology Engineering\n",
         f"The Department of Information Technology Engineering is part of the Faculty of Engineering "
         f"at the Royal University of Phnom Penh (RUPP).",
         f"The vision of the Department of Information Technology Engineering is: {d['vision']}",
         f"The Department of Information Technology Engineering offers these bachelor degrees: "
         f"{', '.join(bachelors)}.",
         f"The Department of Information Technology Engineering offers a master degree, the {mite['name']}, "
         f"which {mite['description'][0].lower()}{mite['description'][1:]}",
         f"Graduates of the Department of Information Technology Engineering can work as: "
         f"{', '.join(d['job_prospects'])}.",
         "The RUPP website does not name the head of the Department of Information Technology "
         "Engineering.",
         "\n## Contact the IT Engineering department\n",
         f"The office of the Department of Information Technology Engineering is in {c['office']}.",
         f"The phone number of the Department of Information Technology Engineering is {c['phone']}.",
         f"The email address of the Department of Information Technology Engineering is {c['email']}.",
         "\n## ITE bachelor degree overview\n",
         f"The {ite['name']} trains students in software, computer organization and mathematics, "
         f"with elective courses such as Artificial Intelligence, Computer Graphics, Game Development, "
         f"Data Mining, and Security.",
         f"In the {ite['name']}, students are required to do practicum from year 2 to year 4 with real world problems.",
         f"In the {ite['name']}, students must go for an internship in private or public organizations "
         f"from the end of the third year.",
         "\n## ITE careers and research\n",
         f"Graduates of the {ite['name']} can work as: {', '.join(ite['prospects']['job_titles'])}.",
         f"ITE graduates can get jobs as {', '.join(ite['prospects']['job_titles'])}.",
         f"The research areas of the IT Engineering department are: "
         f"{', '.join(ite['prospects']['research_areas'])}.",
         "\n## ITE grading, GPA and graduation\n",
         ite["evaluation_and_graduation"]["assessment"]]
    for g in ite["evaluation_and_graduation"]["grading_scale"]:
        L.append(f"In the {ite['name']}, a score of {g['score_range']} earns grade {g['grade']} "
                 f"({g['meaning']}), worth {g['gpa']:.1f} grade points.")
    t1, t2 = ite["final_semester_tracks"]["Option 1"], ite["final_semester_tracks"]["Option 2"]
    tot = ite["computed_total_credits"]
    L += ["\n## ITE final semester options: internship, thesis or coursework\n",
          f"In the final semester (fourth year, Semester II) of the {ite['name']}, students choose "
          f"between two options.",
          f"Option 1 of the ITE final semester is coursework plus a Project Report, worth "
          f"{_n(t1['credits'])} credits.",
          f"Option 2 of the ITE final semester is an Internship (Hard Skill) and Thesis, worth "
          f"{_n(t2['credits'])} credits, and it requires a GPA above 3.5.",
          f"To do the internship and thesis option in the ITE degree, a student needs a GPA above 3.5.",
          f"The {ite['name']} totals {_n(tot['with_option_1'])} credits with Option 1 or "
          f"{_n(tot['with_option_2'])} credits with Option 2 (computed from the curriculum).",
          f"The ITE degree requires {_n(tot['with_option_1'])} credits in total with Option 1, "
          f"or {_n(tot['with_option_2'])} credits with Option 2.",
          f"ITE Option 1 is worth {_n(t1['credits'])} credits and ITE Option 2 is worth "
          f"{_n(t2['credits'])} credits.",
          f"The ITE internship and thesis (Option 2) is worth {_n(t2['credits'])} credits.",
          f"The ITE Option 1 courses are "
          f"{_courses(ite['curriculum']['Fourth Year']['Semester II']['Option 1'])}."]
    L += _curriculum("ITE", ite["name"], ite["curriculum"])
    L += _course_index("ITE", ite["curriculum"])
    L += ["\n## AIE artificial intelligence engineering degree overview\n",
          f"The {aie['name']} is offered by the Department of Information Technology Engineering.",
          f"{aie['curriculum_statement']}",
          f"The {aie['name']} totals {_n(aie['computed_total_credits']['all_listed_courses'])} credits "
          f"across all listed courses (computed from the curriculum).",
          f"The AIE degree requires {_n(aie['computed_total_credits']['all_listed_courses'])} credits in total.",
          f"In the final semester of the {aie['name']}, the curriculum lists Report or Thesis (2 credits) "
          f"and Internship (2 credits); the website does not say whether students choose one or do both."]
    L += _curriculum("AIE", aie["name"], aie["curriculum"])
    L += _course_index("AIE", aie["curriculum"])
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default="rupp_ite_programs.json")
    ap.add_argument("--out", default="data/corpus/rupp_ite_facts.md")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    Path(args.out).write_text(build(data), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
