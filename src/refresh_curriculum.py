"""Refresh the local curriculum snapshot from official RUPP HTML."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from html import unescape
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from retrieve import PROJECT_DIR

BASE = "https://www.rupp.edu.kh/academics/faculties_and_institutes/faculty_of_engineering/department_of_information_technology_engineering"
DEPARTMENT = "Department of Information Technology Engineering"
SNAPSHOT = PROJECT_DIR / "data" / "official" / "curricula.json"
PROGRAMS = {
    "ITE": (
        "Information Technology Engineering",
        BASE + "/information_technology_engineering_bachelor_degree",
        "The program builds foundations in computing, software and networks, with applied practicum and internship experience.",
    ),
    "DSE": (
        "Data Science and Engineering",
        BASE + "/data_science_and_engineering_bachelor_degree",
        "The published curriculum covers statistics, data engineering, analytics and machine learning, with a professional internship and final project.",
    ),
    "AIE": (
        "Artificial Intelligence Engineering",
        BASE + "/aie",
        "The published course tables focus on building AI and machine learning systems, including courses in computer vision, language processing and AI operations.",
    ),
}
REVIEW_NOTES = {
    "DSE": "The introduction includes Environmental Engineering text. The description here comes from the DSE course tables.",
    "AIE": "The Prospect introduction refers to Data Science & Engineering, and the Introduction tab contains placeholder text. The description here comes from the AIE course tables.",
}
YEAR_NAMES = ("Foundation", "Second Year", "Third Year", "Fourth Year")
NUMBER = r"\d+(?:\.\d+)?"


def plain(html):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", html)).split())


def table_rows(table):
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.I | re.S):
        yield [plain(cell) for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.I | re.S)]


def parse_hours(raw, credits):
    if raw in ("", "-", "–", "—"):
        return None, []
    match = re.fullmatch(
        rf"({NUMBER})?\s*\(({NUMBER})\s*-\s*({NUMBER})(?:\s*-\s*({NUMBER}))?\)", raw
    )
    if not match:
        return None, ["Unrecognized hours format."]
    prefix, lecture, practice, study = match.groups()
    if prefix and study is None:
        return None, ["Prefixed hours omit a third component. Verify the original notation."]
    notes = []
    if prefix and Decimal(prefix) != Decimal(credits):
        notes.append("The prefix in the hours cell differs from the Credit column.")
    return {"lecture": lecture, "practice": practice, "self_study": study}, notes


def parse_curriculum(html, program_id, retrieved_at):
    name, url, description = PROGRAMS[program_id]
    if name.casefold() not in plain(html).casefold():
        raise ValueError(f"{program_id}: program name missing from the page.")
    courses = []
    years = []
    previous_table_end = 0
    for table in re.finditer(r"<table\b[^>]*>.*?</table>", html, re.I | re.S):
        rows = list(table_rows(table.group()))
        heading = plain(html[previous_table_end:table.start()])
        previous_table_end = table.end()
        if not rows or [cell.casefold() for cell in rows[0][:2]] != ["subject", "credit"]:
            continue
        year = next((i for i, label in enumerate(YEAR_NAMES, 1) if heading.endswith(label)), None)
        if year is None and heading.endswith("Forth Year"):
            year = 4
        if year is None or year in years:
            raise ValueError(f"{program_id}: unexpected or duplicate year heading.")
        years.append(year)
        semesters = set()
        semester = None
        option = None
        for cells in rows[1:]:
            if not cells or not any(cells):
                continue
            label = cells[0]
            semester_number = {"SEMESTER I": 1, "SEMESTER II": 2}.get(label.upper())
            if semester_number:
                semester = semester_number
                semesters.add(semester)
                option = None
                continue
            if re.fullmatch(r"Option \d+", label, re.I) and not any(cells[1:]):
                option = label
                continue
            if semester is None or len(cells) != 3 or not label or not re.fullmatch(NUMBER, cells[1]):
                raise ValueError(f"{program_id}: unexpected curriculum row {cells!r}")
            course, credits, raw_hours = cells
            hours, notes = parse_hours(raw_hours, credits)
            courses.append({
                "program_id": program_id, "program_name": name, "department": DEPARTMENT,
                "academic_year": year, "semester": semester, "course_name": course,
                "credits": credits, "course_hours": hours, "source_hours": raw_hours,
                "option": option, "source_url": url, "retrieved_at": retrieved_at,
                "verification_status": "review_needed" if notes else "verified",
                "review_notes": notes,
            })
        if semesters != {1, 2} or {row["semester"] for row in courses if row["academic_year"] == year} != {1, 2}:
            raise ValueError(f"{program_id}: incomplete Year {year} curriculum.")
    if years != [1, 2, 3, 4]:
        raise ValueError(f"{program_id}: expected four year tables, got {years}.")
    return {
        "program_id": program_id, "program_name": name, "department": DEPARTMENT,
        "description": description, "source_url": url, "retrieved_at": retrieved_at,
        "source_review_note": REVIEW_NOTES.get(program_id), "courses": courses,
    }


def fetch(url):
    request = Request(url, headers={"User-Agent": "RUPP Student Assistant curriculum refresh"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-dir", type=Path, help="Use saved department.html, ite.html, dse.html and aie.html")
    parser.add_argument("--output", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    if args.html_dir:
        retrieved_at = datetime.fromtimestamp(
            min((args.html_dir / f"{key}.html").stat().st_mtime for key in ("department", "ite", "dse", "aie")),
            timezone.utc,
        ).isoformat(timespec="seconds")
    else:
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def source(key, url):
        return (args.html_dir / f"{key}.html").read_text(encoding="utf-8-sig") if args.html_dir else fetch(url)

    try:
        department_html = source("department", BASE)
        department_text = plain(department_html)
        if DEPARTMENT not in department_text or not all(url.rsplit("/", 1)[1] in department_html for _, url, _ in PROGRAMS.values()):
            raise ValueError("Department page no longer matches the verified program listing.")
        contact = re.search(r"Contact Details\s+Address\s+(.+?)\s+Phone\s+(.+?)\s+Email\s+([\w.+-]+@rupp\.edu\.kh)", department_text, re.I)
        university = re.search(r"Contact Us\s+Russian Federation Blvd, Phnom Penh, Cambodia\s+(info@rupp\.edu\.kh)\s+(\+\d[\d ]+)", department_text, re.I)
        if not contact or not university:
            raise ValueError("Could not verify the department or university contact block.")
        data = {
            "department": DEPARTMENT, "source_url": BASE, "retrieved_at": retrieved_at,
            "contact": dict(zip(("address", "phone", "email"), contact.groups())),
            "university_contact": {
                "address": "Russian Federation Blvd, Phnom Penh, Cambodia",
                "email": university[1], "phone": university[2].strip(),
            },
            "programs": [parse_curriculum(source(key.lower(), details[1]), key, retrieved_at) for key, details in PROGRAMS.items()],
        }
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.close()
                temporary.replace(args.output)
            finally:
                handle.close()
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(f"Refresh stopped. The previous snapshot was preserved. {exc}")

    for program in data["programs"]:
        flagged = sum(row["verification_status"] == "review_needed" for row in program["courses"])
        print(f"{program['program_id']}: {len(program['courses'])} course records ({flagged} flagged rows)")


if __name__ == "__main__":
    main()
