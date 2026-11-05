"""Answer exact academic questions from the local RUPP curriculum snapshot."""
from collections import defaultdict
import json
import re

from refresh_curriculum import SNAPSHOT

UNVERIFIED = (
    "I couldn't verify that information from the available official RUPP sources. "
    "Please contact the university or department for confirmation."
)
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
ALIASES = {
    "ITE": r"\bite\b|\binformation technology engineering\b|វិស្វកម្មបច្ចេកវិទ្យាព័ត៌មាន",
    "DSE": r"\bdse\b|\bdata science\s*(?:and|&)\s*engineering\b|វិទ្យាសាស្ត្រទិន្នន័យ",
    "AIE": r"\baie\b|\bartificial intelligence engineering\b|វិស្វកម្មបញ្ញាសិប្បនិម្មិត",
}
ENROLLMENT = re.compile(
    r"\b(?:tuition|fees?|deadline|scholarships?|admission|admissions|enrol\w*|"
    r"register|registration|eligibility|eligible|apply|applications?)\b|ថ្លៃសិក្សា|អាហារូបករណ៍|ចុះឈ្មោះ|ថ្ងៃផុតកំណត់"
)


def load_curriculum():
    if not SNAPSHOT.exists():
        return None
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def mentioned_programs(question, programs):
    # The department name can appear alongside a different degree.
    text = re.sub(
        r"\b(?:department of information technology engineering|ite department)\b",
        "", question.casefold(),
    )
    return [program for program in programs if re.search(ALIASES[program["program_id"]], text)]


def source(url):
    return f"Source: {url}"


def academic_answer(question, snapshot):
    q = question.casefold().translate(KHMER_DIGITS)
    if ENROLLMENT.search(q):
        return UNVERIFIED
    if snapshot is None:
        return None

    programs = snapshot["programs"]
    named = mentioned_programs(q, programs)
    directory = source(snapshot["source_url"])

    if q.strip(" ?.!") in {"time for study", "study time", "what is the study time"}:
        return (
            "Do you mean how many years the degree takes, when classes meet, "
            "or the weekly hours for a course? Please include the program (ITE, DSE, or AIE)."
        )
    if re.search(r"\b(?:how many years|how long.*(?:take|last|study)|(?:program|degree|study) duration)\b", q):
        if not named:
            return "Which program should I look up: ITE, DSE, or AIE?"
        return "\n".join(
            f"The published {program['program_id']} curriculum covers "
            f"{len({row['academic_year'] for row in program['courses']})} academic years, including Foundation.\n"
            + source(program["source_url"])
            for program in named
        )
    if re.search(r"\b(?:compare|difference|differences|versus|vs)\b|ប្រៀបធៀប|ខុសគ្នា", q):
        if len(named) < 2 and not re.search(r"\b(?:programs|majors|degrees)\b|ជំនាញ", q):
            return None
        return "\n".join(program_summary(program) for program in (named or programs))

    if re.search(r"\b(?:contact|e-?mail|phone|address|where.*office)\b|ទំនាក់ទំនង|អ៊ីមែល", q):
        university = bool(re.search(r"\b(?:university|rupp)\b", q)) and not (
            named or "department" in q or "ដេប៉ាតឺម៉ង់" in q
        )
        contact = snapshot["university_contact" if university else "contact"]
        label = "RUPP" if university else snapshot["department"]
        return f"{label}\nEmail: {contact['email']}\nPhone: {contact['phone']}\nAddress: {contact['address']}\n{directory}"

    if not named and re.search(r"\b(?:majors?|programs|degrees)\b|ជំនាញ", q) and re.search(
        r"\b(?:offer\w*|available|undergraduate|bachelor|list)\b|មាន", q
    ):
        return "Bachelor's programs listed by the department:\n" + "\n".join(
            f"- {program['program_name']} ({program['program_id']})" for program in programs
        ) + f"\n{directory}"

    year_match = re.search(r"\b(?:year\s*([1-4])|([1-4])(?:st|nd|rd|th)?\s*year)\b|ឆ្នាំ\s*(?:ទី\s*)?([1-4])", q)
    year = next((int(value) for value in year_match.groups() if value), None) if year_match else None
    if year is None:
        terms = (("foundation", "first year"), ("second year",), ("third year",), ("fourth year",))
        year = next((i for i, names in enumerate(terms, 1) if any(name in q for name in names)), None)
    semester_match = re.search(r"\b(?:semester|sem)\s*(ii|i|[12])\b|ឆមាស\s*(?:ទី\s*)?([12])", q)
    semester = {"i": 1, "ii": 2, "1": 1, "2": 2}[
        next(value for value in semester_match.groups() if value)
    ] if semester_match else None

    candidates = [
        row for program in (named or programs) for row in program["courses"]
        if (year is None or row["academic_year"] == year)
        and (semester is None or row["semester"] == semester)
    ]
    exact_courses = [row for row in candidates if contains_title(q, row["course_name"])]
    if exact_courses:
        longest = max(len(row["course_name"]) for row in exact_courses)
        exact_courses = [row for row in exact_courses if len(row["course_name"]) == longest]
    course_question = re.search(
        r"\b(?:credits?|hours?|include[sd]?|offer[sd]?|teach(?:es)?|internship|"
        r"which (?:year|semester)|what (?:year|semester))\b|ក្រេឌីត|ម៉ោង|កម្មសិក្សា", q
    )
    matches = exact_courses
    if course_question and not matches:
        q_course = q
        for alias in ALIASES.values():
            q_course = re.sub(alias, " ", q_course)
        q_course = re.sub(r"\b(?:year|semester|sem)\s*(?:[1-4]|ii|i)\b", " ", q_course)
        match = re.search(r"\b(?:credits?|hours?)\s+(?:is|are|for|does|do)\s+(.+)|"
                          r"\b(?:include|offer|teach|study)\s+(?:an?\s+)?(.+)", q_course)
        phrase = next((part for part in match.groups() if part), "") if match else ""
        phrase = re.sub(r"\s+(?:have|take|worth|carry|in|for)\s*$", "", phrase.strip(" ?.")).strip()
        if not phrase and ("internship" in q or "កម្មសិក្សា" in q):
            phrase = "internship"
        if phrase:
            matches = [row for row in candidates if contains_title(row["course_name"].casefold(), phrase)]

    if matches:
        return format_courses(matches)
    if course_question and re.search(r"\b(?:credits?|hours?|include[sd]?|internship)\b|ក្រេឌីត|ម៉ោង|កម្មសិក្សា", q):
        return UNVERIFIED
    if (year or semester or re.search(r"\b(?:curriculum|subjects|courses)\b|មុខវិជ្ជា|កម្មវិធីសិក្សា", q)) and (
        named or year or semester
    ):
        if len(named) != 1:
            return "Which program should I look up: ITE, DSE, or AIE?"
        return format_courses(candidates) if candidates else UNVERIFIED
    if len(named) == 1 and re.search(r"\b(?:what is|tell me about|describe|overview|description)\b|ជាអ្វី", q):
        return program_summary(named[0])
    return None


def contains_title(text, title):
    normalized = " ".join(re.findall(r"\w+", title.casefold()))
    query = " ".join(re.findall(r"\w+", text))
    return bool(normalized and f" {normalized} " in f" {query} ")


def program_summary(program):
    return (
        f"{program['program_name']} ({program['program_id']})\n"
        f"{program['description']}\n{source(program['source_url'])}"
    )


def format_courses(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["program_id"], row["academic_year"], row["semester"])].append(row)
    lines = []
    for (program, year, semester), courses in groups.items():
        lines.append(f"{program} Year {year}, Semester {semester}")
        for row in courses:
            hours = f" | hours as published: {row['source_hours']}" if row["source_hours"] else ""
            option = f" | {row['option']}" if row["option"] else ""
            review = " | Review: " + " ".join(row["review_notes"]) if row["review_notes"] else ""
            lines.append(f"- {row['course_name']}: {row['credits']} credits{option}{hours}{review}")
    lines.extend(source(url) for url in dict.fromkeys(row["source_url"] for row in rows))
    return "\n".join(lines)
