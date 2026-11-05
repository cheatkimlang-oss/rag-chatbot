"""Resolve short student follow-ups against the current conversation."""
import re

from curriculum import ENROLLMENT, mentioned_programs, program_summary, source


OTHER_MAJORS = re.compile(
    r"(?:what (?:is |are )?|are there (?:any )?|any |show me |list )?"
    r"(?:another|other) (?:major|program|degree)s?(?: are available)?"
)


def resolve_followup(question, snapshot, state):
    """Return a short reply for recognized follow-ups, or let normal search handle them."""
    q = " ".join(question.casefold().strip(" ?!.").split())
    programs = snapshot["programs"]
    current = [program for program in programs if program["program_id"] in state.get("programs", ())]
    named = mentioned_programs(q, programs)

    if OTHER_MAJORS.fullmatch(q):
        previous = current[0] if len(current) == 1 else None
        choices = [program for program in programs if program != previous]
        state["programs"] = tuple(program["program_id"] for program in choices)
        heading = f"Besides {previous['program_id']}, you can also study:" if previous else "The ITE Department lists these majors:"
        options = "\n".join(f"- {program['program_name']} ({program['program_id']})" for program in choices)
        return f"{heading}\n{options}\nWhich one would you like to know about?\n{source(snapshot['source_url'])}"

    if named:
        state["programs"] = tuple(program["program_id"] for program in named)
        if len(named) == 1:
            program = named[0]
            labels = {program["program_id"].casefold(), program["program_name"].casefold()}
            if q in labels or (q.startswith("what ") and any(q.endswith(f"know about {label}") for label in labels)):
                return program_summary(program)
        return None

    about_current = re.search(r"\b(?:about (?:it|this|that)|(?:this|that) (?:major|program|degree))\b", q)
    asks_overview = re.search(r"\b(?:know|tell|explain|describe|what)\b", q)
    if about_current and asks_overview and not ENROLLMENT.search(q):
        if len(current) == 1:
            return program_summary(current[0])
        options = " or ".join(program["program_id"] for program in (current or programs))
        return (
            f"Which major do you mean: {options}? "
            "I can explain what you study, how long the program lasts, and what the curriculum says about internships."
        )

    # A new, unrelated question should not inherit the previous major.
    state.pop("programs", None)
    return None
