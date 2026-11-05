import pytest

from retrieval import intent as I

KNOWN = {"sok dara", "dara sok", "សុខ ដារា", "chan sreyneang", "kim lang"}
SURNAMES = {"sok", "dara", "chan", "sreyneang", "kim", "lang"}


def test_normalize_latin_and_whitespace():
    assert I.normalize("  Sók   DÁRA ") == "sok dara"


def test_normalize_preserves_khmer_marks():
    # ុ (U+17BB) and ា (U+17B6) are combining marks; stripping them would break the name.
    assert I.normalize("សុខ ដារា") == "សុខ ដារា"
    assert I.normalize("សុខ​ដារា") == "សុខ ដារា"


@pytest.mark.parametrize("query,intent,name,topic", [
    ("khmer ocr", I.TOPIC_SEARCH, None, None),
    ("Sok Dara", I.SENIOR_LOOKUP, "sok dara", None),
    ("DARA Sok", I.SENIOR_LOOKUP, "dara sok", None),
    ("សុខ ដារា", I.SENIOR_LOOKUP, "សុខ ដារា", None),
    ("Sok Dara's OCR project", I.SENIOR_TOPIC, "sok dara", "ocr"),
    ("projects by Sok Dara", I.PORTFOLIO, "sok dara", None),
    ("what did Sok Dara build?", I.SENIOR_LOOKUP, "sok dara", None),
    ("who built a khmer ocr system?", I.ANSWER, None, None),
    ("2024", I.FACET_BROWSE, None, None),
    ("DSE projects 2024", I.FACET_BROWSE, None, None),
    ("crop disease detection 2023", I.TOPIC_SEARCH, None, None),
])
def test_route(query, intent, name, topic):
    r = I.route(query, KNOWN, SURNAMES)
    assert (r["intent"], r["name"], r["topic"]) == (intent, name, topic)


def test_route_extracts_facets():
    r = I.route("DSE projects 2024", KNOWN, SURNAMES)
    assert r["years"] == [2024] and r["programs"] == ["DSE"]


def test_surname_guess_is_flagged_as_heuristic():
    r = I.route("Chan Visal", KNOWN, SURNAMES)
    assert r["intent"] == I.SENIOR_LOOKUP and r["name_heuristic"] is True
