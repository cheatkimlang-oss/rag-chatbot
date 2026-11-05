import pytest

from retrieval.entity import resolve_senior
from retrieval.intent import normalize


@pytest.mark.parametrize("name", ["Sok Dara", "សុខ ដារា", "sokdaraa", "DARA Sok", "sok  dára"])
def test_three_spellings_one_senior(kb, name):
    r = resolve_senior(normalize(name), kb.db, kb.client)
    assert r["status"] == "resolved"
    assert r["senior_id"] == "s-2024-014"


def test_duplicate_names_are_never_merged(kb):
    kb.db.upsert_senior(dict(senior_id="s-2023-077", full_name_latin="Sok Dara",
                             name_normalized="sok dara", program="ITE", cohort_year=2023))
    r = resolve_senior("sok dara", kb.db, kb.client)
    assert r["status"] == "ambiguous"
    got = {(c["senior_id"], c["program"], c["cohort_year"]) for c in r["candidates"]}
    assert got == {("s-2024-014", "DSE", 2024), ("s-2023-077", "ITE", 2023)}


def test_unknown_name_returns_unconfirmed_closest(kb):
    r = resolve_senior("sok panha", kb.db, kb.client)
    assert r["status"] == "not_found"
    assert 0 < len(r["closest"]) <= 3
    assert all(c["confirmed"] is False for c in r["closest"])


def test_withdrawn_senior_is_invisible(kb):
    r = resolve_senior("meas bopha", kb.db, kb.client)
    assert r["status"] == "not_found"
    assert "s-2023-099" not in {c["senior_id"] for c in r["closest"]}


def test_text_index_does_not_resolve_to_coauthor(kb):
    # "sreyneang" matches the dse2024-07 author string, which also carries Sok Dara's id.
    r = resolve_senior("sreyneang", kb.db, kb.client)
    assert r["status"] == "resolved" and r["senior_id"] == "s-2024-021"


def test_works_without_qdrant(kb):
    r = resolve_senior("sokdaraa", kb.db, client=None)
    assert r["status"] == "resolved" and r["via"] == "fuzzy"
