from answer.extract import _blocks, extract_answer, units_of
from tests.conftest import FakeEmbedder

PROFILE = """[Project: About Me | Source: about_me.pdf | Section: ABOUT ME | p.1]
Introduction
I am a Data Science and Engineering student at the Royal University of Phnom Penh (RUPP). I am interested in
technology, programming, artificial intelligence, and using data to solve real-world problems.
Goals and Interests
•
Improve my English and prepare for the IELTS exam.
•
Explore international study opportunities, especially in Japan.
Toolkit
Programming Languages | Python, R, SQL
Data Frameworks | Pandas, NumPy, PyTorch"""


def _card(text=PROFILE, heading="ABOUT ME"):
    crumb = text.splitlines()[0]
    return {"project_id": "doc-about-me", "title": "About Me", "source": "about_me.pdf",
            "evidence": [{"text": text, "breadcrumb": crumb, "page_number": 1,
                          "page_label": "p.1", "heading": heading}]}


class KeywordScorer:
    """Stand-in cross-encoder: counts question words in the scored text."""
    kind = "cross"

    def score(self, question, texts):
        q = set(question.lower().replace("?", "").split())
        return [float(len(q & set(t.lower().replace(":", " ").split()))) for t in texts]


def test_wrapped_pdf_lines_rejoin_and_lone_bullets_attach():
    kinds = [(k, t[:30]) for k, t in _blocks(PROFILE.split("\n", 1)[1])]
    assert kinds[0] == ("heading", "Introduction")
    assert kinds[1][0] == "sentence" and "technology, programming" in _blocks(PROFILE)[2][1]
    assert ("bullet", "Improve my English and prepare") in kinds
    assert kinds[-1][0] == "table"


def test_units_carry_their_heading_as_context():
    units = units_of(_card(), _card()["evidence"][0])
    bullet = next(u for u in units if u.text.startswith("Improve"))
    assert bullet.context == "ABOUT ME — Goals and Interests"      # page title + sub-heading
    assert bullet.scored_text.startswith("ABOUT ME — Goals and Interests: Improve")


def test_picks_the_sentence_not_the_chunk():
    ex = extract_answer("which university do I study at?", [_card()], FakeEmbedder(), KeywordScorer())
    assert ex["text"] == ("I am a Data Science and Engineering student at the "
                          "Royal University of Phnom Penh (RUPP).")
    assert ex["page_label"] == "p.1" and ex["card"]["source"] == "about_me.pdf"


def test_question_naming_the_list_gets_the_whole_list():
    ex = extract_answer("what are my goals?", [_card()], FakeEmbedder(), KeywordScorer())
    assert ex["text"] == ("Goals and Interests:\n• Improve my English and prepare for the IELTS exam.\n"
                          "• Explore international study opportunities, especially in Japan.")


def test_specific_question_gets_only_the_matching_bullet():
    class Scores(KeywordScorer):
        def score(self, question, texts):
            return [5.0 if "IELTS" in t else 0.0 for t in texts]

    ex = extract_answer("what exam am i preparing for?", [_card()], FakeEmbedder(), Scores())
    assert ex["text"] == "Goals and Interests: Improve my English and prepare for the IELTS exam."


def test_table_is_one_unit():
    ex = extract_answer("which programming languages are in the toolkit?", [_card()],
                        FakeEmbedder(), KeywordScorer())
    assert ex["text"] == "Programming Languages | Python, R, SQL\nData Frameworks | Pandas, NumPy, PyTorch"


def test_falls_back_to_dense_scoring_without_a_scorer():
    ex = extract_answer("royal university phnom penh", [_card()], FakeEmbedder(), scorer=None)
    assert "Royal University of Phnom Penh" in ex["text"] and ex["scored_by"] == "dense"


def test_no_answerable_text_returns_none():
    card = _card("[Project: X | p.1]\nTitle Only", heading=None)
    assert extract_answer("anything?", [card], FakeEmbedder()) is None
