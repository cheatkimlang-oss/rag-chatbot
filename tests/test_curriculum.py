"""Offline regression checks for the installed official and practice collections."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from answer import answer_question
from curriculum import UNVERIFIED, academic_answer, load_curriculum
from retrieve import COLLECTIONS, best_passage, load_passages


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_curriculum()
        cls.programs = {program["program_id"]: program for program in cls.snapshot["programs"]}

    def check(self, question, expected, *programs):
        result = academic_answer(question, self.snapshot)
        self.assertIsNotNone(result)
        self.assertIn(expected, result)
        for program in programs:
            self.assertIn(self.programs[program]["source_url"], result)
        return result

    def test_department_majors(self):
        result = self.check("What undergraduate majors does the ITE Department offer?", self.snapshot["source_url"])
        for program in self.programs.values():
            self.assertIn(program["program_name"], result)

    def test_aie_description(self):
        self.check("What is Artificial Intelligence Engineering?", "building AI and machine learning systems", "AIE")

    def test_aie_third_year_second_semester(self):
        result = self.check("What subjects do AIE students study in Year 3, Semester 2?", "AIE Year 3, Semester 2", "AIE")
        self.assertEqual(result.count("\n- "), 7)
        self.assertIn("Computer Vision: 2 credits", result)
        self.assertNotIn("DSE Year", result)

    def test_aie_deep_learning_uses_credit_column(self):
        result = self.check("How many credits is Deep Learning in AIE?", "Deep Learning: 2 credits", "AIE")
        self.assertIn("3(2-1-0)", result)
        self.assertIn("Review:", result)
        self.assertNotIn("Deep Learning and Application", result)

    def test_natural_language_processing(self):
        self.check("Does AIE include Natural Language Processing?", "Natural Language Processing: 2 credits", "AIE")

    def test_internship(self):
        result = self.check("Does AIE include an internship?", "Internship: 2 credits", "AIE")
        self.assertIn("AIE Year 4, Semester 2", result)
        self.assertNotIn("hours as published", result)

    def test_comparison(self):
        result = self.check("What is the difference between ITE and AIE?", "Information Technology Engineering (ITE)", "ITE", "AIE")
        self.assertIn("Artificial Intelligence Engineering (AIE)", result)
        self.assertNotIn(self.programs["DSE"]["source_url"], result)

    def test_tuition(self):
        self.assertEqual(academic_answer("What is the tuition fee for AIE?", self.snapshot), UNVERIFIED)

    def test_deadline(self):
        self.assertEqual(academic_answer("When is the admission deadline?", self.snapshot), UNVERIFIED)

    def test_contact_email(self):
        result = self.check("What is the contact email for the department?", "fe.info@rupp.edu.kh")
        self.assertIn(self.snapshot["source_url"], result)

    def test_existing_dse_course_lookup(self):
        self.check("When do DSE students take Data Analytics and Machine Learning?", "DSE Year 3, Semester 1", "DSE")

    def test_ite_year_two_and_khmer_year_semester(self):
        self.check("What courses are available in ITE Year 2?", "Data Communication II", "ITE")
        self.check("មុខវិជ្ជា AIE ឆ្នាំទី៣ ឆមាសទី២?", "Deep Learning: 2 credits", "AIE")

    def test_no_cross_degree_match(self):
        self.assertEqual(academic_answer("Does ITE include Large Language Models?", self.snapshot), UNVERIFIED)

    def test_cli_path_keeps_exact_answers_out_of_gemma(self):
        rows = load_passages(COLLECTIONS["official"])
        output = StringIO()
        with patch("answer.generate_answer", side_effect=AssertionError("Exact lookup must bypass Gemma")), redirect_stdout(output):
            answer_question("How many credits is Deep Learning in AIE?", rows, "gemma4:e2b", self.snapshot, official=True)
        self.assertIn("Deep Learning: 2 credits", output.getvalue())

    def test_existing_practice_pdf(self):
        rows = load_passages(COLLECTIONS["practice"])
        result = best_passage("Why does the cafeteria use simulation to decide how many meals to prepare?", rows)
        self.assertEqual(result["page"], 1)
        self.assertIn("Simulation and Decision Analytics.pdf", result["source"])


if __name__ == "__main__":
    unittest.main()
