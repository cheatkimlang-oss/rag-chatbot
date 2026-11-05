"""Compare keyword and hybrid source retrieval on labeled student questions."""
import json

from hybrid import load_hybrid
from retrieve import COLLECTIONS, PROJECT_DIR, best_passage, load_passages


def main():
    rows = load_passages(COLLECTIONS["official"])
    search = load_hybrid(rows, required=True)
    cases = json.loads((PROJECT_DIR / "evaluation" / "retrieval.json").read_text(encoding="utf-8"))
    totals = {"keyword": [0, 0], "hybrid": [0, 0]}
    positives = sum(case["url"] is not None for case in cases)
    negatives = len(cases) - positives

    for case in cases:
        question, expected = case["question"], case["url"]
        ranked = search.rank(question)
        matches = {"keyword": best_passage(question, rows), "hybrid": ranked[0] if ranked else None}
        results = []
        for method, match in matches.items():
            actual = match.get("url") if match else None
            passed = actual == expected if expected else match is None
            totals[method][0 if expected else 1] += passed
            results.append(f"{method}: {'pass' if passed else 'miss'}")
        print(f"{question}  |  {'; '.join(results)}")

    print("\nTop source accuracy on supported questions / abstention on unsupported questions:")
    for method, (correct, abstained) in totals.items():
        print(f"{method}: {correct}/{positives} correct sources; {abstained}/{negatives} abstentions")
    print("Small local evaluation. These results do not estimate accuracy on all student questions.")


if __name__ == "__main__":
    main()
