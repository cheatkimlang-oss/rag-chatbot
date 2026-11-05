"""Find a source passage by keyword overlap."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[1]
COLLECTIONS = {
    "practice": PROJECT_DIR / "index" / "passages.jsonl",
    "official": PROJECT_DIR / "index" / "official.jsonl",
}
DEFAULT_INDEX = COLLECTIONS["practice"]
STOP_WORDS = set(
    "a an are as at be by can do does for from how i in is it many me "
    "of on or should that the their they this to was we what when where "
    "which who why will with would you your".split()
)


def keywords(text):
    return set(re.findall(r"[^\W_]+", text.casefold())) - STOP_WORDS


def load_passages(path):
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as exc:
        raise ValueError(f"Index not found: {path}. Run ingestion first.") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read the index: {exc}") from exc


def best_passage(question, rows):
    query = keywords(question)
    passage_keywords = [keywords(row["text"]) for row in rows]
    frequency = Counter(word for words in passage_keywords for word in words)

    ranked = []
    for row, words in zip(rows, passage_keywords):
        matches = query & words
        if len(matches) >= min(2, len(query)) and matches:
            score = sum(1 / frequency[word] for word in sorted(matches))
            ranked.append((score, row))

    return max(ranked, key=lambda result: result[0])[1] if ranked else None


def source_location(passage):
    if passage.get("url"):
        return f"Source: {passage['source']} | {passage['url']}"
    return f"Source: {passage['source']} | PDF page {passage['page']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    indexes = parser.add_mutually_exclusive_group()
    indexes.add_argument("--index", type=Path, help="Use a custom index file")
    indexes.add_argument("--collection", choices=COLLECTIONS, default="practice")
    args = parser.parse_args()
    try:
        best = best_passage(args.question, load_passages(args.index or COLLECTIONS[args.collection]))
    except ValueError as exc:
        parser.error(str(exc))

    if best is None:
        print("No matching passage found in the indexed documents.")
        return

    print(source_location(best))
    print(best["text"])


if __name__ == "__main__":
    main()
