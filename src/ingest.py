"""Index PDFs and curated web notes with their source information."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from retrieve import COLLECTIONS, DEFAULT_INDEX, PROJECT_DIR


def passages(pdf):
    for page_number, page in enumerate(PdfReader(pdf).pages, start=1):
        words = (page.extract_text() or "").split()
        for start in range(0, len(words), 100):
            yield {
                "source": pdf.name,
                "page": page_number,
                "text": " ".join(words[start:start + 120]),
            }
            if start + 120 >= len(words):
                break


def is_source(file):
    return file.is_file() and (
        file.suffix.lower() == ".pdf" or file.name.lower().endswith(".web.json")
    )


def source_passages(file):
    if file.suffix.lower() == ".pdf":
        return list(passages(file))
    note = json.loads(file.read_text(encoding="utf-8-sig"))
    title, url, sections = note["title"], note["url"], note["passages"]
    if (
        not isinstance(title, str) or not title.strip()
        or not isinstance(url, str) or not url.startswith("https://")
        or not isinstance(sections, list) or not sections
        or any(not isinstance(section, str) or not section.strip() for section in sections)
    ):
        raise ValueError("Expected a title, HTTPS source URL, and non-empty passages.")
    return [
        {"source": title, "url": url, "page": None, "text": section}
        for section in sections
    ]


def write_index(output, records):
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            for record in records:
                temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.close()
            temporary_path.replace(output)
        finally:
            temporary.close()
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("source", type=Path, nargs="?", help="PDF, .web.json source, or folder")
    inputs.add_argument("--collection", choices=COLLECTIONS, help="Use data/<collection>")
    parser.add_argument("--output", type=Path, help="Custom index file")
    args = parser.parse_args()

    if args.collection:
        if args.output:
            parser.error("--collection already selects the output index.")
        source = PROJECT_DIR / "data" / args.collection
        output = COLLECTIONS[args.collection]
    else:
        source = args.source
        output = args.output or DEFAULT_INDEX
        if source.is_dir() and args.output is None:
            parser.error("Use --output for a custom folder, or use --collection.")

    if source.is_dir():
        pdfs = sorted(
            (file for file in source.iterdir() if is_source(file)),
            key=lambda file: file.name.casefold(),
        )
    else:
        pdfs = [source] if is_source(source) else []
    if not pdfs:
        parser.error(f"No sources found at {source}. Add a PDF or .web.json note and run ingestion again.")
    if output.suffix.lower() != ".jsonl":
        parser.error("The output path must end in .jsonl.")

    records = []
    for pdf in pdfs:
        try:
            extracted = source_passages(pdf)
        except (OSError, PdfReadError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Could not read {pdf.name}: {exc}")
        if not extracted:
            parser.error(f"No selectable text in {pdf.name}. This PDF may need OCR.")
        records.extend(extracted)

    try:
        write_index(output, records)
    except OSError as exc:
        parser.error(f"Could not write the index: {exc}")
    print(f"Saved {len(records)} passages from {len(pdfs)} source file(s) to {output}")


if __name__ == "__main__":
    main()
