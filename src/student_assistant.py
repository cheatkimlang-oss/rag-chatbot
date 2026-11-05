"""Print text from a PDF page, using page numbers that start at 1."""
import argparse
from pathlib import Path

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF")
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    reader = PdfReader(args.pdf)
    if not 1 <= args.page <= len(reader.pages):
        parser.error(f"Choose a page from 1 to {len(reader.pages)}.")

    page_text = (reader.pages[args.page - 1].extract_text() or "").strip()
    print(f"{args.pdf.name} | PDF page {args.page}")
    print(page_text or "No selectable text found. This page may need OCR.")


if __name__ == "__main__":
    main()
