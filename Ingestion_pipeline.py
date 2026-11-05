"""Ingestion pipeline entry point (Plane A).

    .\\.venv\\Scripts\\python.exe Ingestion_pipeline.py            index new/changed files
    .\\.venv\\Scripts\\python.exe Ingestion_pipeline.py --rebuild  rebuild the whole index

Same as `python -m ingestion.run`. The stages live in the ingestion/ package:
    ingestion/fetch_web.py  (pre)        — snapshot university web pages into data/corpus/
    ingestion/manifest.py   00 validate  — find files, check projects.csv / seniors.csv
    ingestion/parsers.py    01 parse     — PDF / DOCX / PPTX / MD / TXT / HTML / VTT / SRT
                                           -> LangChain Documents
    ingestion/chunker.py    02 chunk     — RecursiveCharacterTextSplitter + breadcrumbs
    ingestion/embedder.py   03 embed     — dense + sparse vectors
    ingestion/run.py        03 index     — upsert into Qdrant, verify, write logs/ingest_*.json
"""
import sys

from ingestion.run import main

if __name__ == "__main__":
    sys.exit(main())
