"""Build and query a local hybrid index for official student sources."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import logging
import math
from threading import Lock
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from retrieve import COLLECTIONS, PROJECT_DIR, keywords, load_passages


MODEL = "bge-m3:567m"
SNAPSHOT = PROJECT_DIR / "index" / "official-hybrid.jsonl"
SCHEMA = 1
MIN_COSINE = 0.55
COLLECTION = "official_student_sources"


def fingerprint(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(payload).hexdigest()


def embed(texts):
    request = Request(
        "http://127.0.0.1:11434/api/embed",
        data=json.dumps({"model": MODEL, "input": texts, "truncate": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=90) as response:
        vectors = json.load(response)["embeddings"]
    if len(vectors) != len(texts) or any(len(vector) != 1024 for vector in vectors):
        raise ValueError("Expected one 1024-dimensional BGE-M3 vector per input.")
    return vectors


def make_chunks(rows):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = [
        Document(page_content=row["text"], metadata={key: value for key, value in row.items() if key != "text"})
        for row in rows
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return [{**doc.metadata, "text": doc.page_content} for doc in splitter.split_documents(documents)]


class HybridSearch:
    def __init__(self, chunks, vectors):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(":memory:")
        self.lock = Lock()
        self.client.create_collection(
            COLLECTION,
            vectors_config={"meaning": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={"words": models.SparseVectorParams()},
        )
        tokens = [keywords(chunk["text"]) for chunk in chunks]
        counts = Counter(word for terms in tokens for word in terms)
        self.vocabulary = {word: index for index, word in enumerate(sorted(counts))}
        size = len(chunks)
        average_length = sum(map(len, tokens)) / size or 1
        points = []
        for index, (chunk, vector, terms) in enumerate(zip(chunks, vectors, tokens, strict=True)):
            words = sorted(terms)
            # BM25 with binary term frequency, using distinct terms for length normalization.
            length_factor = 1.2 * (0.25 + 0.75 * len(words) / average_length)
            weights = [
                math.log1p((size - counts[word] + 0.5) / (counts[word] + 0.5))
                * 2.2 / (1 + length_factor)
                for word in words
            ]
            named = {"meaning": vector}
            if words:
                named["words"] = models.SparseVector(
                    indices=[self.vocabulary[word] for word in words], values=weights,
                )
            points.append(models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{index}:{fingerprint([chunk])}")),
                vector=named, payload=chunk,
            ))
        self.client.upsert(COLLECTION, points=points)
        if self.client.count(COLLECTION, exact=True).count != len(chunks):
            raise ValueError("Search index did not store every chunk.")

    def rank(self, question, urls=None, *, min_cosine=None, limit=5):
        models = self.models
        if urls is not None and not urls:
            return []
        filter_by = None if urls is None else models.Filter(
            must=[models.FieldCondition(key="url", match=models.MatchAny(any=sorted(urls)))],
        )
        meaning = embed([question])[0]
        threshold = MIN_COSINE if min_cosine is None else min_cosine
        dense = models.Prefetch(query=meaning, using="meaning", filter=filter_by, score_threshold=threshold, limit=20)
        indices = sorted(self.vocabulary[word] for word in keywords(question) if word in self.vocabulary)
        with self.lock:
            if indices:
                sparse = models.Prefetch(
                    query=models.SparseVector(indices=indices, values=[1.0] * len(indices)),
                    using="words", filter=filter_by, score_threshold=0.0, limit=20,
                )
                points = self.client.query_points(
                    COLLECTION, prefetch=[dense, sparse],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=20, with_payload=True,
                ).points
                # A one-word match alone is weak evidence. Ask for either qualifying
                # semantic similarity or the existing minimum lexical overlap.
                semantic_ids = {
                    point.id for point in self.client.query_points(
                        COLLECTION, query=meaning, using="meaning", query_filter=filter_by,
                        score_threshold=threshold, limit=20, with_payload=False,
                    ).points
                }
                terms = keywords(question)
                required = min(2, len(terms))
                points = [
                    point for point in points if point.id in semantic_ids
                    or len(terms & keywords(point.payload["text"])) >= required
                ]
            else:
                points = self.client.query_points(
                    COLLECTION, query=meaning, using="meaning", query_filter=filter_by,
                    score_threshold=threshold, limit=20, with_payload=True,
                ).points

        found = {}
        for point in points:
            row = point.payload
            identity = row.get("url") or (row["source"], row["page"])
            found.setdefault(identity, row)
        return list(found.values())[:limit]


def load_hybrid(rows, *, required=False):
    try:
        stored = load_passages(SNAPSHOT)
        meta, *records = stored
        if meta != {"schema": SCHEMA, "model": MODEL, "source_hash": fingerprint(rows)}:
            raise ValueError("Hybrid snapshot is stale. Run src/hybrid.py to rebuild it.")
        return HybridSearch([record["chunk"] for record in records], [record["vector"] for record in records])
    except (ValueError, ImportError, OSError, KeyError) as exc:
        if required:
            raise
        if SNAPSHOT.exists():
            logging.warning("Hybrid index unavailable. Using keyword search: %s", exc)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Inspect ranked sources without generating an answer")
    args = parser.parse_args()
    try:
        rows = load_passages(COLLECTIONS["official"])
        if args.query:
            from retrieve import source_location
            search = load_hybrid(rows, required=True)
            for rank, match in enumerate(search.rank(args.query), 1):
                print(f"{rank}. {source_location(match)}")
            return
        chunks = make_chunks(rows)
        if not chunks:
            raise ValueError("No official passages to index.")
        vectors = []
        for start in range(0, len(chunks), 16):
            batch = chunks[start:start + 16]
            vectors.extend(embed([chunk["text"] for chunk in batch]))
        HybridSearch(chunks, vectors)
        from ingest import write_index
        meta = {"schema": SCHEMA, "model": MODEL, "source_hash": fingerprint(rows)}
        write_index(SNAPSHOT, [meta] + [
            {"chunk": chunk, "vector": vector} for chunk, vector in zip(chunks, vectors, strict=True)
        ])
    except (OSError, ValueError, ImportError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(chunks)} verified chunks to {SNAPSHOT}")


if __name__ == "__main__":
    main()
