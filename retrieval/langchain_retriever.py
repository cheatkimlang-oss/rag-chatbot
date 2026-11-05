"""LangChain BaseRetriever adapter over the hybrid retriever (FR-05, Spec §9.6).

Exposes `invoke(query)` / `ainvoke(query)` returning LangChain `Document`s, so
the retrieval pipeline composes into LCEL chains (`retriever | prompt | llm`)
while keeping project grouping intact: one Document per distinct project.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models

from retrieval.hybrid import SearchConfig, hybrid_retrieve
from retrieval.intent import segment_for_lexical
from store.vector_schema import COLLECTION


class SeniorProjectKBRetriever(BaseRetriever):
    """`search` carries every retrieval knob (mode, fusion, rrf_k, hnsw_ef, ...);
    build it with `SearchConfig.from_settings(settings)` to match the bot exactly."""
    client: Any
    embedder: Any
    collection: str = COLLECTION
    top_k: int = 5
    search: SearchConfig = SearchConfig()
    flt: models.Filter | None = None

    @classmethod
    def from_settings(cls, client, embedder, settings, **kw) -> "SeniorProjectKBRetriever":
        return cls(client=client, embedder=embedder, collection=settings.collection,
                   top_k=settings.display_top_k, search=SearchConfig.from_settings(settings), **kw)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = hybrid_retrieve(self.client, self.embedder, query,
                               lexical_query=segment_for_lexical(query), flt=self.flt,
                               collection=self.collection, search=self.search, top_k=self.top_k)
        return [
            Document(
                page_content=h.payload["text"],
                metadata={
                    "project_id": h.project_id,
                    "title": h.payload.get("title_en"),
                    "senior_ids": h.payload.get("senior_ids", []),
                    "doc_id": h.payload.get("doc_id"),
                    "page_number": h.payload.get("page_number"),
                    "section": h.payload.get("section"),
                    "score": h.score,
                    "dense_score": h.dense_score,
                    "matched_by": h.matched_by,
                },
            )
            for h in hits
        ]
