"""End-to-end pipeline: load docs, build the hybrid index, retrieve, generate."""
from __future__ import annotations

from pathlib import Path

from .generate import Answer, default_generator
from .ingest import chunk_docs
from .retrievers import BM25Retriever, HybridRetriever, LSARetriever, WordVectorRetriever, dedupe_by_section

DOCS = Path(__file__).resolve().parents[2] / "data" / "docs"


class DocQA:
    # Word vectors are optional: in the evaluation they lowered hybrid accuracy (see notebook).
    def __init__(self, doc_dir=DOCS, use_word_vectors: bool = False, generator=None):
        self.chunks = chunk_docs(doc_dir)
        self.bm25 = BM25Retriever(self.chunks)
        self.lsa = LSARetriever(self.chunks)
        parts = [self.bm25, self.lsa]
        if use_word_vectors:
            try:
                parts.append(WordVectorRetriever(self.chunks))
            except OSError:  # spaCy model not installed
                pass
        self.retriever = HybridRetriever(parts)
        self.generator = generator or default_generator(embed=self.lsa.embed)

    def retrieve(self, question: str, k: int = 4):
        return dedupe_by_section(self.retriever.search(question, k=40), k)

    def ask(self, question: str, k: int = 4) -> Answer:
        chunks = [c for c, _ in self.retrieve(question, k)]
        return self.generator(question, chunks)
