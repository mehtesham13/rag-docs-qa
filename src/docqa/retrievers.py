"""Retrievers: BM25 (keyword), LSA and word-vector embeddings (semantic), and hybrid fusion."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be by can do does for from how i if in is it its of on or
should the this to what when which why with you your my me we""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def _doc_text(c: Chunk) -> str:
    # Repeating the heading gives section titles extra weight, as titles summarize the chunk.
    return f"{c.section}. {c.section}. {c.text}"


class Retriever:
    name = "base"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def scores(self, query: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


class BM25Retriever(Retriever):
    """Classic keyword ranking: rewards rare query words that appear often in a chunk."""
    name = "BM25"

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.bm25 = BM25Okapi([tokenize(_doc_text(c)) for c in chunks], k1=k1, b=b)

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)))


class LSARetriever(Retriever):
    """Latent semantic analysis: TF-IDF compressed with SVD, so related words share dimensions."""
    name = "LSA embeddings"

    def __init__(self, chunks, dims: int = 200, seed: int = 0):
        super().__init__(chunks)
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True,
                                   ngram_range=(1, 2), min_df=2)
        X = self.vec.fit_transform([_doc_text(c) for c in chunks])
        self.svd = TruncatedSVD(dims, random_state=seed)
        self.emb = normalize(self.svd.fit_transform(X))

    def embed(self, text):
        return normalize(self.svd.transform(self.vec.transform([text])))[0]

    def scores(self, query):
        return self.emb @ self.embed(query)


class WordVectorRetriever(Retriever):
    """Pretrained word vectors (spaCy en_core_web_md), averaged with IDF weights."""
    name = "Word-vector embeddings"

    def __init__(self, chunks, model: str = "en_core_web_md"):
        import spacy
        super().__init__(chunks)
        self.nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
        tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None).fit([_doc_text(c) for c in chunks])
        self.idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        self.emb = normalize(np.vstack([self._vec(_doc_text(c)) for c in chunks]))

    def _vec(self, text):
        toks = tokenize(text)
        vocab = self.nlp.vocab
        vecs, weights = [], []
        for t in toks:
            lex = vocab[t]
            if lex.has_vector:
                vecs.append(lex.vector)
                weights.append(self.idf.get(t, 1.0))
        if not vecs:
            return np.zeros(vocab.vectors.shape[1])
        return np.average(np.array(vecs), axis=0, weights=weights)

    def scores(self, query):
        q = self._vec(query)
        n = np.linalg.norm(q)
        return self.emb @ (q / n if n else q)


class HybridRetriever(Retriever):
    """Reciprocal rank fusion: add 1 / (k + rank) from each retriever, so a chunk ranked
    well by several methods rises to the top, and no score scaling is needed."""

    def __init__(self, retrievers: list[Retriever], k: int = 60, weights=None):
        super().__init__(retrievers[0].chunks)
        self.retrievers, self.k = retrievers, k
        self.weights = weights or [1.0] * len(retrievers)
        self.name = "Hybrid (" + " + ".join(r.name.split()[0] for r in retrievers) + ")"

    def scores(self, query):
        fused = np.zeros(len(self.chunks))
        for r, w in zip(self.retrievers, self.weights):
            ranks = np.empty(len(self.chunks))
            ranks[np.argsort(-r.scores(query))] = np.arange(1, len(self.chunks) + 1)
            fused += w / (self.k + ranks)
        return fused


def dedupe_by_section(results: list[tuple[Chunk, float]], k: int) -> list[tuple[Chunk, float]]:
    """Keep the best chunk per section so the answer draws on several sections."""
    seen, out = set(), []
    for c, s in results:
        key = (c.source, c.section)
        if key not in seen:
            seen.add(key)
            out.append((c, s))
        if len(out) == k:
            break
    return out
