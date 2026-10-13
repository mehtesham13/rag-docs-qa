"""Retrieval evaluation against a hand-labeled question set."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .retrievers import dedupe_by_section


def is_relevant(chunk, row) -> bool:
    pats = [p.strip().lower() for p in row["relevant_sections"].split(";")]
    return chunk.source == row["source"] and any(p in chunk.section.lower() for p in pats)


def evaluate(retriever, questions: pd.DataFrame, k_values=(1, 3, 5)) -> dict:
    """Hit rate @k (a relevant section is in the top k) and mean reciprocal rank."""
    kmax = max(k_values)
    hits = {k: [] for k in k_values}
    rr, latency = [], []
    for _, row in questions.iterrows():
        t0 = time.perf_counter()
        results = dedupe_by_section(retriever.search(row["question"], k=60), kmax)
        latency.append(time.perf_counter() - t0)
        rel = [is_relevant(c, row) for c, _ in results]
        first = next((i for i, r in enumerate(rel) if r), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        for k in k_values:
            hits[k].append(any(rel[:k]))
    out = {f"hit@{k}": float(np.mean(v)) for k, v in hits.items()}
    out["MRR"] = float(np.mean(rr))
    out["ms/query"] = 1000 * float(np.mean(latency))
    return out
