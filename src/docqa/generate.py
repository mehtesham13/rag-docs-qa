"""Answer generation from retrieved chunks.

Two generators share one interface:
- ExtractiveGenerator works offline: it picks the sentences from the retrieved chunks
  that best match the question and cites where each came from.
- ClaudeGenerator sends the question and the numbered sources to an LLM with
  instructions to answer only from those sources and cite them. It needs the
  `anthropic` package and an ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .ingest import Chunk
from .retrievers import tokenize

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

PROMPT = """You answer questions about the scikit-learn User Guide.
Use only the numbered sources below. Cite sources inline like [1] or [2][3].
If the sources do not contain the answer, say "The documentation excerpts provided do not cover this."
Keep the answer under 120 words.

Sources:
{sources}

Question: {question}
Answer:"""


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    generator: str


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation}) {c.text}" for i, c in enumerate(chunks, 1))


class ExtractiveGenerator:
    name = "extractive (offline)"

    def __init__(self, n_sentences: int = 3, embed=None):
        self.n, self.embed = n_sentences, embed

    def __call__(self, question: str, chunks: list[Chunk]) -> Answer:
        q_tokens = set(tokenize(question))
        cands = []
        for idx, c in enumerate(chunks, 1):
            for pos, sent in enumerate(SENT_SPLIT.split(c.text)):
                words = sent.split()
                if not 6 <= len(words) <= 60:
                    continue
                toks = set(tokenize(sent))
                overlap = len(q_tokens & toks) / (len(q_tokens) or 1)
                sem = float(self.embed(sent) @ self.embed(question)) if self.embed else 0.0
                # Favor higher-ranked sources and earlier sentences, which tend to define the topic.
                score = overlap + 0.5 * sem + 0.15 / idx + 0.05 / (pos + 1)
                cands.append((score, idx, pos, sent.strip()))
        best = sorted(cands, reverse=True)[: self.n]
        best.sort(key=lambda x: (x[1], x[2]))  # keep reading order
        text = " ".join(f"{s} [{i}]" for _, i, _, s in best) or "No relevant passage found."
        return Answer(text, chunks, self.name)


class ClaudeGenerator:
    name = "Claude (LLM)"

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 400):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def __call__(self, question: str, chunks: list[Chunk]) -> Answer:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": PROMPT.format(sources=format_sources(chunks), question=question)}])
        return Answer(msg.content[0].text.strip(), chunks, f"{self.name}: {self.model}")


def default_generator(embed=None):
    """Use the LLM when an API key is available, otherwise fall back to extractive answers."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeGenerator(model=os.environ.get("DOCQA_MODEL", "claude-sonnet-4-5"))
        except ImportError:
            pass
    return ExtractiveGenerator(embed=embed)


def cited_ids(answer_text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
