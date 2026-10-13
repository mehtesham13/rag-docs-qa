import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docqa.evaluate import evaluate, is_relevant  # noqa: E402
from docqa.generate import ExtractiveGenerator, cited_ids  # noqa: E402
from docqa.ingest import Chunk, chunk_docs, clean_rst, parse_sections  # noqa: E402
from docqa.retrievers import BM25Retriever, HybridRetriever, LSARetriever, dedupe_by_section, tokenize  # noqa: E402

DOCS = ROOT / "data" / "docs"


@pytest.fixture(scope="module")
def chunks():
    return chunk_docs(DOCS)


def test_clean_rst_strips_roles_and_literals():
    assert clean_rst("Use :class:`~sklearn.svm.SVC` with ``C=1``.") == "Use SVC with C=1."


def test_parse_sections_finds_headings(tmp_path):
    f = tmp_path / "x.rst"
    f.write_text("Title\n=====\n\nIntro text.\n\nSub part\n--------\n\nBody text here.\n")
    heads = [h for h, _ in parse_sections(f)]
    assert heads == ["Title", "Sub part"]


def test_chunks_are_bounded_and_cited(chunks):
    assert len(chunks) > 500
    assert all(len(c.text.split()) <= 180 for c in chunks)
    assert all(c.source.endswith(".rst") and c.section for c in chunks)


def test_tokenize_drops_stopwords():
    assert tokenize("How do I scale the data?") == ["scale", "data"]


def test_bm25_finds_obvious_section(chunks):
    top = BM25Retriever(chunks).search("isolation forest anomaly", k=3)
    assert any("Isolation Forest" in c.section for c, _ in top)


def test_hybrid_beats_nothing_on_eval_set(chunks):
    q = pd.read_csv(ROOT / "data" / "eval_questions.csv")
    hybrid = HybridRetriever([BM25Retriever(chunks), LSARetriever(chunks)])
    m = evaluate(hybrid, q)
    assert m["hit@5"] >= 0.7  # regression guard: current value is 0.82


def test_dedupe_keeps_one_chunk_per_section():
    a = Chunk("1", "f.rst", "S", "t")
    b = Chunk("2", "f.rst", "S", "t")
    c = Chunk("3", "f.rst", "T", "t")
    assert [x.chunk_id for x, _ in dedupe_by_section([(a, 3), (b, 2), (c, 1)], 5)] == ["1", "3"]


def test_extractive_answer_cites_sources(chunks):
    srcs = [c for c in chunks if "Isolation Forest" in c.section][:2]
    ans = ExtractiveGenerator()("How does isolation forest find anomalies?", srcs)
    assert cited_ids(ans.text) and cited_ids(ans.text) <= set(range(1, len(srcs) + 1))


def test_is_relevant_matches_any_listed_section():
    row = {"source": "svm.rst", "relevant_sections": "Unbalanced problems;Tips"}
    assert is_relevant(Chunk("1", "svm.rst", "Tips on Practical Use", ""), row)
    assert not is_relevant(Chunk("1", "tree.rst", "Tips on practical use", ""), row)
