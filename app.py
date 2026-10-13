"""Ask the scikit-learn User Guide.

Run locally:  streamlit run app.py
Set ANTHROPIC_API_KEY to get LLM-written answers; without it the app gives extractive answers.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from docqa.pipeline import DocQA  # noqa: E402

st.set_page_config(page_title="Ask the scikit-learn Docs", layout="wide")


@st.cache_resource
def load():
    return DocQA()


qa = load()
st.title("Ask the scikit-learn User Guide")
st.caption(f"{len(qa.chunks):,} chunks from {len({c.source for c in qa.chunks})} pages · "
           f"retrieval: {qa.retriever.name} · answers: {qa.generator.name}")

examples = ["How do I evaluate a model on time series data?",
            "Which scaler is robust to outliers?",
            "How do I choose a decision threshold other than 0.5?",
            "Can gradient boosting handle missing values?"]
choice = st.selectbox("Try an example or type your own below", [""] + examples)
question = st.text_input("Your question", value=choice)
k = st.sidebar.slider("Sections to retrieve", 2, 8, 4)

if question:
    answer = qa.ask(question, k=k)
    st.subheader("Answer")
    st.write(answer.text)
    st.subheader("Sources")
    for i, c in enumerate(answer.sources, 1):
        with st.expander(f"[{i}] {c.citation}"):
            st.write(c.text)

st.divider()
st.caption("Built by Muhammad Ehtesham · github.com/mehtesham13/rag-docs-qa")
