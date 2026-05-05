"""MedRAG-Agent Streamlit Demo (Week 2).

Run:
    streamlit run src/medrag/ui/app.py

Features:
- Pipeline selector: P1 (dense) / P2 (hybrid RRF) / P3 (hybrid + reranker)
- Retrieved document cards with score and citation
- Qwen3-8B answer with inline citations
- In-session chat history
"""
# Windows + CUDA: preload pyarrow before torch to avoid AV 0xC0000005
import pyarrow.dataset  # noqa: F401

import streamlit as st
from qdrant_client import QdrantClient

from medrag.agent.generator import generate_answer
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.retrieval.retriever import DenseRetriever

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG-Agent",
    page_icon="🏥",
    layout="wide",
)

# ── Load heavy resources once per session ────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")
    reranker = BGEReranker()
    return qdrant, embedder, reranker


qdrant, embedder, reranker = load_resources()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    pipeline = st.radio(
        "Retrieval Pipeline",
        options=["P1 · Dense Only", "P2 · Hybrid (Dense + Sparse)", "P3 · Hybrid + Reranker"],
        index=2,
    )
    top_k = st.slider("Top-K documents", min_value=3, max_value=10, value=5)

    st.divider()
    st.markdown("""
**Pipeline details**

| | Method | When it helps |
|---|---|---|
| P1 | BGE-M3 dense cosine | General semantic similarity |
| P2 | P1 + sparse (neural BM25) fused via RRF | Exact medical terms, acronyms |
| P3 | P2 top-20 → cross-encoder rerank | Highest precision, slower |
    """)

    st.divider()
    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()

    st.caption("Corpus: PubMed + PMC · LLM: Qwen3-8B (local)")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏥 MedRAG-Agent")
st.caption("Local medical literature QA · dense + sparse retrieval · Qwen3-8B")

if "history" not in st.session_state:
    st.session_state.history = []

# Render previous turns
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"📚 {len(turn['chunks'])} source documents  ·  [{turn['pipeline']}]"):
            for i, c in enumerate(turn["chunks"], 1):
                st.markdown(f"**[{i}] {c.citation}** &nbsp; `score={c.score:.4f}`")
                st.caption(c.text[:350] + ("…" if len(c.text) > 350 else ""))
                st.divider()

# Input
query = st.chat_input("Ask a medical question (English works best)…")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        mode_key = pipeline.split("·")[0].strip()  # "P1", "P2", or "P3"

        with st.spinner(f"Retrieving [{mode_key}]…"):
            if mode_key == "P1":
                retriever = DenseRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            elif mode_key == "P2":
                retriever = HybridRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            else:  # P3
                retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
                candidates = retriever.retrieve(query, k=20)
                chunks = reranker.rerank(query, candidates, top_k=top_k)

        with st.spinner("Generating answer (Qwen3-8B)…"):
            answer = generate_answer(query, chunks)

        st.write(answer)

        with st.expander(f"📚 {len(chunks)} source documents  ·  [{mode_key}]"):
            for i, c in enumerate(chunks, 1):
                col_meta, col_text = st.columns([1, 3])
                with col_meta:
                    st.metric(label=f"#{i} {c.citation}", value=f"{c.score:.4f}", delta=None)
                with col_text:
                    st.caption(c.text[:350] + ("…" if len(c.text) > 350 else ""))
                st.divider()

        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "pipeline": mode_key,
        })
