"""MedRAG-Agent Streamlit Demo (Week 3).

Run:
    streamlit run src/medrag/ui/app.py

Features:
- Pipeline selector: P1/P2/P3/P4/P5
- Retrieved document cards with score and citation
- HyDE hypothesis display (P4) and multi-query expansion display (P5)
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
from medrag.retrieval.hyde import HyDERetriever
from medrag.retrieval.multi_query import MultiQueryRetriever
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
    hyde = HyDERetriever(qdrant, embedder)
    multi_query = MultiQueryRetriever(qdrant, embedder)
    return qdrant, embedder, reranker, hyde, multi_query


qdrant, embedder, reranker, hyde_retriever, mq_retriever = load_resources()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    pipeline = st.radio(
        "Retrieval Pipeline",
        options=[
            "P1 · Dense Only",
            "P2 · Hybrid (Dense + Sparse)",
            "P3 · Hybrid + Reranker",
            "P4 · HyDE",
            "P5 · Multi-Query",
        ],
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
| P4 | LLM hypothetical doc → dense retrieval | Complex multi-hop questions |
| P5 | 4 query rewrites → RRF fusion | Terminological variation |
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
            if turn.get("hypothesis"):
                st.info(f"**HyDE hypothesis:** {turn['hypothesis']}")
            if turn.get("sub_queries"):
                st.info("**Query expansions:** " + " · ".join(turn["sub_queries"]))
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
        mode_key = pipeline.split("·")[0].strip()  # "P1", "P2", ... "P5"

        hypothesis = None
        sub_queries = None

        with st.spinner(f"Retrieving [{mode_key}]…"):
            if mode_key == "P1":
                retriever = DenseRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            elif mode_key == "P2":
                retriever = HybridRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            elif mode_key == "P3":
                retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
                candidates = retriever.retrieve(query, k=20)
                chunks = reranker.rerank(query, candidates, top_k=top_k)
            elif mode_key == "P4":
                chunks = hyde_retriever.retrieve(query, k=top_k)
                hypothesis = hyde_retriever.last_hypothesis
            else:  # P5
                chunks = mq_retriever.retrieve(query, k=top_k)
                sub_queries = mq_retriever.last_queries

        with st.spinner("Generating answer (Qwen3-8B)…"):
            answer = generate_answer(query, chunks)

        st.write(answer)

        with st.expander(f"📚 {len(chunks)} source documents  ·  [{mode_key}]"):
            if hypothesis:
                st.info(f"**HyDE hypothesis:** {hypothesis}")
            if sub_queries:
                st.info("**Query expansions:** " + " · ".join(sub_queries))
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
            "hypothesis": hypothesis,
            "sub_queries": sub_queries,
        })
