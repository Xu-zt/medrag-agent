"""MedRAG-Agent MCP Server (Week 3).

Exposes two tools to Claude Desktop / Claude Code:
  - retrieve(query, pipeline, k) → list of retrieved document snippets
  - ask(query, pipeline, k)      → LLM-generated answer with citations

Run for local development:
    mcp dev src/medrag/mcp_server/server.py

Install into Claude Desktop (run once):
    mcp install src/medrag/mcp_server/server.py --name "MedRAG-Agent"

Supported pipelines: p1, p2, p3, p4, p5
"""
from __future__ import annotations

# Windows + CUDA: preload pyarrow before torch to avoid access violation (0xC0000005)
import pyarrow.dataset  # noqa: F401

import os
import sys

# Force UTF-8 for Windows terminals
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastmcp import FastMCP
from qdrant_client import QdrantClient

from medrag.agent.generator import generate_answer
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.hyde import HyDERetriever
from medrag.retrieval.multi_query import MultiQueryRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.retrieval.retriever import DenseRetriever, RetrievedChunk

# ---------------------------------------------------------------------------
# Lazy-initialised singletons — loaded on first tool call, not at import time
# ---------------------------------------------------------------------------
_qdrant: QdrantClient | None = None
_embedder: BGEM3Embedder | None = None
_reranker: BGEReranker | None = None
_hyde: HyDERetriever | None = None
_mq: MultiQueryRetriever | None = None


def _get_resources():
    global _qdrant, _embedder, _reranker, _hyde, _mq
    if _qdrant is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
        _qdrant = QdrantClient(url=qdrant_url, timeout=30)
        _embedder = BGEM3Embedder(device="cpu")
        _reranker = BGEReranker(device="cpu")
        _hyde = HyDERetriever(_qdrant, _embedder)
        _mq = MultiQueryRetriever(_qdrant, _embedder)
    return _qdrant, _embedder, _reranker, _hyde, _mq


def _run_pipeline(
    query: str,
    pipeline: str,
    k: int,
) -> list[RetrievedChunk]:
    qdrant, embedder, reranker, hyde, mq = _get_resources()
    pipeline = pipeline.lower().strip()

    if pipeline == "p1":
        return DenseRetriever(qdrant, embedder).retrieve(query, k=k)

    if pipeline == "p2":
        return HybridRetriever(qdrant, embedder).retrieve(query, k=k)

    if pipeline == "p3":
        candidates = HybridRetriever(qdrant, embedder, candidate_k=20).retrieve(query, k=20)
        return reranker.rerank(query, candidates, top_k=k)

    if pipeline == "p4":
        return hyde.retrieve(query, k=k)

    if pipeline == "p5":
        return mq.retrieve(query, k=k)

    raise ValueError(
        f"Unknown pipeline '{pipeline}'. Choose from: p1, p2, p3, p4, p5."
    )


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "MedRAG-Agent",
    instructions=(
        "MedRAG-Agent provides retrieval-augmented QA over a PubMed/PMC corpus. "
        "Use 'retrieve' to get relevant document snippets, and 'ask' to get an "
        "LLM-generated answer with inline citations. "
        "Pipeline options: p1 (dense), p2 (hybrid RRF), p3 (hybrid+reranker, "
        "best quality), p4 (HyDE, good for complex queries), p5 (multi-query, "
        "best coverage)."
    ),
)


@mcp.tool()
def retrieve(
    query: str,
    pipeline: str = "p3",
    k: int = 5,
) -> list[dict]:
    """Retrieve top-k relevant medical document chunks for a query.

    Args:
        query: The medical question or search query.
        pipeline: Retrieval pipeline to use.
            - p1: Dense-only (BGE-M3 cosine similarity) — fastest
            - p2: Hybrid dense+sparse with RRF fusion — better for exact terms
            - p3: Hybrid + cross-encoder reranker — highest precision (default)
            - p4: HyDE — generates a hypothetical answer first, good for complex queries
            - p5: Multi-Query — expands to 4 phrasings then fuses, best coverage
        k: Number of documents to return (1-10).

    Returns:
        List of dicts with keys: rank, citation, score, snippet, source, doc_id.
    """
    k = max(1, min(k, 10))
    chunks = _run_pipeline(query, pipeline, k)
    return [
        {
            "rank": i + 1,
            "citation": c.citation,
            "score": round(float(c.score), 4),
            "snippet": c.text[:500],
            "source": c.payload.get("source", ""),
            "doc_id": c.payload.get("doc_id", ""),
        }
        for i, c in enumerate(chunks)
    ]


@mcp.tool()
def ask(
    query: str,
    pipeline: str = "p3",
    k: int = 5,
) -> str:
    """Answer a medical question using retrieved literature with inline citations.

    Retrieves relevant PubMed/PMC documents using the specified pipeline, then
    generates a grounded answer using Qwen3-8B (running locally via Ollama).
    The answer will cite sources as [PMID:xxx] or [PMC:xxx] inline.

    Args:
        query: The medical question to answer.
        pipeline: Retrieval pipeline (p1/p2/p3/p4/p5). Default p3.
        k: Number of source documents to retrieve (1-10). Default 5.

    Returns:
        A grounded answer string with inline citations.
    """
    k = max(1, min(k, 10))
    chunks = _run_pipeline(query, pipeline, k)
    if not chunks:
        return "No relevant documents found in the corpus for this query."
    return generate_answer(query, chunks)


if __name__ == "__main__":
    mcp.run()
