"""MedRAG-Agent MCP Server (Week 5 — LangGraph + Security).

Tools exposed to Claude Desktop / Claude Code:
  1. search_literature   — hybrid retrieval (P2/P3), returns document snippets
  2. ask_agent           — full LangGraph agentic loop with rewrite + faithfulness check
  3. evaluate_query      — grade how well a set of chunks answers a query (no generation)
  4. search_visual       — stub for future visual / image search capability

Security middleware (applied in order):
  1. auth            — MEDRAG_LOCAL_TOKEN env var (disabled if not set → dev mode)
  2. rate_limit      — 30 rpm global, 10 rpm for ask_agent
  3. injection_guard — prompt injection detection before retrieval
  4. pii             — PII redaction in audit log (query hash only stored)
  5. audit           — structured JSON-Lines to data/logs/audit.jsonl

Run for local development:
    mcp dev src/medrag/mcp_server/server.py

Install into Claude Desktop (run once):
    mcp install src/medrag/mcp_server/server.py --name "MedRAG-Agent"
"""
from __future__ import annotations

# Windows + CUDA: preload pyarrow before torch to avoid access violation (0xC0000005)
import pyarrow.dataset  # noqa: F401

import io
import logging
import os
import sys
import time

# Force UTF-8 for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastmcp import FastMCP

from medrag.mcp_server.security import (
    AuthError,
    InjectionGuardError,
    RateLimitError,
    audit,
    check_injection,
    check_rate_limit,
    log_tool_call,
    sanitise_query,
    verify_token,
    wrap_document,
)

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Lazy singletons ────────────────────────────────────────────────────────────

_retriever = None
_reranker  = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from qdrant_client import QdrantClient
        from medrag.index.embedder import BGEM3Embedder
        from medrag.retrieval.hybrid import HybridRetriever
        qdrant   = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=30)
        embedder = BGEM3Embedder(device="cpu")
        _retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
    return _retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from medrag.retrieval.reranker import BGEReranker
        _reranker = BGEReranker(device="cpu")
    return _reranker


# ── Security helpers ───────────────────────────────────────────────────────────

def _security_check(query: str, token: str, is_generate: bool = False) -> str:
    """Run auth → rate_limit → injection_guard; return sanitised query.

    Raises AuthError, RateLimitError, or InjectionGuardError on violation.
    """
    verify_token(token)
    check_rate_limit(is_generate=is_generate)
    return sanitise_query(query)


# ── MCP server ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "MedRAG-Agent",
    instructions=(
        "MedRAG-Agent provides retrieval-augmented QA over a PubMed/PMC medical corpus. "
        "Tools: "
        "'search_literature' — retrieve relevant document snippets (fast, P2/P3); "
        "'ask_agent' — full agentic loop: retrieves, grades, rewrites if needed, "
        "generates a grounded answer with inline citations and faithfulness check; "
        "'evaluate_query' — grade how well given context answers a query; "
        "'search_visual' — stub for image/figure search (not yet implemented). "
        "All tools require MEDRAG_LOCAL_TOKEN if set in server environment."
    ),
)


@mcp.tool()
def search_literature(
    query: str,
    k: int = 5,
    rerank: bool = True,
    token: str = "",
) -> list[dict]:
    """Retrieve top-k relevant medical document chunks from PubMed/PMC.

    Performs hybrid dense+sparse RRF retrieval (P2), optionally followed
    by BGE cross-encoder reranking (P3 quality).

    Args:
        query: Medical question or search query.
        k: Number of documents to return (1–10, default 5).
        rerank: If True (default), apply cross-encoder reranking for highest precision.
        token: Optional auth token (MEDRAG_LOCAL_TOKEN).

    Returns:
        List of dicts: rank, citation, score, snippet (500 chars), source, doc_id.
    """
    t0 = time.perf_counter()
    status = "ok"
    try:
        sanitised = _security_check(query, token, is_generate=False)
        k = max(1, min(k, 10))

        retriever = _get_retriever()
        chunks = retriever.retrieve(sanitised, k=20 if rerank else k)

        if rerank and chunks:
            chunks = _get_reranker().rerank(sanitised, chunks, top_k=k)
        else:
            chunks = chunks[:k]

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
    except (AuthError, RateLimitError, InjectionGuardError) as exc:
        status = f"rejected:{type(exc).__name__}"
        raise
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        log_tool_call("search_literature", query, status,
                      (time.perf_counter() - t0) * 1000)


@mcp.tool()
def ask_agent(
    query: str,
    thread_id: str = "default",
    token: str = "",
) -> dict:
    """Answer a medical question using the full LangGraph agentic loop.

    Pipeline:
      1. Hybrid retrieval (dense + sparse RRF)
      2. Cross-encoder reranking
      3. Relevance grading — rewrites query up to 2× if chunks are insufficient
      4. Answer generation with inline citations
      5. Faithfulness check — re-generates once if answer contains hallucinations

    Multi-turn: supply the same thread_id across calls to maintain context.
    The agent compresses history via rolling summarisation every 10 turns.

    Args:
        query: Medical question to answer.
        thread_id: Session identifier for multi-turn memory (default: "default").
        token: Optional auth token (MEDRAG_LOCAL_TOKEN).

    Returns:
        Dict with keys: answer, citations, confidence, faithful, faithfulness_issues,
        iterations (rewrites performed), regen_count.
    """
    t0 = time.perf_counter()
    status = "ok"
    try:
        sanitised = _security_check(query, token, is_generate=True)

        from medrag.agent.graph import app

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "query": sanitised,
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "relevance_score": 0.0,
            "grade_reason": "",
            "rewrite_hint": "",
            "iterations": 0,
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "faithful": False,
            "faithfulness_issues": "",
            "regen_count": 0,
            "history": [{"query": sanitised, "answer": ""}],
            "summary": "",
        }

        result = app.invoke(initial_state, config=config)

        # Update history with the answer
        if result.get("answer"):
            history = result.get("history", [])
            if history and history[-1].get("query") == sanitised:
                history[-1]["answer"] = result["answer"]

        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
            "faithful": result.get("faithful", False),
            "faithfulness_issues": result.get("faithfulness_issues", ""),
            "iterations": result.get("iterations", 0),
            "regen_count": result.get("regen_count", 0),
        }
    except (AuthError, RateLimitError, InjectionGuardError) as exc:
        status = f"rejected:{type(exc).__name__}"
        raise
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        log_tool_call("ask_agent", query, status,
                      (time.perf_counter() - t0) * 1000)


@mcp.tool()
def evaluate_query(
    query: str,
    context_chunks: list[str],
    token: str = "",
) -> dict:
    """Grade whether provided context chunks can fully answer a query.

    Useful for debugging retrieval quality or testing custom contexts.
    Uses the same LLM grader as the agentic loop (thinking mode ON).

    Args:
        query: The medical question to evaluate against.
        context_chunks: List of document text strings to evaluate.
        token: Optional auth token (MEDRAG_LOCAL_TOKEN).

    Returns:
        Dict: relevant (bool), score (0–1), reason (str), rewrite_hint (str).
    """
    t0 = time.perf_counter()
    status = "ok"
    try:
        sanitised = _security_check(query, token, is_generate=False)

        from medrag.agent.nodes import grade_relevance
        from medrag.retrieval.retriever import RetrievedChunk

        # Wrap plain strings as minimal RetrievedChunk objects
        chunks = [
            RetrievedChunk(
                chunk_id=f"user-{i}",
                text=c,
                score=1.0,
                payload={"source": "user", "doc_id": f"user-{i}"},
            )
            for i, c in enumerate(context_chunks[:10])  # cap at 10
        ]

        # Build minimal state for the grade node
        state = {
            "query": sanitised,
            "retrieved_chunks": chunks,
            "relevance_score": 0.0,
            "grade_reason": "",
            "rewrite_hint": "",
            "iterations": 0,
            "rewritten_queries": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "faithful": False,
            "faithfulness_issues": "",
            "regen_count": 0,
            "history": [],
            "summary": "",
        }

        result = grade_relevance(state)
        return {
            "relevant": result["relevance_score"] >= 0.6,
            "score": result["relevance_score"],
            "reason": result["grade_reason"],
            "rewrite_hint": result["rewrite_hint"],
        }
    except (AuthError, RateLimitError, InjectionGuardError) as exc:
        status = f"rejected:{type(exc).__name__}"
        raise
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        log_tool_call("evaluate_query", query, status,
                      (time.perf_counter() - t0) * 1000)


@mcp.tool()
def search_visual(
    query: str,
    modality: str = "figure",
    k: int = 5,
    token: str = "",
) -> dict:
    """[STUB] Search for medical images, figures, or tables.

    This tool is not yet implemented. It will support searching PMC
    Open Access figures, radiology images, and anatomical diagrams
    when the visual index is built in a future release.

    Args:
        query: Description of the image or figure to search for.
        modality: Type of visual content: "figure", "table", "radiology".
        k: Number of results to return (1–10).
        token: Optional auth token (MEDRAG_LOCAL_TOKEN).

    Returns:
        Dict with status="not_implemented" and a message.
    """
    _security_check(query, token, is_generate=False)
    log_tool_call("search_visual", query, "stub", 0.0)
    return {
        "status": "not_implemented",
        "message": (
            "Visual search is not yet available. "
            "The PMC figure index is planned for a future release. "
            "Use search_literature for text-based retrieval."
        ),
        "modality": modality,
        "k": k,
    }


if __name__ == "__main__":
    mcp.run()
