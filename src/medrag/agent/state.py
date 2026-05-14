"""LangGraph AgentState definition for MedRAG-Agent.

Two-tier memory architecture:
  L1 — LangGraph SqliteSaver checkpointer (crash recovery, multi-turn)
  L2 — rolling summarization every 10 turns (long-context compression)
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from medrag.retrieval.retriever import RetrievedChunk


class AgentState(TypedDict):
    # ── Core query ────────────────────────────────────────────────────────────
    query: str
    """Current (possibly rewritten) query sent to the retriever."""

    original_query: str
    """The user's original query before any rewrites.
    Set by route_query and preserved through the loop; used by append_history
    so conversation history records what the user actually asked."""

    query_type: str
    """Router classification: 'factual' | 'synthesis' | 'multihop'.
    Used by grade_relevance for dynamic threshold selection."""

    rewritten_queries: Annotated[list[str], add]
    """Accumulates each query rewrite for audit / tracing."""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    """Top-k chunks after reranking, ready for the generator."""

    # ── Grading ──────────────────────────────────────────────────────────────
    relevance_score: float
    """0-1 score from the grade node; < threshold triggers a rewrite."""

    relevant: bool
    """LLM's boolean relevance judgment from the grade node.
    Used by evaluate_query and for audit; routing uses relevance_score."""

    grade_reason: str
    """Human-readable explanation from the grade node (for audit log)."""

    rewrite_hint: str
    """Hint produced by grade node to guide the rewrite."""

    iterations: int
    """Rewrite counter. Starts at 0; incremented inside rewrite node.
    Hard cap: MAX_REWRITES = 1 (2 retrieval attempts total)."""

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    """Final answer string (thinking tags already stripped)."""

    citations: list[str]
    """List of PMID / PMCID strings cited inline in the answer."""

    confidence: float
    """Self-reported confidence 0-1 from the generate node."""

    # ── Faithfulness check ───────────────────────────────────────────────────
    faithful: bool
    """True if check node confirms the answer is grounded in context."""

    faithfulness_issues: str
    """Description of hallucinated claims (empty if faithful)."""

    regen_count: int
    """Re-generation counter. Hard cap: MAX_REGEN = 1."""

    # ── Memory ───────────────────────────────────────────────────────────────
    history: Annotated[list[dict], add]
    """Append-only conversation history (query + answer pairs)."""

    summary: str
    """Rolling summarization of older turns (L2 memory)."""
