"""LLM factory: dual-instance strategy for node-level thinking control.

llm_fast  → router, generate   (thinking OFF, low latency)
llm_think → grade, rewrite, check  (thinking ON, deep reasoning)

See docs/architecture.md §4.1.1 for the design rationale.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama


def make_llm_fast(model: str = "qwen3:8b") -> ChatOllama:
    """Thinking OFF — for router and generate nodes.

    Low latency, deterministic output.  num_ctx=4096 is sufficient for
    direct retrieval-grounded generation (5 chunks × ~300 tokens).
    """
    return ChatOllama(
        model=model,
        base_url="http://127.0.0.1:11434",
        reasoning=False,
        temperature=0.2,
        num_ctx=4096,
    )


def make_llm_think(model: str = "qwen3:8b") -> ChatOllama:
    """Thinking ON — for grade, rewrite, check nodes.

    Deeper reasoning at the cost of ~3-8 s extra latency.
    num_ctx=6144 accommodates <think>...</think> tokens + structured output.
    """
    return ChatOllama(
        model=model,
        base_url="http://127.0.0.1:11434",
        reasoning=True,
        temperature=0.6,
        num_ctx=6144,
    )


__all__ = ["make_llm_fast", "make_llm_think"]
