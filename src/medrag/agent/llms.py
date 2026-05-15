"""LLM factory: dual-backend strategy with thinking control per node.

Backend selection via environment variable LLM_BACKEND (default: mimo):

  LLM_BACKEND=mimo    → ChatOpenAI pointing at MiMo-V2.5 API
  LLM_BACKEND=ollama  → ChatOllama pointing at local Qwen3-8B (fallback)

Two tiers per backend:
  make_llm_fast()   → router, generate, summarize nodes (thinking OFF)
  make_llm_think()  → grade, rewrite, check nodes (thinking ON / Pro tier)

MiMo env vars (read from .env):
  OPENAI_BASE_URL   — MiMo API base URL
  OPENAI_API_KEY    — MiMo API key
  MIMO_MODEL_FAST   — override fast model name  (default: mimo-v2.5)
  MIMO_MODEL_THINK  — override think model name (default: mimo-v2.5-pro)

Ollama env vars:
  OLLAMA_MODEL      — override model name (default: qwen3:8b)

See docs/architecture.md §4.1.1 for design rationale.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────

_BACKEND = os.environ.get("LLM_BACKEND", "mimo").strip().lower()

# ── MiMo model names ───────────────────────────────────────────────────────────
_MIMO_FAST  = os.environ.get("MIMO_MODEL_FAST",  "mimo-v2.5")
_MIMO_THINK = os.environ.get("MIMO_MODEL_THINK", "mimo-v2.5-pro")

# ── Ollama model name ──────────────────────────────────────────────────────────
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def _mimo_base_url() -> str:
    url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE", "")
    ).rstrip("/")
    if not url:
        raise EnvironmentError(
            "MiMo backend requires OPENAI_BASE_URL (or OPENAI_API_BASE) in .env"
        )
    return url


def _mimo_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("MiMo backend requires OPENAI_API_KEY in .env")
    return key


# ── Internal factory ───────────────────────────────────────────────────────────

def _make_llm(thinking: bool):
    """Shared factory — `thinking` selects tier (fast=OFF / think=ON/Pro)."""
    temp  = 0.6 if thinking else 0.2
    model = _MIMO_THINK if thinking else _MIMO_FAST

    if _BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        logger.debug("[llm] %s → Ollama %s (reasoning=%s)", "think" if thinking else "fast",
                     _OLLAMA_MODEL, thinking)
        return ChatOllama(
            model=_OLLAMA_MODEL,
            base_url="http://127.0.0.1:11434",
            reasoning=thinking,
            temperature=temp,
            num_ctx=6144 if thinking else 4096,
        )

    # Default: mimo
    from langchain_openai import ChatOpenAI
    logger.debug("[llm] %s → MiMo %s", "think" if thinking else "fast", model)
    return ChatOpenAI(
        model=model,
        base_url=_mimo_base_url(),
        api_key=_mimo_api_key(),
        temperature=temp,
        max_tokens=4096,
    )


# ── Public factories ───────────────────────────────────────────────────────────

def make_llm_fast():
    """Low-latency LLM — thinking OFF. Used by: route_query, generate_answer_node, summarize_history."""
    return _make_llm(False)


def make_llm_think():
    """Deep-reasoning LLM — thinking ON / Pro tier. Used by: grade_relevance, rewrite_query, check_faithfulness."""
    return _make_llm(True)


__all__ = ["make_llm_fast", "make_llm_think"]
