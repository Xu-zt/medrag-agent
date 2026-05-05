"""Shared agent utilities."""

import re

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> block and return the final answer."""
    return THINK_RE.sub("", text).strip()


__all__ = ["strip_thinking"]
