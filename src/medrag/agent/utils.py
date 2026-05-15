"""Shared agent utilities."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> block and return the final answer."""
    return THINK_RE.sub("", text).strip()


# ── Citation-grounded generation helpers ──────────────────────────────────────

def validate_citations(
    claims: list[dict],
    retrieved_chunks: list,
) -> list[dict]:
    """Filter claims whose citations are all present in the retrieved chunks.

    A claim is valid iff every cite key it declares matches the citation of
    at least one retrieved chunk.  Claims with empty cite lists are dropped
    (they carry no evidence).

    Args:
        claims: List of {"text": str, "cite": [str, ...]} dicts from the LLM.
        retrieved_chunks: RetrievedChunk objects (have a .citation property)
                          or plain dicts with a "citation" key.

    Returns:
        Filtered list containing only evidence-backed claims.
    """
    # Build set of valid citation keys from current retrieval context
    valid: set[str] = set()
    for c in retrieved_chunks:
        if hasattr(c, "citation"):
            valid.add(c.citation)
        elif isinstance(c, dict):
            cit = c.get("citation") or _build_citation(c)
            if cit:
                valid.add(cit)

    filtered: list[dict] = []
    for claim in claims:
        cite_keys: list[str] = claim.get("cite", [])
        if not cite_keys:
            logger.debug("[validate_citations] dropped claim with no cite: %s", claim.get("text", "")[:60])
            continue
        # Strip brackets the LLM may add (e.g. "[PMC:doc205]" → "PMC:doc205")
        cite_keys = [k.strip("[]") for k in cite_keys]
        # All cited keys must be in the valid set
        invalid_keys = [k for k in cite_keys if k not in valid]
        if invalid_keys:
            logger.warning(
                "[validate_citations] dropped claim — invalid cites %s (valid: %s)",
                invalid_keys, sorted(valid),
            )
            continue
        filtered.append(claim)

    logger.info(
        "[validate_citations] %d/%d claims passed (valid cites: %s)",
        len(filtered), len(claims), sorted(valid),
    )
    return filtered


def build_answer_from_claims(claims: list[dict]) -> tuple[str, list[str]]:
    """Reconstruct a readable answer string and citation list from validated claims.

    Returns:
        (answer_text, citations_list)
    """
    if not claims:
        return (
            "The retrieved documents do not contain sufficient cited evidence "
            "to answer this question.",
            [],
        )

    parts: list[str] = []
    all_cites: list[str] = []
    seen_cites: set[str] = set()

    for claim in claims:
        text = claim["text"].rstrip(" .")
        cites: list[str] = claim.get("cite", [])
        inline = " ".join(f"[{c}]" for c in cites)
        parts.append(f"{text} {inline}.")
        for c in cites:
            if c not in seen_cites:
                all_cites.append(c)
                seen_cites.add(c)

    return " ".join(parts), all_cites


def _build_citation(payload: dict) -> str:
    """Reconstruct citation key from a raw payload dict."""
    source = payload.get("source", "")
    pmid = payload.get("pmid")
    doc_id = payload.get("doc_id", "")
    if source == "pubmed" and pmid:
        return f"PMID:{pmid}"
    if source == "pmc":
        return f"PMC:{doc_id}"
    return ""


__all__ = [
    "strip_thinking",
    "validate_citations",
    "build_answer_from_claims",
]
