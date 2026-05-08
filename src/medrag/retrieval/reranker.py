"""Cross-encoder reranker using bge-reranker-v2-m3.

Device selection via environment variable RERANKER_DEVICE (default: auto):

  RERANKER_DEVICE=auto   → cuda if available, else cpu
  RERANKER_DEVICE=cuda   → force GPU (use_fp16=True, ~200 ms/20 pairs on RTX 4060)
  RERANKER_DEVICE=cpu    → force CPU (~20 s/20 pairs)

GPU reranking frees the LLM from having to do its own slow reranking pass
and is the primary source of end-to-end latency reduction in Stage 3.
"""
from __future__ import annotations

import logging
import os

from medrag.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _resolve_device() -> tuple[str, bool]:
    """Return (device_str, use_fp16) based on RERANKER_DEVICE env var."""
    setting = os.environ.get("RERANKER_DEVICE", "auto").strip().lower()

    if setting == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("[reranker] CUDA available → GPU (fp16=True)")
                return "cuda", True
        except ImportError:
            pass
        logger.info("[reranker] CUDA not available → CPU")
        return "cpu", False

    if setting == "cuda":
        logger.info("[reranker] RERANKER_DEVICE=cuda → GPU (fp16=True)")
        return "cuda", True

    logger.info("[reranker] RERANKER_DEVICE=cpu → CPU")
    return "cpu", False


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        use_fp16: bool | None = None,
        batch_size: int = 8,
    ):
        # Explicit args override env; env is resolved once at construction
        if device is None or use_fp16 is None:
            env_device, env_fp16 = _resolve_device()
            device   = device   if device   is not None else env_device
            use_fp16 = use_fp16 if use_fp16 is not None else env_fp16

        from FlagEmbedding import FlagReranker
        try:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self.batch_size = batch_size
        logger.info("[reranker] loaded %s on %s (fp16=%s)", model_name, device, use_fp16)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [[query, c.text] for c in chunks]
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)

        # Sort descending by reranker score, new objects avoid mutating candidates
        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                score=float(s),
                payload=c.payload,
            )
            for s, c in ranked[:top_k]
        ]


__all__ = ["BGEReranker"]
