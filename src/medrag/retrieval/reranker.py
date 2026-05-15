"""Cross-encoder reranker using bge-reranker-v2-m3 via sentence_transformers.

Device selection via RERANKER_DEVICE env var (default: auto):
  auto  → cuda if available, else cpu
  cuda  → force GPU (fp16, ~200 ms/20 pairs on RTX 4060)
  cpu   → force CPU (~20 s/20 pairs)
"""
from __future__ import annotations

import logging
import os

from medrag.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _resolve_device() -> tuple[str, bool]:
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
        return "cuda", True
    return "cpu", False


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        use_fp16: bool | None = None,
        batch_size: int = 8,
    ):
        if device is None or use_fp16 is None:
            env_device, env_fp16 = _resolve_device()
            device   = device   if device   is not None else env_device
            use_fp16 = use_fp16 if use_fp16 is not None else env_fp16

        from sentence_transformers import CrossEncoder
        import torch
        dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
        self._model = CrossEncoder(
            model_name,
            device=device,
            automodel_args={"torch_dtype": dtype},
        )
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
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = sorted(zip(scores, chunks), key=lambda x: -float(x[0]))
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
