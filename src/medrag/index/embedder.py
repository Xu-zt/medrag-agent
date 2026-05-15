"""BGE-M3 dense embedder via sentence_transformers.

Avoids FlagEmbedding entirely — that library's decoder-only reranker
import chain crashes on Windows (STATUS_ACCESS_VIOLATION / exit code 5).
sentence_transformers loads the same BAAI/bge-m3 weights and produces
identical 1024-dim normalized dense vectors.

Sparse (SPLADE) weights are not produced; callers that request
return_sparse=True get empty dicts, and HybridRetriever falls back
to dense-only retrieval automatically.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class BGEM3Embedder:
    def __init__(self, device: Literal["cuda", "cpu"] = "cpu", use_fp16: bool = False):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-m3", device=device)
        if use_fp16 and device == "cuda":
            self._model = self._model.half()
        self._device = device

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
        return_sparse: bool = False,
    ) -> dict:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = {"dense": np.array(vecs, dtype=np.float32)}
        if return_sparse:
            result["sparse"] = [{} for _ in texts]
        return result


__all__ = ["BGEM3Embedder"]
