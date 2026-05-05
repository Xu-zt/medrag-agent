"""BGE-M3 dense embedder (Plan B: CPU by default)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from FlagEmbedding import BGEM3FlagModel


class BGEM3Embedder:
    def __init__(self, device: Literal["cuda", "cpu"] = "cpu", use_fp16: bool = False):
        try:
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
        return_sparse: bool = False,
    ) -> dict:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
            max_length=512,
        )
        result = {"dense": np.array(out["dense_vecs"], dtype=np.float32)}
        if return_sparse:
            result["sparse"] = out["lexical_weights"]
        return result


__all__ = ["BGEM3Embedder"]
