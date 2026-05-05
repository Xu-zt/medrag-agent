"""Cross-encoder reranker using bge-reranker-v2-m3 (CPU, Plan B)."""

from __future__ import annotations

from FlagEmbedding import FlagReranker

from medrag.retrieval.retriever import RetrievedChunk


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 8,
    ):
        try:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self.batch_size = batch_size

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

        # Sort descending by reranker score, return new RetrievedChunk objects
        # (new objects avoid mutating the P2 candidate list in place)
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
