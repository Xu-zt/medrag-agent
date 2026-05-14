"""
GET /api/search — standalone hybrid retrieval (no agent loop).
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from fastapi import APIRouter, Query

from medrag.api._helpers import compute_highlights, payload_to_chunk
from medrag.api.models import ChunkOut, SearchResponse
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker

router = APIRouter()


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)
    embedder = BGEM3Embedder(device="cpu")
    return HybridRetriever(qdrant, embedder, candidate_k=20)


@lru_cache(maxsize=1)
def _reranker() -> BGEReranker:
    return BGEReranker(device=os.environ.get("RERANKER_DEVICE", "cpu"))


@router.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(default=5, ge=1, le=20),
    pipeline: str = Query(default="p2", pattern="^p[123]$"),
    highlight: bool = Query(default=True),
) -> SearchResponse:
    t0 = time.perf_counter()
    retriever = _retriever()

    # Retrieve more candidates for reranking (p3) or direct (p2)
    candidate_k = 20 if pipeline == "p3" else k
    raw_chunks = retriever.retrieve(q, k=candidate_k)

    if pipeline == "p3":
        reranker = _reranker()
        raw_chunks = reranker.rerank(q, raw_chunks, top_k=k)
    else:
        raw_chunks = raw_chunks[:k]

    chunks_out: list[ChunkOut] = []
    for rc in raw_chunks:
        payload = rc.payload if hasattr(rc, "payload") else rc
        score = getattr(rc, "score", None)
        co = payload_to_chunk(payload, score=score)
        if highlight:
            co.highlight_ranges = compute_highlights(co.text, q)
        chunks_out.append(co)

    latency = round((time.perf_counter() - t0) * 1000, 1)
    return SearchResponse(query=q, pipeline=pipeline, latency_ms=latency, chunks=chunks_out)
