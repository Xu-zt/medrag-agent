"""
GET /api/corpus/stats — Qdrant collection statistics.
GET /api/health       — service health check.
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter

from medrag.api._helpers import get_qdrant
from medrag.api.models import CorpusStats, HealthResponse

router = APIRouter()

from medrag.config import COLLECTION_NAME

_COLLECTION = COLLECTION_NAME
_EMBEDDING_MODEL = "BAAI/bge-m3"


@router.get("/api/corpus/stats", response_model=CorpusStats)
async def corpus_stats() -> CorpusStats:
    q = get_qdrant()
    info = q.get_collection(_COLLECTION)
    total = info.points_count or 0

    # Count PubMed vs PMC using facet via scroll (approximate)
    pubmed_count = 0
    pmc_count = 0
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        res = q.count(
            collection_name=_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="pubmed"))]
            ),
        )
        pubmed_count = res.count
        res2 = q.count(
            collection_name=_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="pmc"))]
            ),
        )
        pmc_count = res2.count
    except Exception:
        pubmed_count = total
        pmc_count = 0

    return CorpusStats(
        total_chunks=total,
        pubmed_chunks=pubmed_count,
        pmc_chunks=pmc_count,
        collection=_COLLECTION,
        embedding_model=_EMBEDDING_MODEL,
    )


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Check Qdrant
    qdrant_status = "disconnected"
    try:
        q = get_qdrant()
        q.get_collection(_COLLECTION)
        qdrant_status = "connected"
    except Exception:
        pass

    # Check LLM API (MiMo / OpenAI-compatible)
    llm_status = "disconnected"
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
    if not base_url:
        llm_status = "not_configured"
    else:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "sk-none")
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    base_url.rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if r.status_code in (200, 401):
                    llm_status = "connected"
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        llm=llm_status,
    )
