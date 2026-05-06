"""
Pydantic data models for the VeritasMed API.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Shared ──────────────────────────────────────────────────────────────────

class ChunkOut(BaseModel):
    """A single retrieved chunk, ready to send to the frontend."""

    chunk_id: str = Field(description="e.g. 'pubmed:12345:0' or 'pmc:doc196:3'")
    citation: str = Field(description="e.g. 'PMID:12345' or 'PMC:doc196'")
    source: str = Field(description="'pubmed' or 'pmc'")
    doc_id: str
    title: str
    section: str | None = None
    pmid: str | None = None
    chunk_idx: int
    total_chunks: int
    text: str
    score: float | None = None
    highlight_ranges: list[tuple[int, int]] = Field(default_factory=list)
    external_url: str = ""


# ── WebSocket event stream ───────────────────────────────────────────────────

class AgentEvent(BaseModel):
    event: Literal[
        "node_start",
        "node_end",
        "chunk_retrieved",
        "answer_token",
        "done",
        "error",
    ]
    node: str | None = None
    data: dict = Field(default_factory=dict)


# ── WS request ──────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    thread_id: str = "default"
    pipeline: Literal["p2", "p3"] = "p2"


# ── Full answer (inside "done" event data) ──────────────────────────────────

class AnswerOut(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    faithful: bool
    faithfulness_issues: str
    iterations: int
    regen_count: int
    rewritten_queries: list[str]
    chunks: list[ChunkOut]
    thread_id: str
    latency_ms: float


# ── /api/search ──────────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    query: str
    pipeline: str
    latency_ms: float
    chunks: list[ChunkOut]


# ── /api/document/{citation} ─────────────────────────────────────────────────

class DocumentChunkSlim(BaseModel):
    chunk_id: str
    chunk_idx: int
    section: str | None = None
    text: str


class DocumentResponse(BaseModel):
    citation: str
    source: str
    doc_id: str
    title: str
    pmid: str | None
    external_url: str
    total_chunks: int
    chunks: list[DocumentChunkSlim]


# ── /api/chunk/{chunk_id} ────────────────────────────────────────────────────

class ChunkSlim(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None


class ChunkContextResponse(BaseModel):
    chunk: ChunkSlim
    prev_chunk: ChunkSlim | None = None
    next_chunk: ChunkSlim | None = None
    document: dict  # {title, citation, external_url}


# ── /api/history/{thread_id} ─────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    query: str
    answer: str
    citations: list[str]
    timestamp: str


class HistoryResponse(BaseModel):
    thread_id: str
    turns: list[HistoryTurn]
    summary: str


# ── /api/corpus/stats ────────────────────────────────────────────────────────

class CorpusStats(BaseModel):
    total_chunks: int
    pubmed_chunks: int
    pmc_chunks: int
    collection: str
    embedding_model: str


# ── /api/health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str
