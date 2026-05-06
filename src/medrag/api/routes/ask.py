"""
WebSocket /api/ask — streams the full agentic reasoning loop as events.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from medrag.agent.graph import app as langgraph_app
from medrag.agent.state import AgentState
from medrag.api._helpers import payload_to_chunk
from medrag.api.models import AgentEvent, AnswerOut, AskRequest, ChunkOut

router = APIRouter()

# Node names that appear in LangGraph event streams
_NODES = {
    "route", "retrieve", "rerank", "grade",
    "rewrite", "generate", "check",
    "increment_regen", "summarize_gate", "summarize",
}


def _build_initial_state(query: str) -> dict:
    return {
        "query": query,
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "grade_reason": "",
        "rewrite_hint": "",
        "iterations": 0,
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "faithful": False,
        "faithfulness_issues": "",
        "regen_count": 0,
        "history": [],
        "summary": "",
    }


def _chunks_from_state(state: dict) -> list[ChunkOut]:
    """Convert retrieved_chunks (RetrievedChunk objects or dicts) to ChunkOut."""
    chunks_out: list[ChunkOut] = []
    for c in state.get("retrieved_chunks", []):
        if hasattr(c, "payload"):
            chunks_out.append(payload_to_chunk(c.payload, score=getattr(c, "score", None)))
        elif isinstance(c, dict):
            chunks_out.append(payload_to_chunk(c, score=c.get("score")))
    return chunks_out


@router.websocket("/api/ask")
async def ask_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    t_start = time.perf_counter()

    try:
        raw = await websocket.receive_json()
        req = AskRequest(**raw)
    except Exception as exc:
        await websocket.send_json(
            AgentEvent(event="error", data={"message": str(exc)}).model_dump()
        )
        await websocket.close()
        return

    config: dict = {"configurable": {"thread_id": req.thread_id}}
    initial_state = _build_initial_state(req.query)

    try:
        async for event in langgraph_app.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind: str = event.get("event", "")
            name: str = event.get("name", "")
            data: dict = event.get("data", {})

            # ── node started ──────────────────────────────────────────────
            if kind == "on_chain_start" and name in _NODES:
                payload: dict[str, Any] = {}
                inp = data.get("input") or {}
                if name == "retrieve":
                    payload["query"] = inp.get("query", req.query)
                elif name == "rewrite":
                    payload["iteration"] = inp.get("iterations", 0)
                elif name == "generate":
                    payload["query"] = inp.get("query", req.query)

                await websocket.send_json(
                    AgentEvent(event="node_start", node=name, data=payload).model_dump()
                )

            # ── node finished ─────────────────────────────────────────────
            elif kind == "on_chain_end" and name in _NODES:
                out: dict = data.get("output") or {}
                if not isinstance(out, dict):
                    out = {}

                node_data: dict[str, Any] = {}

                if name == "retrieve":
                    # Push each chunk as a separate chunk_retrieved event
                    for chunk in out.get("retrieved_chunks", []):
                        if hasattr(chunk, "payload"):
                            co = payload_to_chunk(chunk.payload, score=getattr(chunk, "score", None))
                        elif isinstance(chunk, dict):
                            co = payload_to_chunk(chunk, score=chunk.get("score"))
                        else:
                            continue
                        await websocket.send_json(
                            AgentEvent(
                                event="chunk_retrieved",
                                node="retrieve",
                                data={
                                    "chunk_id": co.chunk_id,
                                    "citation": co.citation,
                                    "title": co.title,
                                    "score": co.score,
                                    "text_snippet": co.text[:200],
                                    "source": co.source,
                                    "external_url": co.external_url,
                                },
                            ).model_dump()
                        )
                    node_data["count"] = len(out.get("retrieved_chunks", []))

                elif name == "grade":
                    node_data = {
                        "relevance_score": out.get("relevance_score", 0.0),
                        "relevant": out.get("relevance_score", 0.0) >= 0.6,
                        "reason": out.get("grade_reason", ""),
                        "rewrite_hint": out.get("rewrite_hint", ""),
                    }

                elif name == "rewrite":
                    rqs = out.get("rewritten_queries", [])
                    node_data = {
                        "new_query": rqs[-1] if rqs else "",
                        "rewritten_queries": rqs,
                    }

                elif name == "generate":
                    node_data = {"answer_preview": (out.get("answer", ""))[:120]}

                elif name == "check":
                    node_data = {
                        "faithful": out.get("faithful", False),
                        "issues": out.get("faithfulness_issues", ""),
                        "confidence": out.get("confidence", 0.0),
                    }

                await websocket.send_json(
                    AgentEvent(event="node_end", node=name, data=node_data).model_dump()
                )

        # ── stream finished — fetch final state via checkpointer ──────────
        final = langgraph_app.get_state(config).values
        latency = round((time.perf_counter() - t_start) * 1000, 1)

        answer_out = AnswerOut(
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            confidence=final.get("confidence", 0.0),
            faithful=final.get("faithful", False),
            faithfulness_issues=final.get("faithfulness_issues", ""),
            iterations=final.get("iterations", 0),
            regen_count=final.get("regen_count", 0),
            rewritten_queries=final.get("rewritten_queries", []),
            chunks=_chunks_from_state(final),
            thread_id=req.thread_id,
            latency_ms=latency,
        )

        await websocket.send_json(
            AgentEvent(
                event="done",
                node=None,
                data=answer_out.model_dump(),
            ).model_dump()
        )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json(
                AgentEvent(event="error", data={"message": str(exc)}).model_dump()
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
