"""
WebSocket /api/ask — streams the full agentic reasoning loop as events.

Uses asyncio.to_thread + Queue to safely bridge LangGraph's synchronous
app.stream() into the async WebSocket handler.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from medrag.agent.graph import app as langgraph_app
from medrag.api._helpers import payload_to_chunk
from medrag.api.models import (
    AskRequest,
    AnswerOut,
    ChunkOut,
    ChunkRetrievedData,
    ChunkRetrievedEvent,
    DoneEvent,
    ErrorData,
    ErrorEvent,
    NodeEndData,
    NodeEndEvent,
    NodeStartEvent,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_SENTINEL = object()


def _build_initial_state(query: str) -> dict:
    return {
        "query": query,
        "original_query": "",
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "relevant": False,
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


async def _send_safe(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


def _chunks_from_state(state: dict) -> list[ChunkOut]:
    chunks_out: list[ChunkOut] = []
    for c in state.get("retrieved_chunks", []):
        if hasattr(c, "payload"):
            chunks_out.append(payload_to_chunk(c.payload, score=getattr(c, "score", None)))
        elif isinstance(c, dict):
            chunks_out.append(payload_to_chunk(c, score=c.get("score")))
    return chunks_out


def _node_event(
    node_name: str, output: dict
) -> tuple[NodeEndEvent | None, list[ChunkRetrievedEvent]]:
    extras: list[ChunkRetrievedEvent] = []

    if node_name in ("__start__", "__end__", "summarize_gate"):
        return None, extras

    if node_name in ("retrieve", "rerank"):
        chunks = output.get("retrieved_chunks", [])
        for c in chunks:
            if hasattr(c, "payload"):
                co = payload_to_chunk(c.payload, score=getattr(c, "score", None))
            elif isinstance(c, dict):
                co = payload_to_chunk(c, score=c.get("score"))
            else:
                continue
            extras.append(ChunkRetrievedEvent(
                node=node_name,
                data=ChunkRetrievedData(
                    chunk_id=co.chunk_id,
                    citation=co.citation,
                    title=co.title,
                    score=co.score,
                    text_snippet=co.text[:200],
                    source=co.source,
                    external_url=co.external_url,
                ),
            ))
        data = NodeEndData(count=len(chunks))

    elif node_name == "grade":
        data = NodeEndData(
            relevance_score=output.get("relevance_score", 0.0),
            relevant=output.get("relevant", False),
            reason=output.get("grade_reason", ""),
            rewrite_hint=output.get("rewrite_hint", ""),
        )

    elif node_name == "rewrite":
        rqs = output.get("rewritten_queries", [])
        data = NodeEndData(
            new_query=rqs[-1] if rqs else "",
            rewritten_queries=rqs,
        )

    elif node_name == "generate":
        data = NodeEndData(answer_preview=output.get("answer", "")[:120])

    elif node_name == "check":
        data = NodeEndData(
            faithful=output.get("faithful", False),
            issues=output.get("faithfulness_issues", ""),
            confidence=output.get("confidence", 0.0),
        )

    elif node_name == "route":
        data = NodeEndData(route=output.get("route", ""))

    else:
        data = NodeEndData()

    return NodeEndEvent(node=node_name, data=data), extras


@router.websocket("/api/ask")
async def ask_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    t_start = time.perf_counter()
    logger.info("WS /api/ask accepted")

    try:
        raw = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        req = AskRequest(**raw)
    except Exception as exc:
        logger.exception("WS receive/parse error")
        await _send_safe(websocket, ErrorEvent(data=ErrorData(message=str(exc))).model_dump())
        await websocket.close()
        return

    logger.info("WS query=%r thread=%s pipeline=%s", req.query, req.thread_id, req.pipeline)

    config: dict = {"configurable": {"thread_id": req.thread_id}}
    initial_state = _build_initial_state(req.query)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _stream_worker() -> None:
        try:
            for chunk in langgraph_app.stream(
                initial_state, config=config, stream_mode="updates"
            ):
                for node_name, output in chunk.items():
                    if not isinstance(output, dict):
                        output = {}
                    loop.call_soon_threadsafe(queue.put_nowait, ("node_start", node_name, {}))
                    loop.call_soon_threadsafe(queue.put_nowait, ("node_output", node_name, output))
        except Exception as exc:
            logger.exception("LangGraph stream error")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc), {}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL, None, None))

    stream_task = asyncio.ensure_future(asyncio.to_thread(_stream_worker))

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                logger.warning("WS queue timeout after 300s")
                break

            kind, name, data = item
            if kind is _SENTINEL:
                break

            if kind == "error":
                await _send_safe(
                    websocket,
                    ErrorEvent(data=ErrorData(message=name)).model_dump(),
                )
                break

            if kind == "node_start":
                if name in ("__start__", "__end__", "summarize_gate", "inc_regen"):
                    continue
                await _send_safe(websocket, NodeStartEvent(node=name).model_dump())

            elif kind == "node_output":
                if name in ("__start__", "__end__", "summarize_gate"):
                    continue
                node_end_ev, extras = _node_event(name, data)
                for ev in extras:
                    await _send_safe(websocket, ev.model_dump())
                if node_end_ev is not None:
                    await _send_safe(websocket, node_end_ev.model_dump())

    except WebSocketDisconnect:
        logger.info("WS client disconnected during stream")
        stream_task.cancel()
        return
    except Exception as exc:
        logger.exception("WS consumer error")
        await _send_safe(websocket, ErrorEvent(data=ErrorData(message=str(exc))).model_dump())

    try:
        snapshot = langgraph_app.get_state(config)
        final = snapshot.values if snapshot else {}
    except Exception:
        final = {}

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

    await _send_safe(websocket, DoneEvent(data=answer_out).model_dump())
    logger.info("WS done in %.0fms", latency)

    try:
        await websocket.close()
    except Exception:
        pass
