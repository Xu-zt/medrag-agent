"""
WebSocket /api/ask — streams the full agentic reasoning loop as events.

Uses asyncio.to_thread + Queue to safely bridge LangGraph's synchronous
app.stream() into the async WebSocket handler, avoiding issues with
SqliteSaver not being async-compatible.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from medrag.agent.graph import app as langgraph_app
from medrag.api._helpers import payload_to_chunk
from medrag.api.models import AgentEvent, AnswerOut, AskRequest, ChunkOut

router = APIRouter()
logger = logging.getLogger(__name__)

_SENTINEL = object()  # marks end-of-stream in the queue


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
    """Send JSON to WebSocket, silently ignore closed-connection errors."""
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


def _node_event(node_name: str, output: dict) -> tuple[AgentEvent | None, list[AgentEvent]]:
    """
    Convert a LangGraph stream chunk {node_name: output_dict} into
    AgentEvents.  Returns (node_end_event, extra_events).
    """
    extra: list[AgentEvent] = []

    if node_name in ("__start__", "__end__", "summarize_gate"):
        return None, extra

    # ── chunk_retrieved events (one per chunk) ──────────────────────────
    if node_name in ("retrieve", "rerank"):
        chunks = output.get("retrieved_chunks", [])
        for c in chunks:
            if hasattr(c, "payload"):
                co = payload_to_chunk(c.payload, score=getattr(c, "score", None))
            elif isinstance(c, dict):
                co = payload_to_chunk(c, score=c.get("score"))
            else:
                continue
            extra.append(AgentEvent(
                event="chunk_retrieved",
                node=node_name,
                data={
                    "chunk_id": co.chunk_id,
                    "citation": co.citation,
                    "title": co.title,
                    "score": co.score,
                    "text_snippet": co.text[:200],
                    "source": co.source,
                    "external_url": co.external_url,
                },
            ))
        node_data: dict[str, Any] = {"count": len(chunks)}

    elif node_name == "grade":
        node_data = {
            "relevance_score": output.get("relevance_score", 0.0),
            "relevant": output.get("relevant", False),
            "reason": output.get("grade_reason", ""),
            "rewrite_hint": output.get("rewrite_hint", ""),
        }

    elif node_name == "rewrite":
        rqs = output.get("rewritten_queries", [])
        node_data = {
            "new_query": rqs[-1] if rqs else "",
            "rewritten_queries": rqs,
        }

    elif node_name == "generate":
        node_data = {"answer_preview": output.get("answer", "")[:120]}

    elif node_name == "check":
        node_data = {
            "faithful": output.get("faithful", False),
            "issues": output.get("faithfulness_issues", ""),
            "confidence": output.get("confidence", 0.0),
        }

    elif node_name == "route":
        node_data = {"route": output.get("route", "")}

    else:
        node_data = {}

    return AgentEvent(event="node_end", node=node_name, data=node_data), extra


@router.websocket("/api/ask")
async def ask_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    t_start = time.perf_counter()
    logger.info("WS /api/ask accepted")

    # ── Parse request ────────────────────────────────────────────────────
    try:
        raw = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        req = AskRequest(**raw)
    except Exception as exc:
        logger.exception("WS receive/parse error")
        await _send_safe(websocket, AgentEvent(
            event="error", data={"message": str(exc)}
        ).model_dump())
        await websocket.close()
        return

    logger.info("WS query=%r thread=%s pipeline=%s", req.query, req.thread_id, req.pipeline)

    config: dict = {"configurable": {"thread_id": req.thread_id}}
    initial_state = _build_initial_state(req.query)

    # ── Queue bridge: LangGraph (sync thread) → WebSocket (async) ────────
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _stream_worker() -> None:
        """Run app.stream() in a thread pool; push chunks into the queue."""
        try:
            prev_node: str | None = None
            for chunk in langgraph_app.stream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                # chunk = {node_name: state_update_dict}
                for node_name, output in chunk.items():
                    if not isinstance(output, dict):
                        output = {}
                    # Send node_start first
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("node_start", node_name, {}),
                    )
                    # Send node_end + any extra events
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("node_output", node_name, output),
                    )
                    prev_node = node_name
        except Exception as exc:
            logger.exception("LangGraph stream error")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc), {}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL, None, None))

    # Start the stream worker in a thread
    stream_task = asyncio.ensure_future(asyncio.to_thread(_stream_worker))

    # ── Consume queue and push events to WebSocket ────────────────────────
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
                await _send_safe(websocket, AgentEvent(
                    event="error", data={"message": name}
                ).model_dump())
                break

            if kind == "node_start":
                if name in ("__start__", "__end__", "summarize_gate", "inc_regen"):
                    continue
                await _send_safe(websocket, AgentEvent(
                    event="node_start", node=name, data={}
                ).model_dump())

            elif kind == "node_output":
                if name in ("__start__", "__end__", "summarize_gate"):
                    continue
                node_end_ev, extra_evs = _node_event(name, data)
                # Send extra events first (chunk_retrieved)
                for ev in extra_evs:
                    await _send_safe(websocket, ev.model_dump())
                # Then node_end
                if node_end_ev is not None:
                    await _send_safe(websocket, node_end_ev.model_dump())

    except WebSocketDisconnect:
        logger.info("WS client disconnected during stream")
        stream_task.cancel()
        return
    except Exception as exc:
        logger.exception("WS consumer error")
        await _send_safe(websocket, AgentEvent(
            event="error", data={"message": str(exc)}
        ).model_dump())

    # ── Final "done" event ───────────────────────────────────────────────
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

    await _send_safe(websocket, AgentEvent(
        event="done", node=None, data=answer_out.model_dump()
    ).model_dump())
    logger.info("WS done in %.0fms", latency)

    try:
        await websocket.close()
    except Exception:
        pass
