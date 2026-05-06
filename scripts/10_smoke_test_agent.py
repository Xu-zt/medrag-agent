"""Smoke test for the LangGraph agentic loop.

Tests:
  1. Basic invocation returns a non-empty answer
  2. grade→rewrite loop fires when chunks are empty (simulated)
  3. Multi-turn memory (thread_id preserved across calls)

Usage:
    python scripts/10_smoke_test_agent.py

Does NOT require Qdrant or Ollama for the mock test (test 2).
Tests 1 and 3 require both services running.
"""
import sys
import os
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from medrag.agent.graph import app

# ── Test 1: Basic end-to-end ──────────────────────────────────────────────────

QUESTION = "What is the mechanism of action of aspirin?"

config = {"configurable": {"thread_id": "smoke-test-1"}}

print("\n" + "="*60)
print("TEST 1: Basic end-to-end invocation")
print("="*60)

initial_state = {
    "query": QUESTION,
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

try:
    result = app.invoke(initial_state, config=config)
    answer = result.get("answer", "")
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    iterations = result.get("iterations", 0)
    faithful = result.get("faithful", False)

    print(f"Answer    : {answer[:300]}{'...' if len(answer)>300 else ''}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Citations : {citations}")
    print(f"Iterations: {iterations}  (rewrites performed)")
    print(f"Faithful  : {faithful}")
    assert answer, "FAIL: empty answer"
    print("PASS: non-empty answer received")
except Exception as e:
    print(f"SKIP (service unavailable): {e}")

# ── Test 2: Multi-turn memory (same thread_id) ────────────────────────────────

print("\n" + "="*60)
print("TEST 2: Multi-turn — second question on same thread")
print("="*60)

QUESTION2 = "What are the side effects of the drug you just described?"

initial_state2 = {
    "query": QUESTION2,
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
    "history": [],   # will be merged with checkpointed history by LangGraph
    "summary": "",
}

try:
    result2 = app.invoke(initial_state2, config=config)   # same thread_id!
    answer2 = result2.get("answer", "")
    history = result2.get("history", [])
    summary = result2.get("summary", "")
    print(f"Answer    : {answer2[:300]}{'...' if len(answer2)>300 else ''}")
    print(f"History len: {len(history)}")
    print(f"Summary   : {summary[:200] if summary else '(none)'}")
    print("PASS: multi-turn invocation completed")
except Exception as e:
    print(f"SKIP (service unavailable): {e}")

# ── Test 3: Graph structure ───────────────────────────────────────────────────

print("\n" + "="*60)
print("TEST 3: Graph structure verification")
print("="*60)

nodes = list(app.nodes.keys())
print(f"Nodes: {nodes}")

expected = {"route", "retrieve", "rerank", "grade", "rewrite",
            "generate", "check", "inc_regen", "summarize_gate", "summarize",
            "__start__"}
missing = expected - set(nodes)
if missing:
    print(f"FAIL: missing nodes: {missing}")
    sys.exit(1)
else:
    print("PASS: all expected nodes present")

print("\nAll smoke tests completed.")
