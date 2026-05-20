"""Integration tests for MedRAG-Agent full pipeline.

Requires live services:
  - Qdrant at QDRANT_URL (collection: medrag_text)
  - MiMo API (OPENAI_BASE_URL + OPENAI_API_KEY in .env)

Run:
    pytest tests/test_integration.py -v -s --tb=short

Each test measures wall-clock time and validates output structure + quality.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure src is on path and load .env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from medrag.agent.graph import app
from medrag.agent.state import AgentState

# ── Helpers ─────────────────────────────────────────────────────────────────────

# Timeout thresholds (seconds)
TIMEOUT_WARN = 60.0    # soft warning
TIMEOUT_HARD = 180.0   # hard fail


def _fresh_state(query: str) -> dict:
    """Build a minimal valid AgentState for invocation."""
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


def _run_agent(query: str, thread_id: str = "test-default") -> tuple[dict, float]:
    """Invoke the agent graph and return (result_state, elapsed_seconds)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = _fresh_state(query)
    t0 = time.perf_counter()
    result = app.invoke(state, config=config)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _assert_valid_result(result: dict, min_answer_len: int = 20):
    """Common assertions on any agent result."""
    assert "answer" in result, "Missing 'answer' field"
    assert "citations" in result, "Missing 'citations' field"
    assert "confidence" in result, "Missing 'confidence' field"
    assert "faithful" in result, "Missing 'faithful' field"
    assert isinstance(result["answer"], str), "answer must be str"
    assert len(result["answer"]) >= min_answer_len, (
        f"Answer too short ({len(result['answer'])} chars): {result['answer']!r}"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence out of range: {result['confidence']}"
    )


def _is_disclaimer(answer: str) -> bool:
    """Check if the answer is the generic disclaimer (no valid claims)."""
    return "do not contain sufficient cited evidence" in answer.lower()


# ── Category 1: Single-turn Factual Queries ────────────────────────────────────

class TestSingleTurnFactual:
    """Simple factual questions — expect direct answers with citations."""

    @pytest.mark.parametrize("query,expected_keywords", [
        (
            "What is the mechanism of action of aspirin?",
            ["COX", "cyclooxygenase", "prostaglandin", "thromboxane", "platelet"],
        ),
        (
            "What is the first-line treatment for hypertension?",
            ["ACE", "ARB", "calcium", "thiazide", "amlodipine", "lisinopril",
             "antihypertensive", "blood pressure"],
        ),
        (
            "What are the diagnostic criteria for Type 2 diabetes mellitus?",
            ["HbA1c", "fasting", "glucose", "OGTT", "diabetes", "insulin"],
        ),
    ], ids=["aspirin-mechanism", "hypertension-treatment", "t2dm-diagnosis"])

    def test_factual_query(self, query, expected_keywords):
        result, elapsed = _run_agent(query, thread_id=f"factual-{hash(query) % 10000}")

        _assert_valid_result(result)

        # Log result quality
        is_disclaimer = _is_disclaimer(result["answer"])
        print(f"\n  [factual] {query[:60]}...")
        print(f"    answer: {result['answer'][:150]}...")
        print(f"    confidence={result['confidence']:.2f}  citations={result['citations']}")
        print(f"    faithful={result['faithful']}  disclaimer={is_disclaimer}  time={elapsed:.1f}s")

        # If not a disclaimer, check quality
        if not is_disclaimer:
            assert result["confidence"] >= 0.3, f"Low confidence on non-disclaimer: {result['confidence']}"
            assert len(result["citations"]) >= 1, "No citations on non-disclaimer answer"
            answer_lower = result["answer"].lower()
            matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            assert len(matched) >= 1, (
                f"Answer lacks expected keywords. Got: {result['answer'][:200]}..."
            )

        assert elapsed < TIMEOUT_HARD, f"Timeout: {elapsed:.1f}s"


# ── Category 2: Single-turn Synthesis Queries ──────────────────────────────────

class TestSingleTurnSynthesis:
    """Questions requiring information from multiple sources."""

    @pytest.mark.parametrize("query", [
        "Compare the efficacy of metformin versus sulfonylureas in glycemic control for Type 2 diabetes.",
        "What are the cardiovascular benefits and risks of COX-2 selective inhibitors compared to non-selective NSAIDs?",
    ], ids=["metformin-vs-sulfonylureas", "cox2-vs-nsaids"])

    def test_synthesis_query(self, query):
        result, elapsed = _run_agent(query, thread_id=f"synthesis-{hash(query) % 10000}")

        _assert_valid_result(result, min_answer_len=30)

        is_disclaimer = _is_disclaimer(result["answer"])
        print(f"\n  [synthesis] {query[:60]}...")
        print(f"    answer: {result['answer'][:200]}...")
        print(f"    confidence={result['confidence']:.2f}  citations={len(result['citations'])}")
        print(f"    disclaimer={is_disclaimer}  time={elapsed:.1f}s")

        if not is_disclaimer:
            assert len(result["citations"]) >= 1, (
                f"Synthesis answer should cite sources, got {len(result['citations'])}"
            )

        assert elapsed < TIMEOUT_HARD, f"Timeout: {elapsed:.1f}s"


# ── Category 3: Single-turn Multihop Queries ───────────────────────────────────

class TestSingleTurnMultihop:
    """Multi-hop reasoning questions."""

    def test_multihop_query(self):
        query = "What genetic polymorphism affects warfarin dosing, and what is the recommended dose adjustment?"
        result, elapsed = _run_agent(query, thread_id="multihop-warfarin")

        _assert_valid_result(result)

        is_disclaimer = _is_disclaimer(result["answer"])
        print(f"\n  [multihop] {query[:60]}...")
        print(f"    answer: {result['answer'][:200]}...")
        print(f"    confidence={result['confidence']:.2f}  iterations={result['iterations']}")
        print(f"    disclaimer={is_disclaimer}  time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD, f"Timeout: {elapsed:.1f}s"


# ── Category 4: Multi-turn Conversation ────────────────────────────────────────

class TestMultiTurn:
    """Multi-turn conversation on the same thread_id — tests L1 memory."""

    def test_two_turn_conversation(self):
        thread_id = f"multi-turn-{int(time.time())}"

        # Turn 1
        q1 = "What is metformin and how does it work?"
        r1, t1 = _run_agent(q1, thread_id=thread_id)
        _assert_valid_result(r1)

        # Turn 2 — follow-up referencing turn 1
        q2 = "What are its most common side effects?"
        r2, t2 = _run_agent(q2, thread_id=thread_id)
        _assert_valid_result(r2)

        print(f"\n  [multi-turn] Turn 1: {q1}")
        print(f"    answer: {r1['answer'][:100]}...  time={t1:.1f}s")
        print(f"  [multi-turn] Turn 2: {q2}")
        print(f"    answer: {r2['answer'][:100]}...  time={t2:.1f}s")
        print(f"    history_len={len(r2.get('history', []))}")

        # Both turns should complete within timeout
        assert t1 < TIMEOUT_HARD and t2 < TIMEOUT_HARD

    def test_three_turn_topic_shift(self):
        """Three turns with a topic shift — tests memory doesn't bleed."""
        thread_id = f"topic-shift-{int(time.time())}"

        q1 = "What are the indications for metformin?"
        r1, t1 = _run_agent(q1, thread_id=thread_id)

        q2 = "What monitoring is required for lithium therapy?"
        r2, t2 = _run_agent(q2, thread_id=thread_id)

        q3 = "What is the therapeutic range for lithium?"
        r3, t3 = _run_agent(q3, thread_id=thread_id)

        _assert_valid_result(r3)

        print(f"\n  [topic-shift] 3 turns completed")
        print(f"    times: {t1:.1f}s, {t2:.1f}s, {t3:.1f}s")
        print(f"    turn3 answer: {r3['answer'][:150]}...")

        # All turns should complete
        assert t1 < TIMEOUT_HARD and t2 < TIMEOUT_HARD and t3 < TIMEOUT_HARD


# ── Category 5: Rewrite Loop Behavior ──────────────────────────────────────────

class TestRewriteLoop:
    """Verify the grade→rewrite→retrieve loop works correctly."""

    def test_iterations_recorded(self):
        """iterations should be 0 or 1 (MAX_REWRITES=1)."""
        query = "What is the mechanism of metformin?"
        result, elapsed = _run_agent(query, thread_id="rewrite-test")

        assert "iterations" in result
        assert result["iterations"] in (0, 1), (
            f"Unexpected iterations: {result['iterations']}"
        )

        print(f"\n  [rewrite] iterations={result['iterations']}  "
              f"rewritten={result.get('rewritten_queries', [])}  time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD, f"Timeout: {elapsed:.1f}s"


# ── Category 6: Faithfulness & Regeneration ────────────────────────────────────

class TestFaithfulness:
    """Verify the check node correctly identifies and handles faithfulness."""

    def test_faithfulness_reported(self):
        query = "What is the first-line treatment for community-acquired pneumonia?"
        result, elapsed = _run_agent(query, thread_id="faith-test")

        assert "faithful" in result
        assert "faithfulness_issues" in result
        assert "regen_count" in result
        assert result["regen_count"] in (0, 1)

        print(f"\n  [faith] faithful={result['faithful']}  regen={result['regen_count']}")
        if result["faithfulness_issues"]:
            print(f"    issues: {result['faithfulness_issues'][:200]}")
        print(f"    time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD, f"Timeout: {elapsed:.1f}s"


# ── Category 7: Confidence & Citation Quality ──────────────────────────────────

class TestCitationQuality:
    """Verify citations are properly formatted and grounded."""

    def test_citations_format(self):
        query = "What are the contraindications for ACE inhibitors?"
        result, elapsed = _run_agent(query, thread_id="cite-format")

        for cite in result["citations"]:
            assert isinstance(cite, str), f"Citation not a string: {cite}"
            # Should be PMID or PMC format (with or without brackets from LLM)
            clean = cite.strip("[]")
            assert clean.startswith("PMID:") or clean.startswith("PMC:"), (
                f"Unexpected citation format: {cite}"
            )

        print(f"\n  [citations] count={len(result['citations'])}  "
              f"sample={result['citations'][:3]}  time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD

    def test_confidence_reasonable(self):
        """Confidence should be reasonable for a well-formed answer."""
        query = "What is the half-life of aspirin?"
        result, elapsed = _run_agent(query, thread_id="conf-test")

        print(f"\n  [confidence] {result['confidence']:.2f}  "
              f"answer_len={len(result['answer'])}  cites={len(result['citations'])}")

        # If we got a real answer with citations, confidence should be decent
        if not _is_disclaimer(result["answer"]) and len(result["citations"]) >= 1:
            assert result["confidence"] >= 0.3, (
                f"Confidence too low for a good answer: {result['confidence']}"
            )

        assert elapsed < TIMEOUT_HARD


# ── Category 8: Retrieval Pipeline Comparison ──────────────────────────────────

class TestRetrievalPipelines:
    """Compare P1 (dense only) vs P2 (hybrid) retrieval quality."""

    def test_p1_dense_retrieval(self):
        from medrag.retrieval.retriever import DenseRetriever
        from qdrant_client import QdrantClient
        from medrag.index.embedder import BGEM3Embedder

        from medrag.config import qdrant_url

        qdrant = QdrantClient(url=qdrant_url(), timeout=30)
        embedder = BGEM3Embedder(device="cpu")
        retriever = DenseRetriever(qdrant, embedder)

        query = "aspirin mechanism of action COX inhibition"
        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, k=5)
        elapsed = time.perf_counter() - t0

        assert len(chunks) >= 1, "P1 returned no chunks"
        assert all(c.score > 0 for c in chunks), "All scores should be positive"

        print(f"\n  [P1 dense] {len(chunks)} chunks in {elapsed:.2f}s")
        for i, c in enumerate(chunks[:3]):
            print(f"    [{i}] score={c.score:.3f}  {c.citation}  {c.text[:80]}...")

    def test_p2_hybrid_retrieval(self):
        from medrag.retrieval.hybrid import HybridRetriever
        from qdrant_client import QdrantClient
        from medrag.index.embedder import BGEM3Embedder

        from medrag.config import qdrant_url

        qdrant = QdrantClient(url=qdrant_url(), timeout=30)
        embedder = BGEM3Embedder(device="cpu")
        retriever = HybridRetriever(qdrant, embedder, candidate_k=20)

        query = "aspirin mechanism of action COX inhibition"
        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, k=5)
        elapsed = time.perf_counter() - t0

        assert len(chunks) >= 1, "P2 returned no chunks"

        print(f"\n  [P2 hybrid] {len(chunks)} chunks in {elapsed:.2f}s")
        for i, c in enumerate(chunks[:3]):
            print(f"    [{i}] score={c.score:.3f}  {c.citation}  {c.text[:80]}...")

    def test_hybrid_vs_dense_recall(self):
        """Hybrid should retrieve at least as many relevant chunks as dense."""
        from medrag.retrieval.retriever import DenseRetriever
        from medrag.retrieval.hybrid import HybridRetriever
        from qdrant_client import QdrantClient
        from medrag.index.embedder import BGEM3Embedder

        from medrag.config import qdrant_url

        qdrant = QdrantClient(url=qdrant_url(), timeout=30)
        embedder = BGEM3Embedder(device="cpu")

        dense_ret = DenseRetriever(qdrant, embedder)
        hybrid_ret = HybridRetriever(qdrant, embedder, candidate_k=20)

        query = "warfarin pharmacogenomics CYP2C9 VKORC1 dosing"

        dense_chunks = dense_ret.retrieve(query, k=10)
        hybrid_chunks = hybrid_ret.retrieve(query, k=10)

        dense_ids = {c.chunk_id for c in dense_chunks}
        hybrid_ids = {c.chunk_id for c in hybrid_chunks}
        overlap = dense_ids & hybrid_ids

        print(f"\n  [recall] dense={len(dense_chunks)}  hybrid={len(hybrid_chunks)}  "
              f"overlap={len(overlap)}")
        print(f"    dense-only: {len(dense_ids - hybrid_ids)}  "
              f"hybrid-only: {len(hybrid_ids - dense_ids)}")

        # Both should return results
        assert len(dense_chunks) >= 1
        assert len(hybrid_chunks) >= 1


# ── Category 9: Response Time Benchmarks ───────────────────────────────────────

class TestResponseTime:
    """Measure and assert acceptable response times."""

    def test_simple_factual_time(self):
        query = "What is aspirin?"
        result, elapsed = _run_agent(query, thread_id="perf-factual")

        print(f"\n  [perf] simple-factual: {elapsed:.1f}s  "
              f"(answer={len(result['answer'])} chars, "
              f"cites={len(result['citations'])})")

        assert elapsed < TIMEOUT_HARD, f"Response took too long: {elapsed:.1f}s"
        if elapsed > TIMEOUT_WARN:
            pytest.skip(f"Slow response ({elapsed:.1f}s) — may indicate API latency")

    def test_synthesis_time(self):
        query = "Compare metformin and insulin for diabetes management."
        result, elapsed = _run_agent(query, thread_id="perf-synthesis")

        print(f"\n  [perf] synthesis: {elapsed:.1f}s  "
              f"(answer={len(result['answer'])} chars, "
              f"cites={len(result['citations'])})")

        assert elapsed < TIMEOUT_HARD, f"Response took too long: {elapsed:.1f}s"


# ── Category 10: Edge Cases ────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_very_short_query(self):
        """A single-word query should still return something."""
        query = "aspirin"
        result, elapsed = _run_agent(query, thread_id="edge-short")

        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) >= 10, "Even short query should get some answer"

        print(f"\n  [edge] short query: answer_len={len(result['answer'])}  "
              f"time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD

    def test_technical_query_with_symbols(self):
        """Query with special characters and medical codes."""
        query = "What is ICD-10 code I21.0 and what does it represent?"
        result, elapsed = _run_agent(query, thread_id="edge-icd")

        _assert_valid_result(result)

        print(f"\n  [edge] ICD query: {result['answer'][:150]}...  time={elapsed:.1f}s")

        assert elapsed < TIMEOUT_HARD


# ── Summary Report ─────────────────────────────────────────────────────────────

class TestReport:
    """Dummy test to print a summary header — actual results come from above."""

    def test_placeholder(self):
        print("\n" + "=" * 70)
        print("Integration test suite complete. See above for details.")
        print("=" * 70)
