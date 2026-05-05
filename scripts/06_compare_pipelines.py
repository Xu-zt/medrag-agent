"""Pipeline comparison script: P1 through P5 side-by-side with latency.

Usage:
    python scripts/06_compare_pipelines.py
    python scripts/06_compare_pipelines.py --pipelines p1,p2,p3
    python scripts/06_compare_pipelines.py --k 5 --candidate-k 20
    python scripts/06_compare_pipelines.py --output data/eval/my_run.json

Output:
    - Markdown table printed to stdout
    - JSON saved to data/eval/pipeline_comparison.json (or --output path)
"""
# Windows + CUDA: preload pyarrow before torch to avoid access violation (0xC0000005)
import pyarrow.dataset  # noqa: F401

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Force UTF-8 output on Windows (avoids GBK codec errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever, RetrievedChunk
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker

# ---------------------------------------------------------------------------
# Sample queries (diverse medical sub-domains for meaningful ablation)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "What is the typical spatial resolution of 3T MRI in clinical practice?",
    "What is the mechanism of action of PARP inhibitors in BRCA-mutated cancers?",
    "How do PI-RADS v2.1 criteria distinguish between score 3 and score 4?",
    "What are the contraindications for thrombolytic therapy in ischemic stroke?",
    "Describe the role of EGFR T790M mutation in NSCLC treatment resistance.",
]


# ---------------------------------------------------------------------------
# Pipeline factory — returns a callable (query, k) -> list[RetrievedChunk]
# ---------------------------------------------------------------------------

def build_pipelines(
    qdrant: QdrantClient,
    embedder: BGEM3Embedder,
    reranker: BGEReranker,
    candidate_k: int,
    requested: list[str],
) -> dict[str, Callable]:
    available: dict[str, Callable] = {}

    if "p1" in requested:
        p1 = DenseRetriever(qdrant, embedder)
        available["p1"] = lambda q, k, _r=p1: _r.retrieve(q, k=k)

    if "p2" in requested:
        p2 = HybridRetriever(qdrant, embedder, candidate_k=candidate_k)
        available["p2"] = lambda q, k, _r=p2: _r.retrieve(q, k=k)

    if "p3" in requested:
        p3_retriever = HybridRetriever(qdrant, embedder, candidate_k=candidate_k)
        available["p3"] = lambda q, k, _r=p3_retriever, _rr=reranker: _rr.rerank(
            q, _r.retrieve(q, k=candidate_k), top_k=k
        )

    if "p4" in requested:
        try:
            from medrag.retrieval.hyde import HyDERetriever
            p4 = HyDERetriever(qdrant, embedder)
            available["p4"] = lambda q, k, _r=p4: _r.retrieve(q, k=k)
        except ImportError:
            print("[skip] p4: medrag.retrieval.hyde not found yet", file=sys.stderr)

    if "p5" in requested:
        try:
            from medrag.retrieval.multi_query import MultiQueryRetriever
            p5 = MultiQueryRetriever(qdrant, embedder)
            available["p5"] = lambda q, k, _r=p5: _r.retrieve(q, k=k)
        except ImportError:
            print("[skip] p5: medrag.retrieval.multi_query not found yet", file=sys.stderr)

    return available


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def run_comparison(
    queries: list[str],
    pipelines: dict[str, Callable],
    k: int,
) -> list[dict]:
    """Return a list of result dicts, one per (query × pipeline) pair."""
    results = []
    total = len(queries) * len(pipelines)
    done = 0

    for query in queries:
        for pid, fn in pipelines.items():
            done += 1
            print(f"[{done}/{total}] {pid.upper()} | {query[:60]}...", flush=True)
            t0 = time.perf_counter()
            try:
                chunks = fn(query, k)
                latency = time.perf_counter() - t0
                results.append({
                    "query": query,
                    "pipeline": pid,
                    "latency_s": round(latency, 3),
                    "error": None,
                    "hits": [
                        {
                            "rank": i + 1,
                            "citation": c.citation,
                            "score": round(float(c.score), 4),
                            "snippet": c.text[:200],
                        }
                        for i, c in enumerate(chunks)
                    ],
                })
            except Exception as exc:
                latency = time.perf_counter() - t0
                print(f"  [error] {exc}", file=sys.stderr)
                results.append({
                    "query": query,
                    "pipeline": pid,
                    "latency_s": round(latency, 3),
                    "error": str(exc),
                    "hits": [],
                })

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

PIPELINE_LABELS = {
    "p1": "P1 Dense",
    "p2": "P2 Hybrid",
    "p3": "P3 Hybrid+Reranker",
    "p4": "P4 HyDE",
    "p5": "P5 Multi-Query",
}


def print_markdown_table(results: list[dict], pipelines: list[str]) -> None:
    """Print a compact Markdown table: rows = queries, cols = pipelines."""
    queries = list(dict.fromkeys(r["query"] for r in results))

    # Build lookup: (query, pipeline) -> result
    lookup = {(r["query"], r["pipeline"]): r for r in results}

    # Header
    cols = [PIPELINE_LABELS.get(p, p.upper()) for p in pipelines]
    header = "| Query | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    print("\n## Pipeline Comparison Results\n")
    print(header)
    print(sep)

    for q in queries:
        short_q = q[:55] + "..." if len(q) > 55 else q
        row_parts = [f"`{short_q}`"]
        for p in pipelines:
            r = lookup.get((q, p))
            if r is None:
                row_parts.append("—")
            elif r["error"]:
                row_parts.append("ERROR")
            else:
                top_hits = ", ".join(h["citation"] for h in r["hits"][:3])
                row_parts.append(f"{r['latency_s']}s · {top_hits}")
        print("| " + " | ".join(row_parts) + " |")

    # Latency summary table
    print("\n## Latency Summary (avg over queries)\n")
    print("| Pipeline | Avg Latency (s) | Min (s) | Max (s) |")
    print("|---|---|---|---|")
    for p in pipelines:
        times = [r["latency_s"] for r in results if r["pipeline"] == p and not r["error"]]
        if not times:
            print(f"| {PIPELINE_LABELS.get(p, p.upper())} | — | — | — |")
        else:
            avg = round(sum(times) / len(times), 3)
            print(
                f"| {PIPELINE_LABELS.get(p, p.upper())} "
                f"| {avg} | {min(times)} | {max(times)} |"
            )


def print_detailed_results(results: list[dict], k: int) -> None:
    """Print per-query per-pipeline top-k hits for detailed inspection."""
    queries = list(dict.fromkeys(r["query"] for r in results))
    lookup = {(r["query"], r["pipeline"]): r for r in results}

    print("\n## Detailed Results\n")
    for q in queries:
        print(f"### Query: {q}\n")
        for r in [lookup.get((q, p)) for p in sorted(set(r["pipeline"] for r in results))]:
            if r is None:
                continue
            label = PIPELINE_LABELS.get(r["pipeline"], r["pipeline"].upper())
            print(f"**{label}** ({r['latency_s']}s)")
            if r["error"]:
                print(f"> ERROR: {r['error']}")
            else:
                for h in r["hits"]:
                    print(f"  {h['rank']}. [{h['citation']}] score={h['score']:.4f}")
                    print(f"     {h['snippet'][:150]}...")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipelines",
        default="p1,p2,p3",
        help="Comma-separated pipeline IDs to run (e.g. p1,p2,p3,p4,p5)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results per pipeline")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate pool for P2/P3")
    parser.add_argument(
        "--output",
        default="data/eval/pipeline_comparison.json",
        help="Path to save JSON output",
    )
    parser.add_argument("--detailed", action="store_true", help="Print per-query hit details")
    args = parser.parse_args()

    requested = [p.strip().lower() for p in args.pipelines.split(",")]
    print(f"[init] pipelines: {requested}", flush=True)

    # Initialize shared resources once
    print("[init] connecting to Qdrant...", flush=True)
    qdrant = QdrantClient(url="http://localhost:6333", timeout=30)

    print("[init] loading BGE-M3 embedder (CPU)...", flush=True)
    embedder = BGEM3Embedder(device="cpu")

    reranker = None
    if "p3" in requested:
        print("[init] loading BGE reranker (CPU)...", flush=True)
        reranker = BGEReranker(device="cpu")

    pipelines = build_pipelines(qdrant, embedder, reranker, args.candidate_k, requested)
    if not pipelines:
        print("[error] No valid pipelines found. Exiting.", file=sys.stderr)
        sys.exit(1)

    active = list(pipelines.keys())
    print(f"[init] active pipelines: {active}", flush=True)
    print(f"[init] queries: {len(SAMPLE_QUERIES)}, k={args.k}\n", flush=True)

    results = run_comparison(SAMPLE_QUERIES, pipelines, args.k)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"k": args.k, "candidate_k": args.candidate_k, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[saved] {out_path}")

    # Print Markdown
    print_markdown_table(results, active)
    if args.detailed:
        print_detailed_results(results, args.k)


if __name__ == "__main__":
    main()
