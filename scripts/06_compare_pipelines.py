"""Pipeline comparison script: P1 through P5 side-by-side with latency.

Each pipeline runs in its own subprocess to avoid simultaneous model loading
(BGE-M3 + BGE-Reranker together exceed Windows virtual memory in one process).

Usage:
    python scripts/06_compare_pipelines.py
    python scripts/06_compare_pipelines.py --pipelines p1,p2,p3
    python scripts/06_compare_pipelines.py --pipelines p1,p2,p3,p4,p5
    python scripts/06_compare_pipelines.py --k 5 --candidate-k 20
    python scripts/06_compare_pipelines.py --detailed

Output:
    - Markdown table printed to stdout
    - JSON saved to data/eval/pipeline_comparison.json
"""
# Windows + CUDA: preload pyarrow before torch to avoid access violation (0xC0000005).
# Must be at the very top of the file so it runs even in worker subprocess mode.
import pyarrow.dataset  # noqa: F401

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Sample queries — diverse medical sub-domains for meaningful ablation
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "What is the typical spatial resolution of 3T MRI in clinical practice?",
    "What is the mechanism of action of PARP inhibitors in BRCA-mutated cancers?",
    "How do PI-RADS v2.1 criteria distinguish between score 3 and score 4?",
    "What are the contraindications for thrombolytic therapy in ischemic stroke?",
    "Describe the role of EGFR T790M mutation in NSCLC treatment resistance.",
]

PIPELINE_LABELS = {
    "p1": "P1 Dense",
    "p2": "P2 Hybrid",
    "p3": "P3 Hybrid+Reranker",
    "p4": "P4 HyDE",
    "p5": "P5 Multi-Query",
}

# ---------------------------------------------------------------------------
# Worker mode — called by subprocess with --worker flag
# ---------------------------------------------------------------------------

def _worker_main(pipeline: str, k: int, candidate_k: int) -> None:
    """Run a single pipeline over all SAMPLE_QUERIES, print JSON to stdout."""

    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder

    print(f"[worker:{pipeline}] loading embedder...", file=sys.stderr, flush=True)
    from medrag.config import qdrant_url

    qdrant = QdrantClient(url=qdrant_url(), timeout=30)
    embedder = BGEM3Embedder(device="cpu")

    from medrag.retrieval.retriever import DenseRetriever, RetrievedChunk
    from medrag.retrieval.hybrid import HybridRetriever

    if pipeline == "p1":
        retriever = DenseRetriever(qdrant, embedder)
        fn = lambda q, k_: retriever.retrieve(q, k=k_)

    elif pipeline == "p2":
        retriever = HybridRetriever(qdrant, embedder, candidate_k=candidate_k)
        fn = lambda q, k_: retriever.retrieve(q, k=k_)

    elif pipeline == "p3":
        from medrag.retrieval.reranker import BGEReranker
        print(f"[worker:{pipeline}] loading reranker...", file=sys.stderr, flush=True)
        reranker = BGEReranker(device="cpu")
        hybrid = HybridRetriever(qdrant, embedder, candidate_k=candidate_k)
        fn = lambda q, k_: reranker.rerank(q, hybrid.retrieve(q, k=candidate_k), top_k=k_)

    elif pipeline == "p4":
        from medrag.retrieval.hyde import HyDERetriever
        hyde = HyDERetriever(qdrant, embedder)
        fn = lambda q, k_: hyde.retrieve(q, k=k_)

    elif pipeline == "p5":
        from medrag.retrieval.multi_query import MultiQueryRetriever
        mq = MultiQueryRetriever(qdrant, embedder)
        fn = lambda q, k_: mq.retrieve(q, k=k_)

    else:
        print(json.dumps({"error": f"Unknown pipeline: {pipeline}"}))
        return

    print(f"[worker:{pipeline}] running {len(SAMPLE_QUERIES)} queries...", file=sys.stderr, flush=True)
    results = []
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"[worker:{pipeline}] {i}/{len(SAMPLE_QUERIES)} {query[:50]}...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        try:
            chunks = fn(query, k)
            latency = time.perf_counter() - t0
            results.append({
                "query": query,
                "pipeline": pipeline,
                "latency_s": round(latency, 3),
                "error": None,
                "hits": [
                    {
                        "rank": j + 1,
                        "citation": c.citation,
                        "score": round(float(c.score), 4),
                        "snippet": c.text[:200],
                    }
                    for j, c in enumerate(chunks)
                ],
            })
        except Exception as exc:
            latency = time.perf_counter() - t0
            print(f"[worker:{pipeline}] ERROR: {exc}", file=sys.stderr, flush=True)
            results.append({
                "query": query,
                "pipeline": pipeline,
                "latency_s": round(latency, 3),
                "error": str(exc),
                "hits": [],
            })

    # Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Subprocess runner — called from main process
# ---------------------------------------------------------------------------

def _run_pipeline_subprocess(
    pipeline: str,
    k: int,
    candidate_k: int,
    python_exe: str,
    src_root: str,
) -> list[dict]:
    """Invoke this script with --worker in a subprocess, return parsed results."""
    cmd = [
        python_exe,
        str(Path(__file__).resolve()),
        "--worker", pipeline,
        "--k", str(k),
        "--candidate-k", str(candidate_k),
    ]
    env = {
        **__import__("os").environ,
        "PYTHONPATH": src_root,
        "PYTHONIOENCODING": "utf-8",
    }
    print(f"\n[{pipeline.upper()}] starting subprocess...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    elapsed = round(time.perf_counter() - t0, 1)

    # Print stderr (progress messages) to console
    for line in (proc.stderr or "").splitlines():
        if line.strip():
            print(f"  {line}", flush=True)

    if proc.returncode != 0:
        print(f"[{pipeline.upper()}] subprocess failed (exit {proc.returncode})", flush=True)
        return [
            {
                "query": q,
                "pipeline": pipeline,
                "latency_s": 0.0,
                "error": f"subprocess exit {proc.returncode}",
                "hits": [],
            }
            for q in SAMPLE_QUERIES
        ]

    try:
        results = json.loads(proc.stdout.strip())
        print(f"[{pipeline.upper()}] done in {elapsed}s total", flush=True)
        return results
    except json.JSONDecodeError as e:
        print(f"[{pipeline.upper()}] JSON parse error: {e}", flush=True)
        print(f"  stdout: {proc.stdout[:200]}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_markdown_table(results: list[dict], pipelines: list[str]) -> None:
    queries = list(dict.fromkeys(r["query"] for r in results))
    lookup = {(r["query"], r["pipeline"]): r for r in results}

    cols = [PIPELINE_LABELS.get(p, p.upper()) for p in pipelines]
    print("\n## Pipeline Comparison Results\n")
    print("| Query | " + " | ".join(cols) + " |")
    print("|---|" + "|".join(["---"] * len(cols)) + "|")

    for q in queries:
        short_q = q[:52] + "..." if len(q) > 52 else q
        row_parts = [f"`{short_q}`"]
        for p in pipelines:
            r = lookup.get((q, p))
            if r is None:
                row_parts.append("—")
            elif r.get("error"):
                row_parts.append("ERROR")
            else:
                top = ", ".join(h["citation"] for h in r["hits"][:3])
                row_parts.append(f"{r['latency_s']}s · {top}")
        print("| " + " | ".join(row_parts) + " |")

    print("\n## Latency Summary (avg over queries)\n")
    print("| Pipeline | Avg (s) | Min (s) | Max (s) |")
    print("|---|---|---|---|")
    for p in pipelines:
        times = [r["latency_s"] for r in results if r["pipeline"] == p and not r.get("error")]
        if not times:
            print(f"| {PIPELINE_LABELS.get(p, p.upper())} | — | — | — |")
        else:
            avg = round(sum(times) / len(times), 3)
            print(f"| {PIPELINE_LABELS.get(p, p.upper())} | {avg} | {min(times)} | {max(times)} |")


def print_detailed_results(results: list[dict]) -> None:
    queries = list(dict.fromkeys(r["query"] for r in results))
    lookup = {(r["query"], r["pipeline"]): r for r in results}
    pipelines_seen = sorted(set(r["pipeline"] for r in results))

    print("\n## Detailed Results\n")
    for q in queries:
        print(f"### {q}\n")
        for p in pipelines_seen:
            r = lookup.get((q, p))
            if r is None:
                continue
            label = PIPELINE_LABELS.get(p, p.upper())
            print(f"**{label}** ({r['latency_s']}s)")
            if r.get("error"):
                print(f"> ERROR: {r['error']}")
            else:
                for h in r["hits"]:
                    print(f"  {h['rank']}. `{h['citation']}` score={h['score']:.4f}")
                    print(f"     _{h['snippet'][:140]}..._")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", default="p1,p2,p3",
                        help="Comma-separated pipeline IDs (p1,p2,p3,p4,p5)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--output", default="data/eval/pipeline_comparison.json")
    parser.add_argument("--detailed", action="store_true")
    # Internal: called by subprocess runner
    parser.add_argument("--worker", metavar="PIPELINE",
                        help="(internal) run as worker for this pipeline ID")
    args = parser.parse_args()

    if args.worker:
        _worker_main(args.worker, args.k, args.candidate_k)
        return

    requested = [p.strip().lower() for p in args.pipelines.split(",")]
    print(f"[main] pipelines: {requested}  k={args.k}  candidate_k={args.candidate_k}")
    print(f"[main] queries: {len(SAMPLE_QUERIES)}")
    print(f"[main] each pipeline runs in a separate subprocess (memory isolation)\n")

    python_exe = sys.executable
    src_root = str(Path(__file__).resolve().parent.parent / "src")

    all_results: list[dict] = []
    for i, pid in enumerate(requested):
        if i > 0:
            # Brief pause between subprocesses to let the OS reclaim memory and
            # page file from the previous model-loading process before the next
            # one starts (avoids Windows 0xC0000005 on back-to-back heavy loads).
            time.sleep(3)
        results = _run_pipeline_subprocess(pid, args.k, args.candidate_k, python_exe, src_root)
        all_results.extend(results)

    if not all_results:
        print("[error] No results collected.", file=sys.stderr)
        sys.exit(1)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"k": args.k, "candidate_k": args.candidate_k, "results": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}")

    print_markdown_table(all_results, requested)
    if args.detailed:
        print_detailed_results(all_results)


if __name__ == "__main__":
    main()
