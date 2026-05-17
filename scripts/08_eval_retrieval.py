"""Retrieval evaluation script: P1-P5 on the Golden Dataset.

Computes Recall@K (K=1,3,5,10,20) and MRR@20 for each pipeline using the
50-question golden dataset. Each pipeline runs in a subprocess to avoid OOM.

Usage:
    python scripts/08_eval_retrieval.py
    python scripts/08_eval_retrieval.py --pipelines p1,p2,p3
    python scripts/08_eval_retrieval.py --k-values 1,3,5,10,20

Output:
    - Markdown table printed to stdout
    - JSON saved to data/eval/retrieval_eval.json
"""
import pyarrow.dataset  # noqa: F401  — must be first (Windows torch DLL fix)

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = ROOT / "data" / "golden" / "golden_dataset.jsonl"
OUTPUT_FILE = ROOT / "data" / "eval" / "retrieval_eval.json"


def _verify_golden_checksum(jsonl_path: Path) -> None:
    """Warn if golden dataset does not match its companion .sha256 file."""
    import hashlib
    sha_path = jsonl_path.with_suffix(jsonl_path.suffix + ".sha256")
    if not sha_path.exists():
        return
    expected = sha_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    if actual != expected:
        print(
            f"[WARN] golden_dataset.jsonl checksum mismatch! "
            f"Expected {expected[:16]}… got {actual[:16]}…\n"
            "       Dataset may have been modified. Re-run parse_golden_dataset.py to update.",
            file=sys.stderr,
        )


PIPELINE_LABELS = {
    "p1": "P1 Dense",
    "p2": "P2 Hybrid",
    "p3": "P3 Hybrid+Rerank",
    "p4": "P4 HyDE",
    "p5": "P5 Multi-Query",
}

K_VALUES = [1, 3, 5, 10, 20]


# ---------------------------------------------------------------------------
# Python executable detection
# ---------------------------------------------------------------------------

def _find_python_exe() -> str:
    """Return a Python executable that has qdrant_client installed."""
    import subprocess as sp
    candidates = [
        sys.executable,
        str(Path.home() / ".conda" / "envs" / "medrag" / "python.exe"),
        str(Path.home() / "miniconda3" / "envs" / "medrag" / "python.exe"),
        str(Path.home() / "anaconda3" / "envs" / "medrag" / "python.exe"),
    ]
    for exe in candidates:
        if not Path(exe).exists():
            continue
        try:
            r = sp.run([exe, "-c", "import qdrant_client"], capture_output=True, timeout=10)
            if r.returncode == 0:
                print(f"[main] using python: {exe}", flush=True)
                return exe
        except Exception:
            pass
    print(f"[warn] could not find python with qdrant_client, using {sys.executable}", flush=True)
    return sys.executable


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------

def _worker_main(pipeline: str, top_k: int) -> None:
    """Retrieve top_k for every golden question; print JSON to stdout."""
    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder
    from medrag.retrieval.retriever import DenseRetriever
    from medrag.retrieval.hybrid import HybridRetriever

    print(f"[worker:{pipeline}] loading embedder...", file=sys.stderr, flush=True)
    qdrant = QdrantClient(url="http://localhost:6333", timeout=30)
    embedder = BGEM3Embedder(device="auto")

    if pipeline == "p1":
        retriever = DenseRetriever(qdrant, embedder)
        fn = lambda q, k: retriever.retrieve(q, k=k)

    elif pipeline == "p2":
        retriever = HybridRetriever(qdrant, embedder, candidate_k=top_k)
        fn = lambda q, k: retriever.retrieve(q, k=k)

    elif pipeline == "p3":
        from medrag.retrieval.reranker import BGEReranker
        print(f"[worker:{pipeline}] loading reranker...", file=sys.stderr, flush=True)
        reranker = BGEReranker(device="auto")
        hybrid = HybridRetriever(qdrant, embedder, candidate_k=top_k)
        fn = lambda q, k: reranker.rerank(q, hybrid.retrieve(q, k=top_k), top_k=k)

    elif pipeline == "p4":
        from medrag.retrieval.hyde import HyDERetriever
        fn_obj = HyDERetriever(qdrant, embedder)
        fn = lambda q, k: fn_obj.retrieve(q, k=k)

    elif pipeline == "p5":
        from medrag.retrieval.multi_query import MultiQueryRetriever
        fn_obj = MultiQueryRetriever(qdrant, embedder)
        fn = lambda q, k: fn_obj.retrieve(q, k=k)

    else:
        print(json.dumps({"error": f"Unknown pipeline: {pipeline}"}))
        return

    golden = [json.loads(l) for l in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"[worker:{pipeline}] evaluating {len(golden)} questions...", file=sys.stderr, flush=True)

    results = []
    for i, item in enumerate(golden, 1):
        qid      = item["id"]
        question = item["question"]
        # Support both v2 (gold_chunk_ids list) and v1 (source_chunk_id single)
        gold_ids: list[str] = (
            item.get("gold_chunk_ids")
            or ([item["source_chunk_id"]] if item.get("source_chunk_id") else [])
        )
        print(f"[worker:{pipeline}] {i}/{len(golden)} {qid} ({len(gold_ids)} gold chunks)",
              file=sys.stderr, flush=True)

        t0 = time.perf_counter()
        try:
            chunks = fn(question, top_k)
            latency = time.perf_counter() - t0
            retrieved_ids = [c.payload.get("chunk_id", "") for c in chunks]

            # Find rank for each gold chunk; rank = position in retrieved list (1-based)
            per_chunk_ranks: dict[str, int | None] = {}
            for gid in gold_ids:
                per_chunk_ranks[gid] = None
                for j, rid in enumerate(retrieved_ids, 1):
                    if rid == gid:
                        per_chunk_ranks[gid] = j
                        break

            # "best rank" = lowest rank among found gold chunks (used by MRR/Recall@k)
            found_ranks = [r for r in per_chunk_ranks.values() if r is not None]
            best_rank   = min(found_ranks) if found_ranks else None
            all_found   = len(found_ranks) == len(gold_ids)

            results.append({
                "id":              qid,
                "question":        question,
                "gold_chunk_ids":  gold_ids,
                "source_chunk_id": gold_ids[0] if gold_ids else "",   # v1 compat
                "category":        item.get("category", ""),
                "difficulty":      item.get("difficulty", ""),
                "difficulty_band": item.get("difficulty_band", ""),
                "pipeline":        pipeline,
                "rank":            best_rank,     # best rank (for Recall@k / MRR compat)
                "per_chunk_ranks": per_chunk_ranks,
                "all_found":       all_found,
                "latency_s":       round(latency, 3),
                "retrieved_ids":   retrieved_ids[:20],
            })
        except Exception as exc:
            latency = time.perf_counter() - t0
            print(f"[worker:{pipeline}] ERROR {qid}: {exc}", file=sys.stderr, flush=True)
            results.append({
                "id":              qid,
                "question":        question,
                "gold_chunk_ids":  gold_ids,
                "source_chunk_id": gold_ids[0] if gold_ids else "",
                "category":        item.get("category", ""),
                "difficulty":      item.get("difficulty", ""),
                "difficulty_band": item.get("difficulty_band", ""),
                "pipeline":        pipeline,
                "rank":            None,
                "per_chunk_ranks": {gid: None for gid in gold_ids},
                "all_found":       False,
                "latency_s":       round(latency, 3),
                "error":           str(exc),
                "retrieved_ids":   [],
            })

    print(json.dumps(results, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_pipeline_subprocess(pipeline: str, top_k: int, python_exe: str, src_root: str) -> list[dict]:
    cmd = [
        python_exe,
        str(Path(__file__).resolve()),
        "--worker", pipeline,
        "--top-k", str(top_k),
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

    for line in (proc.stderr or "").splitlines():
        if line.strip():
            print(f"  {line}", flush=True)

    if proc.returncode != 0:
        print(f"[{pipeline.upper()}] subprocess failed (exit {proc.returncode})", flush=True)
        golden = [json.loads(l) for l in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()]
        return [{"id": g["id"], "pipeline": pipeline, "rank": None,
                 "category": g["category"], "difficulty": g["difficulty"],
                 "error": f"exit {proc.returncode}"} for g in golden]

    try:
        results = json.loads(proc.stdout.strip())
        print(f"[{pipeline.upper()}] done in {elapsed}s", flush=True)
        return results
    except json.JSONDecodeError as e:
        print(f"[{pipeline.upper()}] JSON parse error: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], k_values: list[int]) -> dict:
    """Compute Recall@K, MRR@max_k, and AllFound@K for a list of result records.

    rank = best rank among gold chunks (any gold chunk hit counts for Recall/MRR).
    AllFound@k = fraction of questions where ALL gold chunks appear in top-k.
    """
    max_k = max(k_values)
    n = len(results)
    if n == 0:
        return {}

    recall_at_k  = {}
    all_found_at_k = {}
    for k in k_values:
        hits      = sum(1 for r in results if r.get("rank") is not None and r["rank"] <= k)
        all_found = sum(
            1 for r in results
            if r.get("per_chunk_ranks") and all(
                v is not None and v <= k for v in r["per_chunk_ranks"].values()
            )
        )
        recall_at_k[f"Recall@{k}"]   = round(hits / n, 4)
        all_found_at_k[f"AllFound@{k}"] = round(all_found / n, 4)

    mrr = sum(
        1.0 / r["rank"]
        for r in results
        if r.get("rank") is not None and r["rank"] <= max_k
    )
    avg_latency = [r["latency_s"] for r in results if "latency_s" in r and not r.get("error")]

    return {
        **recall_at_k,
        **all_found_at_k,
        f"MRR@{max_k}": round(mrr / n, 4),
        "Avg_latency_s": round(sum(avg_latency) / len(avg_latency), 3) if avg_latency else 0,
        "n": n,
    }


def compute_breakdown(results: list[dict], k_values: list[int], by: str) -> dict[str, dict]:
    """Compute metrics grouped by category, difficulty, or difficulty_band."""
    groups: dict[str, list] = {}
    for r in results:
        key = r.get(by, "Unknown") or "Unknown"
        groups.setdefault(key, []).append(r)
    return {k: compute_metrics(v, k_values) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_summary_table(all_metrics: dict[str, dict], k_values: list[int], pipelines: list[str]) -> None:
    max_k = max(k_values)
    cols = ["Pipeline"] + [f"R@{k}" for k in k_values] + [f"MRR@{max_k}", "Lat(s)"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    print("\n## Retrieval Evaluation — Summary\n")
    print(header)
    print(sep)
    for p in pipelines:
        m = all_metrics.get(p, {})
        label = PIPELINE_LABELS.get(p, p.upper())
        row = [label]
        for k in k_values:
            row.append(str(m.get(f"Recall@{k}", "—")))
        row.append(str(m.get(f"MRR@{max_k}", "—")))
        row.append(str(m.get("Avg_latency_s", "—")))
        print("| " + " | ".join(row) + " |")


def print_breakdown_table(breakdown: dict[str, dict[str, dict]], k_values: list[int], pipelines: list[str], by: str) -> None:
    max_k = max(k_values)
    recall_col = f"Recall@{k_values[2]}" if len(k_values) > 2 else f"Recall@{k_values[-1]}"

    print(f"\n## Breakdown by {by.capitalize()} (Recall@5 / MRR@{max_k})\n")
    groups = sorted(set(g for p_bd in breakdown.values() for g in p_bd))
    header = "| " + by.capitalize() + " | " + " | ".join(PIPELINE_LABELS.get(p, p) for p in pipelines) + " |"
    sep = "|" + "|".join(["---"] * (len(pipelines) + 1)) + "|"
    print(header)
    print(sep)
    for g in groups:
        row = [g]
        for p in pipelines:
            m = breakdown.get(p, {}).get(g, {})
            r5 = m.get(recall_col, "—")
            mrr = m.get(f"MRR@{max_k}", "—")
            row.append(f"{r5} / {mrr}")
        print("| " + " | ".join(row) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", default="p1,p2,p3",
                        help="Comma-separated pipeline IDs")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Max candidates to retrieve per query")
    parser.add_argument("--k-values", default="1,3,5,10,20",
                        help="Comma-separated K values for Recall@K")
    parser.add_argument("--output", default="data/eval/retrieval_eval.json")
    parser.add_argument("--worker", metavar="PIPELINE",
                        help="(internal) run as worker subprocess")
    args = parser.parse_args()

    if args.worker:
        _worker_main(args.worker, args.top_k)
        return

    k_values = [int(k) for k in args.k_values.split(",")]
    pipelines = [p.strip().lower() for p in args.pipelines.split(",")]

    if not GOLDEN_FILE.exists():
        print(f"[error] {GOLDEN_FILE} not found. Run 07_generate_golden.py first.")
        sys.exit(1)

    _verify_golden_checksum(GOLDEN_FILE)
    n_questions = sum(1 for _ in GOLDEN_FILE.read_text(encoding="utf-8").splitlines())
    print(f"[main] pipelines: {pipelines}")
    print(f"[main] questions: {n_questions}")
    print(f"[main] top_k={args.top_k}  k_values={k_values}")
    print(f"[main] each pipeline runs in a separate subprocess\n")

    python_exe = _find_python_exe()
    src_root = str(ROOT / "src")

    all_results: list[dict] = []
    for i, pid in enumerate(pipelines):
        if i > 0:
            time.sleep(3)
        results = _run_pipeline_subprocess(pid, args.top_k, python_exe, src_root)
        all_results.extend(results)

    if not all_results:
        print("[error] No results collected.", file=sys.stderr)
        sys.exit(1)

    # Compute per-pipeline metrics
    all_metrics: dict[str, dict] = {}
    category_breakdown:   dict[str, dict] = {}
    difficulty_breakdown: dict[str, dict] = {}
    band_breakdown:       dict[str, dict] = {}

    for p in pipelines:
        p_results = [r for r in all_results if r.get("pipeline") == p]
        all_metrics[p]           = compute_metrics(p_results, k_values)
        category_breakdown[p]   = compute_breakdown(p_results, k_values, "category")
        difficulty_breakdown[p] = compute_breakdown(p_results, k_values, "difficulty")
        band_breakdown[p]       = compute_breakdown(p_results, k_values, "difficulty_band")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "top_k": args.top_k,
        "k_values": k_values,
        "n_questions": n_questions,
        "pipelines": pipelines,
        "summary": all_metrics,
        "category_breakdown":   category_breakdown,
        "difficulty_breakdown": difficulty_breakdown,
        "band_breakdown":       band_breakdown,
        "results": all_results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}")

    print_summary_table(all_metrics, k_values, pipelines)
    print_breakdown_table(category_breakdown,   k_values, pipelines, "category")
    print_breakdown_table(difficulty_breakdown, k_values, pipelines, "difficulty")
    print_breakdown_table(band_breakdown,       k_values, pipelines, "difficulty_band")


if __name__ == "__main__":
    main()
