"""Answer quality evaluation using MiMo-V2.5-Pro as judge.

Evaluates the full RAG pipeline (retrieval + generation) on the Golden Dataset
using P3 (best pipeline) by default. Three dimensions are scored:

  faithfulness   — is the generated answer grounded in retrieved chunks?
  relevance      — does the answer actually address the question?
  correctness    — does it convey the same information as the golden answer?

Each dimension is scored 0.0-1.0. Resume-safe: skips already-evaluated IDs.

Usage:
    python scripts/09_eval_answer.py
    python scripts/09_eval_answer.py --pipeline p3 --model mimo-v2.5-pro
    python scripts/09_eval_answer.py --pipeline p5

Output:
    data/eval/answer_eval.json   (per-question scores)
"""
import pyarrow.dataset  # noqa: F401

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = ROOT / "data" / "golden" / "golden_dataset.jsonl"
OUTPUT_FILE = ROOT / "data" / "eval" / "answer_eval.json"

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_SYS = (
    "You are evaluating a RAG system's answer for faithfulness to retrieved context. "
    "Score only whether the answer is grounded in the provided context chunks, "
    "not whether it is medically correct in general."
)
FAITHFULNESS_USR = """\
Context chunks:
{context}

Question: {question}
Generated answer: {answer}

Score the faithfulness (0.0-1.0): how well every claim in the answer is supported by the context.
Output ONLY valid JSON: {{"score": 0.0-1.0, "issues": "brief note or none"}}"""

RELEVANCE_SYS = "You are evaluating whether a generated answer addresses the question asked."
RELEVANCE_USR = """\
Question: {question}
Generated answer: {answer}

Score the relevance (0.0-1.0): does the answer directly address what was asked?
0.0 = completely off-topic, 1.0 = fully addresses the question.
Output ONLY valid JSON: {{"score": 0.0-1.0, "issues": "brief note or none"}}"""

CORRECTNESS_SYS = (
    "You are comparing a generated answer to a golden reference answer for a medical RAG benchmark. "
    "Score how much correct information the generated answer conveys relative to the reference."
)
CORRECTNESS_USR = """\
Question: {question}
Golden answer: {golden}
Generated answer: {generated}

Score correctness (0.0-1.0): overlap of correct information between generated and golden answer.
1.0 = all key facts present and correct, 0.0 = mostly wrong or missing.
Output ONLY valid JSON: {{"score": 0.0-1.0, "issues": "brief note or none"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> OpenAI:
    """Build judge client.  Prefers JUDGE_* vars over OPENAI_* to avoid
    self-evaluation bias when the generator also uses the OpenAI-compat backend."""
    load_dotenv(ROOT / ".env")
    judge_key = os.environ.get("JUDGE_API_KEY")
    judge_url = os.environ.get("JUDGE_BASE_URL")
    if judge_key and judge_url:
        print("[judge] using independent judge API", flush=True)
        return OpenAI(api_key=judge_key, base_url=judge_url)
    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("[error] Set JUDGE_API_KEY+JUDGE_BASE_URL or OPENAI_API_KEY+OPENAI_BASE_URL in .env")
    print("[judge] WARNING: JUDGE_API_KEY not set — generator and judge share the same backend.", flush=True)
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat(client: OpenAI, model: str, system: str, user: str, retries: int = 3) -> str:
    max_tokens = 2000
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content and resp.choices[0].finish_reason == "length":
                max_tokens = min(max_tokens * 2, 6000)
                continue
            return content
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [api] attempt {attempt+1} failed: {exc}. retry in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API failed after {retries} attempts")


def _parse_score(text: str) -> tuple[float, str]:
    """Extract score and issues from judge JSON output."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        d = json.loads(m.group()) if m else {}
    score = float(d.get("score", 0.0))
    issues = str(d.get("issues", ""))
    return max(0.0, min(1.0, score)), issues


def _get_pipeline(pipeline: str):
    """Return a retrieval function for the given pipeline ID."""
    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder
    from medrag.retrieval.retriever import DenseRetriever
    from medrag.retrieval.hybrid import HybridRetriever

    from medrag.config import COLLECTION_NAME, qdrant_url

    qdrant = QdrantClient(url=qdrant_url(), timeout=30)
    embedder = BGEM3Embedder(device="cpu")

    if pipeline == "p1":
        r = DenseRetriever(qdrant, embedder)
        return lambda q, k: r.retrieve(q, k=k)
    if pipeline == "p2":
        r = HybridRetriever(qdrant, embedder, candidate_k=20)
        return lambda q, k: r.retrieve(q, k=k)
    if pipeline == "p3":
        from medrag.retrieval.reranker import BGEReranker
        reranker = BGEReranker(device="cpu")
        hybrid = HybridRetriever(qdrant, embedder, candidate_k=20)
        return lambda q, k: reranker.rerank(q, hybrid.retrieve(q, k=20), top_k=k)
    if pipeline == "p4":
        from medrag.retrieval.hyde import HyDERetriever
        r = HyDERetriever(qdrant, embedder)
        return lambda q, k: r.retrieve(q, k=k)
    if pipeline == "p5":
        from medrag.retrieval.multi_query import MultiQueryRetriever
        r = MultiQueryRetriever(qdrant, embedder)
        return lambda q, k: r.retrieve(q, k=k)
    raise ValueError(f"Unknown pipeline: {pipeline}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="p3")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="mimo-v2.5-pro")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument(
        "--golden",
        default=str(GOLDEN_FILE),
        help="Path to golden dataset JSONL (default: golden_dataset.jsonl)",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise SystemExit(f"[error] {golden_path} not found.")

    golden = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines()]
    print(f"[eval] {len(golden)} questions | pipeline={args.pipeline} | k={args.k}", flush=True)

    # Resume: load already-evaluated IDs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        for r in existing.get("results", []):
            done[r["id"]] = r
    print(f"[eval] already done: {len(done)}, remaining: {len(golden)-len(done)}", flush=True)

    from medrag.agent.generator import generate_answer
    retrieve = _get_pipeline(args.pipeline)
    client = _make_client()

    results = list(done.values())

    for i, item in enumerate(golden, 1):
        qid = item["id"]
        if qid in done:
            continue

        question = item["question"]
        golden_answer = item["answer"]
        print(f"[eval] {i}/{len(golden)} {qid}", flush=True)

        # 1. Retrieve
        t0 = time.perf_counter()
        try:
            chunks = retrieve(question, args.k)
        except Exception as exc:
            print(f"  [skip] retrieval error: {exc}", file=sys.stderr, flush=True)
            results.append({"id": qid, "error": str(exc)})
            continue
        retrieval_latency = time.perf_counter() - t0

        # 2. Generate
        t1 = time.perf_counter()
        try:
            generated = generate_answer(question, chunks)
        except Exception as exc:
            print(f"  [skip] generation error: {exc}", file=sys.stderr, flush=True)
            results.append({"id": qid, "error": str(exc)})
            continue
        gen_latency = time.perf_counter() - t1

        context_text = "\n\n".join(
            f"[{j+1}] {c.citation}: {c.text[:2000]}" for j, c in enumerate(chunks)
        )

        # 3. Judge: faithfulness
        try:
            raw = _chat(client, args.model, FAITHFULNESS_SYS,
                        FAITHFULNESS_USR.format(context=context_text[:10000],
                                                question=question, answer=generated))
            faith_score, faith_issues = _parse_score(raw)
        except Exception:
            faith_score, faith_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        # 4. Judge: relevance
        try:
            raw = _chat(client, args.model, RELEVANCE_SYS,
                        RELEVANCE_USR.format(question=question, answer=generated))
            rel_score, rel_issues = _parse_score(raw)
        except Exception:
            rel_score, rel_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        # 5. Judge: correctness vs golden
        try:
            raw = _chat(client, args.model, CORRECTNESS_SYS,
                        CORRECTNESS_USR.format(question=question,
                                               golden=golden_answer[:600],
                                               generated=generated[:600]))
            corr_score, corr_issues = _parse_score(raw)
        except Exception:
            corr_score, corr_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        composite = round((faith_score + rel_score + corr_score) / 3, 4)
        print(f"  faith={faith_score:.2f} rel={rel_score:.2f} corr={corr_score:.2f} "
              f"→ composite={composite:.2f}", flush=True)

        rec = {
            "id": qid,
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated,
            "pipeline": args.pipeline,
            "k": args.k,
            "retrieval_latency_s": round(retrieval_latency, 3),
            "generation_latency_s": round(gen_latency, 3),
            "faithfulness": faith_score,
            "faithfulness_issues": faith_issues,
            "relevance": rel_score,
            "relevance_issues": rel_issues,
            "correctness": corr_score,
            "correctness_issues": corr_issues,
            "composite": composite,
        }
        results.append(rec)

        # Save incrementally
        payload = {
            "pipeline": args.pipeline,
            "k": args.k,
            "model_judge": args.model,
            "n": len(results),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final summary
    scored = [r for r in results if "composite" in r]
    if scored:
        avg_faith = round(sum(r["faithfulness"] for r in scored) / len(scored), 4)
        avg_rel = round(sum(r["relevance"] for r in scored) / len(scored), 4)
        avg_corr = round(sum(r["correctness"] for r in scored) / len(scored), 4)
        avg_comp = round(sum(r["composite"] for r in scored) / len(scored), 4)
        print(f"\n[done] {len(scored)} questions evaluated")
        print(f"  Avg faithfulness : {avg_faith}")
        print(f"  Avg relevance    : {avg_rel}")
        print(f"  Avg correctness  : {avg_corr}")
        print(f"  Avg composite    : {avg_comp}")
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
