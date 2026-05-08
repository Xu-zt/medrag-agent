"""Agentic pipeline evaluation using LangGraph loop as P4-Agentic.

Runs the full LangGraph agent (route → hybrid retrieve → rerank → grade/rewrite
→ generate → faithfulness check) on the 50-question golden dataset and scores
each answer using MiMo-V2.5-Pro as judge.

Comparison target: P3 (Hybrid+Reranker, static pipeline, answer_eval.json).

Resume-safe: skips already-evaluated IDs.

Usage:
    python scripts/11_eval_agent.py
    python scripts/11_eval_agent.py --model mimo-v2.5-pro --sleep 0.5

Output:
    data/eval/agent_eval.json   (per-question scores)
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
OUTPUT_FILE = ROOT / "data" / "eval" / "agent_eval.json"

# ── Judge prompts (reuse same prompts as script 09) ───────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    """Build the judge OpenAI client.

    Priority:
      1. JUDGE_API_KEY + JUDGE_BASE_URL   — independent third-party judge
      2. OPENAI_API_KEY + OPENAI_BASE_URL — fallback (same as generator; warns about bias)
    """
    load_dotenv(ROOT / ".env")

    judge_key  = os.environ.get("JUDGE_API_KEY")
    judge_url  = os.environ.get("JUDGE_BASE_URL")

    if judge_key and judge_url:
        print("[judge] using independent judge API (JUDGE_API_KEY / JUDGE_BASE_URL)", flush=True)
        return OpenAI(api_key=judge_key, base_url=judge_url)

    # Fallback — warn about self-evaluation bias
    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("[error] Set JUDGE_API_KEY+JUDGE_BASE_URL or OPENAI_API_KEY+OPENAI_BASE_URL in .env")
    print(
        "[judge] WARNING: JUDGE_API_KEY not set — falling back to OPENAI_API_KEY.\n"
        "        Generator and judge share the same backend; scores may be inflated.\n"
        "        Set JUDGE_BASE_URL + JUDGE_API_KEY + JUDGE_MODEL for an independent judge.",
        flush=True,
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat(client: OpenAI, model: str, system: str, user: str, retries: int = 3) -> str:
    max_tokens = 2000
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
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


def _parse_score(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        d = json.loads(m.group()) if m else {}
    score  = float(d.get("score", 0.0))
    issues = str(d.get("issues", ""))
    return max(0.0, min(1.0, score)), issues


def _build_initial_state(question: str) -> dict:
    return {
        "query": question,
        "query_type": "synthesis",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "mimo-v2.5-pro"),
        help="Judge model name (default: JUDGE_MODEL env or mimo-v2.5-pro)",
    )
    parser.add_argument("--sleep",  type=float, default=0.5)
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
    print(f"[eval] {len(golden)} questions | pipeline=P4-Agentic (LangGraph)", flush=True)

    # Resume: load already-evaluated IDs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        for r in existing.get("results", []):
            done[r["id"]] = r
    print(f"[eval] already done: {len(done)}, remaining: {len(golden) - len(done)}", flush=True)

    from medrag.agent.graph import app

    client  = _make_client()
    results = list(done.values())

    for i, item in enumerate(golden, 1):
        qid = item["id"]
        if qid in done:
            continue

        question      = item["question"]
        golden_answer = item["answer"]
        print(f"[eval] {i}/{len(golden)} {qid}", flush=True)

        # 1. Run LangGraph agent
        t0 = time.perf_counter()
        config = {"configurable": {"thread_id": f"eval-{qid}"}}
        try:
            result = app.invoke(_build_initial_state(question), config=config)
        except Exception as exc:
            print(f"  [skip] agent error: {exc}", file=sys.stderr, flush=True)
            results.append({"id": qid, "error": str(exc)})
            continue
        agent_latency = time.perf_counter() - t0

        generated    = result.get("answer", "")
        citations    = result.get("citations", [])
        confidence   = result.get("confidence", 0.0)
        agent_faith  = result.get("faithful", False)
        iterations   = result.get("iterations", 0)
        regen_count  = result.get("regen_count", 0)

        # Build context from retrieved chunks for judge
        chunks = result.get("retrieved_chunks", [])
        context_text = "\n\n".join(
            f"[{j+1}] {c.citation}: {c.text[:2000]}" for j, c in enumerate(chunks)
        ) if chunks else "(no chunks)"

        print(f"  agent: {agent_latency:.1f}s  iter={iterations}  regen={regen_count}"
              f"  faithful={agent_faith}  conf={confidence:.2f}", flush=True)

        if not generated:
            results.append({"id": qid, "error": "empty answer"})
            continue

        # 2. Judge: faithfulness
        try:
            raw = _chat(client, args.model, FAITHFULNESS_SYS,
                        FAITHFULNESS_USR.format(
                            context=context_text[:10000],
                            question=question,
                            answer=generated))
            faith_score, faith_issues = _parse_score(raw)
        except Exception:
            faith_score, faith_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        # 3. Judge: relevance
        try:
            raw = _chat(client, args.model, RELEVANCE_SYS,
                        RELEVANCE_USR.format(question=question, answer=generated))
            rel_score, rel_issues = _parse_score(raw)
        except Exception:
            rel_score, rel_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        # 4. Judge: correctness
        try:
            raw = _chat(client, args.model, CORRECTNESS_SYS,
                        CORRECTNESS_USR.format(
                            question=question,
                            golden=golden_answer[:600],
                            generated=generated[:600]))
            corr_score, corr_issues = _parse_score(raw)
        except Exception:
            corr_score, corr_issues = 0.0, "api_error"
        time.sleep(args.sleep)

        composite = round((faith_score + rel_score + corr_score) / 3, 4)
        print(f"  judge: faith={faith_score:.2f} rel={rel_score:.2f}"
              f" corr={corr_score:.2f} -> comp={composite:.2f}", flush=True)

        rec = {
            "id": qid,
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated,
            "pipeline": "p4-agentic",
            "citations": citations,
            "confidence": confidence,
            "agent_faithful": agent_faith,
            "agent_latency_s": round(agent_latency, 2),
            "iterations": iterations,
            "regen_count": regen_count,
            "faithfulness": faith_score,
            "faithfulness_issues": faith_issues,
            "relevance": rel_score,
            "relevance_issues": rel_issues,
            "correctness": corr_score,
            "correctness_issues": corr_issues,
            "composite": composite,
        }
        results.append(rec)

        # Incremental save
        payload = {
            "pipeline": "p4-agentic",
            "model_judge": args.model,
            "n": len(results),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final summary
    scored = [r for r in results if "composite" in r]
    if scored:
        n = len(scored)
        avg_faith = round(sum(r["faithfulness"] for r in scored) / n, 4)
        avg_rel   = round(sum(r["relevance"]    for r in scored) / n, 4)
        avg_corr  = round(sum(r["correctness"]  for r in scored) / n, 4)
        avg_comp  = round(sum(r["composite"]    for r in scored) / n, 4)
        avg_iter  = round(sum(r.get("iterations",  0) for r in scored) / n, 2)
        avg_regen = round(sum(r.get("regen_count", 0) for r in scored) / n, 2)
        pct_af    = round(100 * sum(1 for r in scored if r.get("agent_faithful")) / n, 1)

        print(f"\n[done] {n} questions evaluated (P4-Agentic)")
        print(f"  Avg faithfulness  : {avg_faith}")
        print(f"  Avg relevance     : {avg_rel}")
        print(f"  Avg correctness   : {avg_corr}")
        print(f"  Avg composite     : {avg_comp}")
        print(f"  Avg rewrites/q    : {avg_iter}")
        print(f"  Avg regens/q      : {avg_regen}")
        print(f"  Agent faithful%   : {pct_af}%")
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
