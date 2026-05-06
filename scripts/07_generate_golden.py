"""Corpus-grounded Golden Dataset generator using MiMo-V2.5-Pro API.

Four phases (run --phase all to execute in sequence):
  sample    — select substantive chunks from corpus (→ data/eval/gd_candidates.jsonl)
  generate  — MiMo generates Q-A pairs from each candidate chunk
              (→ data/eval/gd_raw.jsonl, resume-safe)
  verify    — retrieval coverage + MiMo faithfulness + parametric check
              (→ data/eval/gd_verified.jsonl, resume-safe)
  finalize  — balance by category/difficulty, write final dataset
              (→ data/golden/golden_dataset.md  +  data/golden/golden_dataset.jsonl)

Usage:
    python scripts/07_generate_golden.py --phase all
    python scripts/07_generate_golden.py --phase sample --n-candidates 200
    python scripts/07_generate_golden.py --phase generate
    python scripts/07_generate_golden.py --phase verify
    python scripts/07_generate_golden.py --phase finalize --n-final 50

Requirements in .env:
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...   (MiMo-V2.5-Pro endpoint)
"""
from __future__ import annotations

# Windows + CUDA: preload pyarrow before torch
import pyarrow.dataset  # noqa: F401

import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
GOLDEN_DIR = ROOT / "data" / "golden"
CHUNKS_FILE = ROOT / "data" / "index_cache" / "chunks.jsonl"
CANDIDATES_FILE = EVAL_DIR / "gd_candidates.jsonl"
RAW_FILE = EVAL_DIR / "gd_raw.jsonl"
VERIFIED_FILE = EVAL_DIR / "gd_verified.jsonl"
FINAL_MD = GOLDEN_DIR / "golden_dataset.md"
FINAL_JSONL = GOLDEN_DIR / "golden_dataset.jsonl"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

# ── Target dataset distribution (total = n-final) ─────────────────────────────
# weights used for proportional sampling in finalize
CATEGORY_WEIGHTS = {
    "Radiology": 3,
    "Oncology": 2,
    "Cardiology": 1,
    "Neurology": 2,
    "Pharmacology": 1,
    "Infectious Disease": 1,
    "General": 2,
}
DIFFICULTY_WEIGHTS = {"Easy": 2, "Medium": 3, "Hard": 2}

VALID_CATEGORIES = set(CATEGORY_WEIGHTS)
VALID_DIFFICULTIES = set(DIFFICULTY_WEIGHTS)

# ── MiMo prompts ──────────────────────────────────────────────────────────────

GENERATE_SYSTEM = """\
You are building a benchmark to evaluate medical RAG (Retrieval-Augmented Generation) systems.
Your questions must be corpus-grounded: they should require retrieving the specific passage
provided to answer correctly, and should NOT be answerable from general medical knowledge alone."""

GENERATE_USER_TMPL = """\
Create a question-answer pair from the medical text below.

PASSAGE:
---
{text}
---
Source: {source}

STRICT REQUIREMENTS:
1. Ask about SPECIFIC details in this passage: exact percentages, measurements, study findings,
   diagnostic criteria, drug mechanisms, specific patient populations, named techniques, etc.
2. The question must NOT be answerable from standard medical education without this passage.
3. The answer must be directly and fully supported by the passage text (no extrapolation).
4. Write the answer as 2-4 complete, self-contained sentences.

Category — choose ONE: Pharmacology / Oncology / Radiology / Cardiology /
                        Neurology / Infectious Disease / General

Difficulty:
  Easy   = single direct fact (one number, one name, one definition)
  Medium = requires relating 2+ facts from the passage
  Hard   = requires understanding a mechanism, comparing findings, or identifying a subtle point

Output ONLY valid JSON (no markdown fence, no extra text):
{{
  "question": "...",
  "answer": "...",
  "difficulty": "Easy|Medium|Hard",
  "category": "...",
  "key_evidence": "exact sentence(s) from the passage the answer relies on",
  "notes": "one sentence on what retrieval skill this tests"
}}"""

FAITHFULNESS_SYSTEM = """\
You are a precise medical fact-checker for a retrieval benchmark.
Assess only whether the candidate answer is faithful to the passage — not whether it is
medically correct in general. Every claim in the answer must be directly traceable to the passage."""

FAITHFULNESS_USER_TMPL = """\
Passage:
{text}

Question: {question}
Candidate answer: {answer}

Output ONLY valid JSON:
{{
  "faithful": true or false,
  "confidence": 0.0-1.0,
  "issues": "list unsupported or contradicted claims, or write none"
}}"""

PARAMETRIC_SYSTEM = """\
You are a medical knowledge system. Answer from general medical education only.
If the question requires specific study data, measurements, or research findings
that would only appear in a retrieved document, respond with exactly: INSUFFICIENT"""

PARAMETRIC_USER_TMPL = """\
Medical question: {question}

Answer briefly (2-3 sentences) using only general medical knowledge.
If this requires specific research data, say INSUFFICIENT."""

EQUIV_SYSTEM = "You assess answer equivalence for retrieval benchmarking."

EQUIV_USER_TMPL = """\
Reference answer: {reference}
Parametric answer: {candidate}

Do these answers convey substantially the same information?
Output ONLY valid JSON:
{{
  "equivalent": true or false,
  "confidence": 0.0-1.0
}}"""


# ── MiMo API helpers ──────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("[error] OPENAI_API_KEY not found in .env")
    if not base_url:
        raise SystemExit("[error] OPENAI_BASE_URL not found in .env — add it: OPENAI_BASE_URL=https://your-mimo-endpoint/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat(client: OpenAI, model: str, system: str, user: str,
          max_tokens: int = 3000, retries: int = 3) -> str:
    """Call the MiMo API with retry + exponential backoff.

    MiMo-V2.5-Pro is a reasoning model: reasoning_tokens count against
    max_tokens. A typical reasoning pass uses 800-1500 tokens, so we
    default to 3000 to leave room for the actual JSON output.
    If content is empty (all tokens consumed by reasoning), we raise so
    the caller can retry with a higher limit.
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()
            if not content and resp.choices[0].finish_reason == "length":
                # All tokens consumed by reasoning; retry with higher limit
                max_tokens = min(max_tokens * 2, 8000)
                print(f"  [api] empty content (finish=length), retrying with max_tokens={max_tokens}",
                      file=sys.stderr, flush=True)
                continue
            return content
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [api] attempt {attempt+1} failed: {exc}. retrying in {wait}s...",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {retries} attempts")


def _parse_json(text: str) -> dict | None:
    """Extract the first {...} JSON block from model output."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Phase 1: Sample ───────────────────────────────────────────────────────────

def phase_sample(n_candidates: int, seed: int = 42) -> None:
    print(f"[sample] loading chunks from {CHUNKS_FILE}...", flush=True)
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"[error] {CHUNKS_FILE} not found. Run build_index first.")

    all_chunks: list[dict] = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    print(f"[sample] total chunks: {len(all_chunks)}", flush=True)

    def is_substantive(c: dict) -> bool:
        t = c.get("text", "").strip()
        # Skip very short chunks, pure headings, or mostly whitespace
        return (len(t) >= 200 and
                len(t.split()) >= 35 and
                not re.match(r"^[A-Z][A-Za-z\s]{0,30}$", t[:60]))

    # ── PubMed: all abstracts are good; sample proportionally ─────────────────
    pub_good = [c for c in all_chunks if c["source"] == "pubmed" and is_substantive(c)]

    # ── PMC: group by paper, sample up to 5 per paper (prefer Results/Methods) ─
    pmc_by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in all_chunks:
        if c["source"] == "pmc" and is_substantive(c):
            title = c.get("metadata", {}).get("title", "")
            pmc_by_paper[title].append(c)

    pmc_good: list[dict] = []
    rng = random.Random(seed)
    for title, chunks in pmc_by_paper.items():
        # Prefer Results / Discussion / Methods sections
        def section_priority(ch: dict) -> int:
            sec = ch.get("metadata", {}).get("section", "").upper()
            if any(k in sec for k in ("RESULT", "FINDING", "CONCLUS")):
                return 0
            if any(k in sec for k in ("DISCUSS", "METHOD", "MATERIAL")):
                return 1
            return 2
        sorted_chunks = sorted(chunks, key=section_priority)
        # Pick up to 5, skipping duplicates by first 100 chars
        seen: set[str] = set()
        picked = 0
        for ch in sorted_chunks:
            key = ch["text"][:100]
            if key not in seen:
                seen.add(key)
                pmc_good.append(ch)
                picked += 1
                if picked >= 5:
                    break

    print(f"[sample] PubMed substantive: {len(pub_good)}", flush=True)
    print(f"[sample] PMC substantive: {len(pmc_good)} (from {len(pmc_by_paper)} papers)", flush=True)

    # Combine and subsample
    pool = pub_good + pmc_good
    rng.shuffle(pool)
    candidates = pool[:n_candidates]

    with CANDIDATES_FILE.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[sample] saved {len(candidates)} candidates → {CANDIDATES_FILE}", flush=True)


# ── Phase 2: Generate ─────────────────────────────────────────────────────────

def phase_generate(model: str, sleep_s: float = 1.0) -> None:
    if not CANDIDATES_FILE.exists():
        raise SystemExit("[error] Run --phase sample first.")

    candidates: list[dict] = []
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"[generate] {len(candidates)} candidates", flush=True)

    # Load already-generated chunk_ids (for resume)
    done_ids: set[str] = set()
    if RAW_FILE.exists():
        with RAW_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["source_chunk_id"])
                except Exception:
                    pass
    print(f"[generate] already done: {len(done_ids)}, remaining: {len(candidates)-len(done_ids)}",
          flush=True)

    client = _make_client()

    with RAW_FILE.open("a", encoding="utf-8") as out:
        for i, chunk in enumerate(candidates):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            if cid in done_ids:
                continue

            source_label = (
                f"PMID:{chunk['doc_id']}" if chunk["source"] == "pubmed"
                else f"PMC:{chunk.get('metadata',{}).get('title','')[:60]}"
            )
            print(f"[generate] {i+1}/{len(candidates)} | {cid}", flush=True)

            try:
                raw = _chat(
                    client, model,
                    system=GENERATE_SYSTEM,
                    user=GENERATE_USER_TMPL.format(
                        text=chunk["text"][:2000],
                        source=source_label,
                    ),
                    max_tokens=3000,
                )
                parsed = _parse_json(raw)
            except Exception as exc:
                print(f"  [skip] API error: {exc}", file=sys.stderr, flush=True)
                time.sleep(sleep_s * 2)
                continue

            if not parsed:
                print(f"  [skip] JSON parse failed. raw={raw[:100]}", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
                continue

            # Validate fields
            q = str(parsed.get("question", "")).strip()
            a = str(parsed.get("answer", "")).strip()
            diff = str(parsed.get("difficulty", "")).strip()
            cat = str(parsed.get("category", "")).strip()

            if not q or not a:
                print("  [skip] empty question or answer", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
                continue
            if diff not in VALID_DIFFICULTIES:
                diff = "Medium"
            if cat not in VALID_CATEGORIES:
                cat = "General"

            record = {
                "source_chunk_id": cid,
                "source": chunk["source"],
                "doc_id": chunk.get("doc_id", ""),
                "chunk_text": chunk["text"],
                "question": q,
                "answer": a,
                "difficulty": diff,
                "category": cat,
                "key_evidence": str(parsed.get("key_evidence", "")),
                "notes": str(parsed.get("notes", "")),
                "verification": {},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(sleep_s)

    n = sum(1 for _ in RAW_FILE.open(encoding="utf-8"))
    print(f"[generate] done. {n} raw Q-A pairs → {RAW_FILE}", flush=True)


# ── Phase 3: Verify ───────────────────────────────────────────────────────────

def phase_verify(model: str, retrieval_top_k: int = 20, sleep_s: float = 0.8) -> None:
    if not RAW_FILE.exists():
        raise SystemExit("[error] Run --phase generate first.")

    raw: list[dict] = []
    with RAW_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except Exception:
                pass
    print(f"[verify] {len(raw)} raw pairs to verify", flush=True)

    # Load already-verified (for resume)
    done_ids: set[str] = set()
    if VERIFIED_FILE.exists():
        with VERIFIED_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["source_chunk_id"])
                except Exception:
                    pass
    print(f"[verify] already verified: {len(done_ids)}", flush=True)

    # Setup retrieval (P1 dense — fastest, sufficient for coverage check)
    print("[verify] loading retrieval components...", flush=True)
    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder
    qdrant = QdrantClient(url="http://localhost:6333", timeout=30)
    embedder = BGEM3Embedder(device="cpu")
    print("[verify] retrieval ready", flush=True)

    client = _make_client()

    def check_retrieval(question: str, source_chunk_id: str) -> dict:
        enc = embedder.encode([question])
        vec = enc["dense"][0].tolist()
        results = qdrant.query_points(
            collection_name="medrag_text",
            query=vec,
            using="dense",
            limit=retrieval_top_k,
            with_payload=True,
        )
        ids = [p.payload.get("chunk_id", "") for p in results.points]
        rank = None
        for idx, cid in enumerate(ids):
            if cid == source_chunk_id:
                rank = idx + 1
                break
        return {
            "retrieval_rank": rank,
            "retrieval_pass": rank is not None and rank <= retrieval_top_k,
        }

    def check_faithfulness(question: str, answer: str, chunk_text: str) -> dict:
        try:
            raw_resp = _chat(
                client, model,
                system=FAITHFULNESS_SYSTEM,
                user=FAITHFULNESS_USER_TMPL.format(
                    text=chunk_text[:1500],
                    question=question,
                    answer=answer,
                ),
                max_tokens=2000,
            )
            parsed = _parse_json(raw_resp)
            if parsed:
                faithful = bool(parsed.get("faithful", False))
                conf = float(parsed.get("confidence", 0.5))
                return {
                    "faithful": faithful,
                    "faithfulness_confidence": conf,
                    "faithfulness_issues": str(parsed.get("issues", "")),
                    "faithfulness_pass": faithful and conf >= 0.8,
                }
        except Exception as exc:
            print(f"  [faithfulness] error: {exc}", file=sys.stderr)
        return {"faithful": False, "faithfulness_confidence": 0.0,
                "faithfulness_issues": "api_error", "faithfulness_pass": False}

    def check_parametric(question: str, reference_answer: str) -> dict:
        try:
            param_raw = _chat(
                client, model,
                system=PARAMETRIC_SYSTEM,
                user=PARAMETRIC_USER_TMPL.format(question=question),
                max_tokens=2000,
            )
            is_insufficient = "INSUFFICIENT" in param_raw.upper()
            if is_insufficient:
                return {"is_parametric": False, "parametric_answer": "INSUFFICIENT"}

            # Compare parametric answer to reference
            equiv_raw = _chat(
                client, model,
                system=EQUIV_SYSTEM,
                user=EQUIV_USER_TMPL.format(
                    reference=reference_answer[:500],
                    candidate=param_raw[:500],
                ),
                max_tokens=2000,
            )
            equiv_parsed = _parse_json(equiv_raw)
            is_equiv = bool(equiv_parsed.get("equivalent", False)) if equiv_parsed else False
            conf = float(equiv_parsed.get("confidence", 0.0)) if equiv_parsed else 0.0
            return {
                "is_parametric": is_equiv and conf >= 0.75,
                "parametric_answer": param_raw[:300],
                "parametric_confidence": conf,
            }
        except Exception as exc:
            print(f"  [parametric] error: {exc}", file=sys.stderr)
        return {"is_parametric": False, "parametric_answer": "error"}

    passed = failed_retrieval = failed_faith = 0

    with VERIFIED_FILE.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(raw):
            cid = rec["source_chunk_id"]
            if cid in done_ids:
                continue

            q = rec["question"]
            a = rec["answer"]
            print(f"[verify] {i+1}/{len(raw)} | {cid[:50]}", flush=True)

            # 3a. Retrieval coverage
            ret = check_retrieval(q, cid)
            if not ret["retrieval_pass"]:
                failed_retrieval += 1
                print(f"  [retrieval] FAIL (not in top-{retrieval_top_k})", flush=True)

            # 3b. Faithfulness (always check, even if retrieval failed)
            time.sleep(sleep_s)
            faith = check_faithfulness(q, a, rec["chunk_text"])
            if not faith["faithfulness_pass"]:
                failed_faith += 1
                print(f"  [faithful] FAIL conf={faith['faithfulness_confidence']:.2f} | {faith['faithfulness_issues'][:80]}", flush=True)

            # 3c. Parametric knowledge (only if both above pass — save API quota)
            time.sleep(sleep_s)
            if ret["retrieval_pass"] and faith["faithfulness_pass"]:
                param = check_parametric(q, a)
                if param["is_parametric"]:
                    print(f"  [parametric] WARNING: answerable from general knowledge", flush=True)
            else:
                param = {"is_parametric": False, "parametric_answer": "skipped"}

            # Compute overall pass
            overall_pass = ret["retrieval_pass"] and faith["faithfulness_pass"]
            if overall_pass:
                passed += 1

            verified_rec = {
                **rec,
                "verification": {
                    **ret,
                    **faith,
                    **param,
                    "overall_pass": overall_pass,
                },
            }
            out.write(json.dumps(verified_rec, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(sleep_s)

    total = len(raw) - len(done_ids)
    print(f"\n[verify] results (new in this run):", flush=True)
    print(f"  total processed: {total}", flush=True)
    print(f"  retrieval fail : {failed_retrieval}", flush=True)
    print(f"  faithfulness fail: {failed_faith}", flush=True)
    print(f"  overall pass   : {passed}", flush=True)
    print(f"[verify] saved → {VERIFIED_FILE}", flush=True)


# ── Phase 4: Finalize ─────────────────────────────────────────────────────────

def phase_finalize(n_final: int, seed: int = 42) -> None:
    if not VERIFIED_FILE.exists():
        raise SystemExit("[error] Run --phase verify first.")

    verified: list[dict] = []
    with VERIFIED_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                verified.append(json.loads(line))
            except Exception:
                pass
    print(f"[finalize] loaded {len(verified)} verified pairs", flush=True)

    passed = [r for r in verified if r["verification"].get("overall_pass", False)]
    print(f"[finalize] overall_pass=True: {len(passed)}", flush=True)

    if len(passed) < n_final:
        print(f"[finalize] WARNING: only {len(passed)} passed but need {n_final}. "
              f"Including best failures.", file=sys.stderr, flush=True)
        # Add best retrieval-pass items even if faithfulness borderline
        borderline = [r for r in verified
                      if not r["verification"].get("overall_pass")
                      and r["verification"].get("retrieval_pass")]
        borderline.sort(key=lambda r: r["verification"].get("faithfulness_confidence", 0), reverse=True)
        passed = passed + borderline
        passed = passed[:n_final * 2]  # work with double pool

    # ── Balanced selection ──────────────────────────────────────────────────────
    # Group by (category, difficulty)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in passed:
        key = (r["category"], r["difficulty"])
        groups[key].append(r)

    # Compute per-cell target quota proportionally
    total_cat_weight = sum(CATEGORY_WEIGHTS.values())
    total_diff_weight = sum(DIFFICULTY_WEIGHTS.values())
    quotas: dict[tuple, float] = {}
    for cat, cw in CATEGORY_WEIGHTS.items():
        for diff, dw in DIFFICULTY_WEIGHTS.items():
            quotas[(cat, diff)] = n_final * (cw / total_cat_weight) * (dw / total_diff_weight)

    rng = random.Random(seed)
    selected: list[dict] = []

    # Round-1: fill each cell up to floor(quota), prefer non-parametric
    remaining: dict[tuple, list[dict]] = {}
    for key, items in groups.items():
        # Sort: non-parametric first, then by faithfulness confidence
        items_sorted = sorted(
            items,
            key=lambda r: (
                int(r["verification"].get("is_parametric", False)),
                -r["verification"].get("faithfulness_confidence", 0),
            ),
        )
        remaining[key] = items_sorted

    cell_taken: dict[tuple, int] = defaultdict(int)
    for key, quota in sorted(quotas.items(), key=lambda x: -x[1]):
        take = int(quota)
        pool = remaining.get(key, [])
        for item in pool[:take]:
            selected.append(item)
            cell_taken[key] += 1
        remaining[key] = pool[take:]

    # Round-2: fill remaining slots greedily from leftover items
    leftover = [item for pool in remaining.values() for item in pool]
    rng.shuffle(leftover)
    for item in leftover:
        if len(selected) >= n_final:
            break
        selected.append(item)

    selected = selected[:n_final]
    rng.shuffle(selected)

    # ── Print balance summary ──────────────────────────────────────────────────
    print(f"\n[finalize] selected {len(selected)} questions:", flush=True)
    from collections import Counter
    cats = Counter(r["category"] for r in selected)
    diffs = Counter(r["difficulty"] for r in selected)
    print(f"  Categories: {dict(cats)}", flush=True)
    print(f"  Difficulties: {dict(diffs)}", flush=True)
    param_count = sum(1 for r in selected if r["verification"].get("is_parametric"))
    print(f"  Parametric (too easy): {param_count}", flush=True)
    avg_rank = [r["verification"]["retrieval_rank"] for r in selected
                if r["verification"].get("retrieval_rank")]
    if avg_rank:
        print(f"  Avg retrieval rank: {sum(avg_rank)/len(avg_rank):.1f}", flush=True)

    # ── Write Markdown ─────────────────────────────────────────────────────────
    md_lines = [
        "# MedRAG-Agent Golden Dataset\n",
        "> Auto-generated from corpus using MiMo-V2.5-Pro. Each question is corpus-grounded:\n",
        "> the answer can only be found by retrieving the source passage.\n",
        "> Verified: retrieval_rank ≤ 20 AND faithfulness_confidence ≥ 0.80.\n",
        "\n---\n",
    ]
    for idx, rec in enumerate(selected, 1):
        qid = f"Q{idx:03d}"
        v = rec["verification"]
        ret_rank = v.get("retrieval_rank", "?")
        faith_conf = v.get("faithfulness_confidence", 0)
        parametric_flag = " [parametric]" if v.get("is_parametric") else ""
        notes_extra = (
            f"Generated from {rec['source']}:{rec['doc_id']} | "
            f"retrieval_rank={ret_rank} | faithfulness={faith_conf:.2f}{parametric_flag}"
        )
        md_lines += [
            f"## {qid}\n\n",
            f"**Category**: {rec['category']}\n",
            f"**Difficulty**: {rec['difficulty']}\n\n",
            f"**Question**: {rec['question']}\n\n",
            f"**Answer**: {rec['answer']}\n\n",
            f"**Notes**: {notes_extra}\n\n",
            "---\n\n",
        ]
    FINAL_MD.write_text("".join(md_lines), encoding="utf-8")
    print(f"[finalize] written → {FINAL_MD}", flush=True)

    # ── Write JSONL ────────────────────────────────────────────────────────────
    with FINAL_JSONL.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(selected, 1):
            qid = f"Q{idx:03d}"
            v = rec["verification"]
            faith_conf = v.get("faithfulness_confidence", 0)
            ret_rank = v.get("retrieval_rank", None)
            parametric_flag = " [parametric]" if v.get("is_parametric") else ""
            entry = {
                "id": qid,
                "category": rec["category"],
                "difficulty": rec["difficulty"],
                "question": rec["question"],
                "answer": rec["answer"],
                "notes": (
                    f"Generated from {rec['source']}:{rec['doc_id']} | "
                    f"retrieval_rank={ret_rank} | faithfulness={faith_conf:.2f}{parametric_flag}"
                ),
                "source_chunk_id": rec["source_chunk_id"],
                "retrieval_rank": ret_rank,
                "faithfulness_confidence": faith_conf,
                "is_parametric": v.get("is_parametric", False),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[finalize] written → {FINAL_JSONL}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sample", "generate", "verify", "finalize", "all"],
                    default="all")
    ap.add_argument("--n-candidates", type=int, default=200,
                    help="Number of corpus chunks to sample (phase: sample)")
    ap.add_argument("--n-final", type=int, default=50,
                    help="Target number of golden questions (phase: finalize)")
    ap.add_argument("--model", default="mimo-v2.5-pro",
                    help="MiMo model name as required by the API endpoint")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between API calls")
    ap.add_argument("--retrieval-top-k", type=int, default=20,
                    help="Top-k for retrieval coverage check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Add src/ to path so medrag imports work
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    phases = (
        ["sample", "generate", "verify", "finalize"]
        if args.phase == "all"
        else [args.phase]
    )

    for phase in phases:
        print(f"\n{'='*60}", flush=True)
        print(f"Phase: {phase.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        if phase == "sample":
            phase_sample(args.n_candidates, seed=args.seed)
        elif phase == "generate":
            phase_generate(args.model, sleep_s=args.sleep)
        elif phase == "verify":
            phase_verify(args.model, retrieval_top_k=args.retrieval_top_k, sleep_s=args.sleep)
        elif phase == "finalize":
            phase_finalize(args.n_final, seed=args.seed)


if __name__ == "__main__":
    main()
