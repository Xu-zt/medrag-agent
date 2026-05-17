"""Golden Dataset generator v2 — corpus-grounded, multi-claim, difficulty-spread.

Key changes vs v1:
  1. Sample   — chunk CLUSTERS (2–4 chunks from the same document), not isolated chunks
  2. Generate — clinical paraphrase constraint + structured claim answers
  3. Verify   — no rank-based pass/fail filter; assigns difficulty_band from P1 rank
  4. Finalize — balances by category × difficulty_band so all pipelines have headroom

Output schema (per item in golden_dataset.jsonl):
  {
    "id":                    "Q001",
    "question":              "<clinical paraphrase, no verbatim rare terms>",
    "answer":                "<4–8 sentence complete reference>",
    "gold_chunk_ids":        ["pubmed:123:0", "pubmed:123:2"],
    "claims":                [{"text": "...", "chunk_id": "pubmed:123:0"}, ...],
    "difficulty_band":       "easy|medium|hard",   # from retrieval rank
    "difficulty":            "Easy|Medium|Hard",   # question complexity
    "category":              "Cardiology",
    "source":                "pubmed|pmc",
    "doc_id":                "...",
    "source_chunk_id":       "pubmed:123:0",       # first gold chunk (backward compat)
    "retrieval_rank_p1":     {"pubmed:123:0": 3, "pubmed:123:2": 12},
    "faithfulness_confidence": 0.9,
    "is_parametric":         false
  }

Usage:
    python scripts/07_generate_golden.py --phase all
    python scripts/07_generate_golden.py --phase sample --n-candidates 150
    python scripts/07_generate_golden.py --phase generate
    python scripts/07_generate_golden.py --phase verify
    python scripts/07_generate_golden.py --phase finalize --n-final 50
"""
from __future__ import annotations

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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# sentence_transformers MUST be imported before qdrant_client and FlagEmbedding
# to initialise PyTorch before the gRPC C++ runtime on Windows — avoids
# STATUS_ACCESS_VIOLATION when BGEM3Embedder loads later in the verify phase.
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
EVAL_DIR     = ROOT / "data" / "eval"
GOLDEN_DIR   = ROOT / "data" / "golden"
CHUNKS_FILE  = ROOT / "data" / "index_cache" / "chunks.jsonl"

CANDIDATES_FILE = EVAL_DIR / "gd_candidates.jsonl"   # list of cluster dicts
RAW_FILE        = EVAL_DIR / "gd_raw.jsonl"
VERIFIED_FILE   = EVAL_DIR / "gd_verified.jsonl"
FINAL_MD        = GOLDEN_DIR / "golden_dataset.md"
FINAL_JSONL     = GOLDEN_DIR / "golden_dataset.jsonl"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset target distribution ───────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    "Radiology": 3, "Oncology": 2, "Cardiology": 2,
    "Neurology": 2, "Pharmacology": 1, "Infectious Disease": 1, "General": 2,
}
# difficulty_band: rank-based retrieval difficulty (not question complexity)
BAND_WEIGHTS    = {"easy": 1, "medium": 2, "hard": 1}   # 25% easy / 50% medium / 25% hard
DIFFICULTY_WEIGHTS = {"Easy": 2, "Medium": 3, "Hard": 2}

VALID_CATEGORIES   = set(CATEGORY_WEIGHTS)
VALID_DIFFICULTIES = set(DIFFICULTY_WEIGHTS)

EASY_RANK_MAX   = 3     # P1 worst-case rank ≤ 3  → easy band
HARD_RANK_MIN   = 16    # P1 worst-case rank ≥ 16 → hard band

# ── MiMo prompts ──────────────────────────────────────────────────────────────

GENERATE_SYSTEM = """\
You are building a benchmark to evaluate medical RAG systems. Your questions must:
  1. Require information from MULTIPLE passages to answer fully.
  2. Be written the way a CLINICIAN would ask — in natural clinical language.
     Do NOT copy rare acronyms, model names, exact measurements, or distinctive
     phrases verbatim from the passages. Paraphrase into everyday clinical phrasing.
  3. Not be answerable from standard medical education without these specific passages.
  4. Have answers that are fully traceable to the provided passages."""

GENERATE_USER_TMPL = """\
Create a question-answer pair that requires synthesising information from ALL the passages below.

PASSAGES:
---
{passages}
---

REQUIREMENTS:
- The question must integrate ≥2 distinct facts drawn from ≥2 different passages.
- Phrase the question as a clinician would ask it in a hospital or journal club.
  FORBIDDEN: copying any rare acronym, technique name, exact number, or study-specific
  phrase verbatim from the passages — always paraphrase into natural clinical language.
- The answer must be 4–8 complete sentences covering all essential facts.
- Each answer claim must be tagged with the passage ID it comes from.

Category — choose ONE: Pharmacology / Oncology / Radiology / Cardiology /
                        Neurology / Infectious Disease / General

Question complexity:
  Easy   = facts from passages can be extracted directly with minimal reasoning
  Medium = requires relating or comparing ≥2 facts across passages
  Hard   = requires understanding a mechanism, trend, or subtle clinical implication

Output ONLY valid JSON (no markdown, no extra text):
{{
  "question": "...",
  "answer": "...",
  "difficulty": "Easy|Medium|Hard",
  "category": "...",
  "claims": [
    {{"text": "one sentence claim", "chunk_id": "{chunk_ids[0]}"}},
    {{"text": "another claim",      "chunk_id": "{chunk_ids[1]}"}}
  ],
  "notes": "one sentence on what retrieval skill this tests"
}}"""

FAITHFULNESS_SYSTEM = """\
You are a medical fact-checker for a retrieval benchmark.
For each claim, check whether it is directly supported by the specific chunk tagged in claim.chunk_id.
Every claim must trace to its tagged chunk — not to another chunk or general knowledge."""

FAITHFULNESS_USER_TMPL = """\
Chunks:
{chunks_block}

Claims to verify (each claim references a specific chunk_id):
{claims_json}

Output ONLY valid JSON:
{{
  "overall_faithful": true or false,
  "confidence": 0.0-1.0,
  "issues": "list any unsupported claims, or 'none'"
}}"""

PARAMETRIC_SYSTEM = """\
You are a medical knowledge system. Answer from general medical education only.
If the question requires specific study data, measurements, or research findings
that would only appear in a retrieved document, respond with exactly: INSUFFICIENT"""

PARAMETRIC_USER_TMPL = "Medical question: {question}\n\nAnswer briefly using only general knowledge. Say INSUFFICIENT if you need specific research data."

PARAPHRASE_CHECK_SYSTEM = """\
Check whether a question contains phrases copied verbatim from source passages.
A 'lexical leak' occurs when the question uses a rare acronym, technique name,
model name, or 3+ consecutive content words lifted directly from a passage."""

PARAPHRASE_CHECK_USER_TMPL = """\
Question: {question}

Source passages (combined):
{passages_combined}

Does the question contain verbatim rare phrases from the passages?
Output ONLY valid JSON: {{"has_leak": true or false, "leaked_phrases": ["..."]}}"""


# ── API helpers ───────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_BASE_URL")
    if not key:
        raise SystemExit("[error] OPENAI_API_KEY not in .env")
    if not url:
        raise SystemExit("[error] OPENAI_BASE_URL not in .env")
    return OpenAI(api_key=key, base_url=url)


def _chat(client: OpenAI, model: str, system: str, user: str,
          max_tokens: int = 2000, retries: int = 3, temp: float = 0.3) -> str:
    """Call MiMo API with thinking disabled (required for non-empty content)."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                print(f"  [api] empty content (finish={resp.choices[0].finish_reason}), "
                      f"retrying with max_tokens={min(max_tokens*2, 6000)}",
                      file=sys.stderr, flush=True)
                max_tokens = min(max_tokens * 2, 6000)
                continue
            return content
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [api] attempt {attempt+1} failed: {exc}. retry in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {retries} attempts")


def _parse_json(text: str) -> dict | None:
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


# ── Word-bigram lexical leak detector (no external deps) ─────────────────────

def _word_bigrams(text: str) -> set[tuple]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return set(zip(words, words[1:])) if len(words) >= 2 else set()


def _lexical_leak_score(question: str, chunks_text: str) -> float:
    """Fraction of question word-bigrams that appear in source chunk text."""
    q_bg = _word_bigrams(question)
    if not q_bg:
        return 0.0
    c_bg = _word_bigrams(chunks_text)
    return len(q_bg & c_bg) / len(q_bg)


# ── Phase 1: Sample chunk clusters ────────────────────────────────────────────

def phase_sample(n_candidates: int, seed: int = 42) -> None:
    # Load chunks from Qdrant (authoritative source — guarantees chunk_id matches index).
    # Fall back to local CHUNKS_FILE only if Qdrant is unreachable.
    print("[sample] loading chunks from Qdrant (authoritative source)...", flush=True)
    all_chunks: list[dict] = []
    try:
        from qdrant_client import QdrantClient as _QC
        _qc = _QC(url="http://localhost:6333", timeout=30)
        offset = None
        while True:
            batch, offset = _qc.scroll(
                "medrag_text",
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in batch:
                pl = p.payload or {}
                all_chunks.append({
                    "chunk_id": pl.get("chunk_id", ""),
                    "source":   pl.get("source", ""),
                    "doc_id":   pl.get("doc_id", ""),
                    "text":     pl.get("text", ""),
                    "metadata": pl.get("metadata", {}),
                })
            if offset is None:
                break
        print(f"[sample] loaded {len(all_chunks)} chunks from Qdrant", flush=True)
    except Exception as e:
        print(f"[sample] Qdrant unavailable ({e}), falling back to {CHUNKS_FILE}", flush=True)
        if not CHUNKS_FILE.exists():
            raise SystemExit(f"[error] {CHUNKS_FILE} not found. Run 04_build_index.py first.")
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            for line in f:
                all_chunks.append(json.loads(line))
        print(f"[sample] total chunks: {len(all_chunks)}", flush=True)

    def is_substantive(c: dict) -> bool:
        t = c.get("text", "").strip()
        return (
            len(t) >= 200
            and len(t.split()) >= 35
            and not re.match(r"^[A-Z][A-Za-z\s]{0,30}$", t[:60])
        )

    # ── Group chunks by document ───────────────────────────────────────────────
    # Key: (source, doc_id)  →  list of chunk dicts
    # PMC chunks often have doc_id=""; fall back to metadata['pmid'] or title hash.
    by_doc: dict[tuple, list[dict]] = defaultdict(list)
    for c in all_chunks:
        if not is_substantive(c):
            continue
        doc_id = c.get("doc_id", "")
        if not doc_id:
            # PMC: prefer pmid in metadata, then title, then chunk_id prefix
            meta = c.get("metadata", {})
            pmid = meta.get("pmid")
            title = meta.get("title") or ""
            doc_id = (
                str(pmid) if pmid is not None else None
            ) or title[:80] or c.get("chunk_id", "").rsplit(":", 1)[0]
        doc_key = (c["source"], doc_id)
        by_doc[doc_key].append(c)

    # ── Build clusters ─────────────────────────────────────────────────────────
    # A cluster = 2–4 chunks from the same document covering complementary sections.
    rng = random.Random(seed)
    clusters: list[dict] = []

    def section_priority(ch: dict) -> int:
        sec = ch.get("metadata", {}).get("section", "").upper()
        if any(k in sec for k in ("RESULT", "FINDING", "CONCLUS")):
            return 0
        if any(k in sec for k in ("DISCUSS", "METHOD", "MATERIAL")):
            return 1
        return 2

    for (source, doc_id), chunks in by_doc.items():
        if len(chunks) < 2:
            continue   # need ≥ 2 chunks for multi-fact questions

        # Sort: best sections first, dedupe by first-100-chars
        sorted_chunks = sorted(chunks, key=section_priority)
        seen: set[str] = set()
        deduped: list[dict] = []
        for ch in sorted_chunks:
            key = ch["text"][:100]
            if key not in seen:
                seen.add(key)
                deduped.append(ch)

        if len(deduped) < 2:
            continue

        # Pick 2–4 chunks, biased toward diverse sections
        pick_n = min(4, len(deduped))
        # Try to get chunks from different section priority levels
        by_prio: dict[int, list] = defaultdict(list)
        for ch in deduped:
            by_prio[section_priority(ch)].append(ch)
        selected_chunks: list[dict] = []
        for prio in sorted(by_prio.keys()):
            candidates = by_prio[prio]
            rng.shuffle(candidates)
            selected_chunks.append(candidates[0])
            if len(selected_chunks) >= pick_n:
                break
        # Fill up to pick_n if needed
        remaining = [ch for ch in deduped if ch not in selected_chunks]
        rng.shuffle(remaining)
        selected_chunks += remaining[:pick_n - len(selected_chunks)]

        cluster = {
            "source": source,
            "doc_id": doc_id,
            "cluster_id": f"{source}:{doc_id}",
            "chunks": [
                {
                    "chunk_id": ch.get("chunk_id", ""),
                    "text": ch["text"],
                    "section": ch.get("metadata", {}).get("section", ""),
                    "metadata": ch.get("metadata", {}),
                }
                for ch in selected_chunks
            ],
        }
        clusters.append(cluster)

    print(f"[sample] total clusters: {len(clusters)}", flush=True)
    rng.shuffle(clusters)
    selected = clusters[:n_candidates]

    with CANDIDATES_FILE.open("w", encoding="utf-8") as f:
        for cl in selected:
            f.write(json.dumps(cl, ensure_ascii=False) + "\n")
    print(f"[sample] saved {len(selected)} clusters → {CANDIDATES_FILE}", flush=True)


# ── Phase 2: Generate ─────────────────────────────────────────────────────────

def phase_generate(model: str, sleep_s: float = 1.0) -> None:
    if not CANDIDATES_FILE.exists():
        raise SystemExit("[error] Run --phase sample first.")

    candidates: list[dict] = []
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"[generate] {len(candidates)} clusters", flush=True)

    done_ids: set[str] = set()
    if RAW_FILE.exists():
        with RAW_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["cluster_id"])
                except Exception:
                    pass
    print(f"[generate] already done: {len(done_ids)}, remaining: {len(candidates)-len(done_ids)}",
          flush=True)

    client = _make_client()

    with RAW_FILE.open("a", encoding="utf-8") as out:
        for i, cluster in enumerate(candidates):
            cid = cluster["cluster_id"]
            if cid in done_ids:
                continue

            chunks = cluster["chunks"]
            chunk_ids = [ch["chunk_id"] for ch in chunks]

            # Build passage block for prompt
            passages_parts = []
            for j, ch in enumerate(chunks):
                section = ch.get("section", "")
                label = f"[{ch['chunk_id']}]" + (f" ({section})" if section else "")
                passages_parts.append(f"{label}:\n{ch['text'][:1500]}")
            passages_block = "\n\n---\n\n".join(passages_parts)

            print(f"[generate] {i+1}/{len(candidates)} | {cid} ({len(chunks)} chunks)", flush=True)

            try:
                # Build prompt with actual chunk_ids interpolated for claim tagging
                user_prompt = GENERATE_USER_TMPL.format(
                    passages=passages_block,
                    chunk_ids=chunk_ids,
                )
                raw = _chat(client, model, GENERATE_SYSTEM, user_prompt,
                            max_tokens=2000, temp=0.4)
                parsed = _parse_json(raw)
            except Exception as exc:
                print(f"  [skip] API error: {exc}", file=sys.stderr, flush=True)
                time.sleep(sleep_s * 2)
                continue

            if not parsed:
                print(f"  [skip] JSON parse failed. raw={raw[:120]}", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
                continue

            q    = str(parsed.get("question", "")).strip()
            a    = str(parsed.get("answer", "")).strip()
            diff = str(parsed.get("difficulty", "")).strip()
            cat  = str(parsed.get("category", "")).strip()
            claims_raw = parsed.get("claims", [])
            notes = str(parsed.get("notes", ""))

            if not q or not a:
                print("  [skip] empty question or answer", file=sys.stderr, flush=True)
                continue
            if diff not in VALID_DIFFICULTIES:
                diff = "Medium"
            if cat not in VALID_CATEGORIES:
                cat = "General"

            # Normalize claims list
            valid_chunk_ids = set(chunk_ids)
            claims: list[dict] = []
            for cl in claims_raw:
                if isinstance(cl, dict) and cl.get("text") and cl.get("chunk_id"):
                    tag = str(cl["chunk_id"])
                    # Fuzzy match: allow partial chunk_id matches
                    matched = tag if tag in valid_chunk_ids else next(
                        (cid for cid in valid_chunk_ids if cid.endswith(tag.split(":")[-1])), None
                    )
                    if matched:
                        claims.append({"text": str(cl["text"]), "chunk_id": matched})

            if len(claims) < 2:
                print(f"  [skip] only {len(claims)} valid claims (need ≥2)", file=sys.stderr, flush=True)
                continue

            # Quick lexical leak check (fast, no API call)
            passages_combined = " ".join(ch["text"] for ch in chunks)
            leak_score = _lexical_leak_score(q, passages_combined)
            if leak_score > 0.45:
                print(f"  [warn] high lexical leak ({leak_score:.2f}) — keeping but flagging",
                      file=sys.stderr, flush=True)

            record = {
                "cluster_id":  cid,
                "source":      cluster["source"],
                "doc_id":      cluster["doc_id"],
                "chunks":      chunks,
                "question":    q,
                "answer":      a,
                "difficulty":  diff,
                "category":    cat,
                "claims":      claims,
                "notes":       notes,
                "lexical_leak_score": round(leak_score, 3),
                "verification": {},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(sleep_s)

    n = sum(1 for _ in open(RAW_FILE, encoding="utf-8"))
    print(f"[generate] done. {n} raw Q-A pairs → {RAW_FILE}", flush=True)


# ── Phase 3: Verify ───────────────────────────────────────────────────────────

def phase_verify(model: str, retrieval_top_k: int = 50, sleep_s: float = 0.8) -> None:
    if not RAW_FILE.exists():
        raise SystemExit("[error] Run --phase generate first.")

    raw: list[dict] = []
    with RAW_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except Exception:
                pass
    print(f"[verify] {len(raw)} raw pairs", flush=True)

    done_ids: set[str] = set()
    if VERIFIED_FILE.exists():
        with VERIFIED_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["cluster_id"])
                except Exception:
                    pass
    print(f"[verify] already verified: {len(done_ids)}", flush=True)

    # Setup P1 dense retrieval for rank recording
    print("[verify] loading retrieval components...", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    # sentence_transformers MUST be imported before BGEM3Embedder / qdrant_client
    # to ensure PyTorch initialises first — avoids STATUS_ACCESS_VIOLATION on Windows.
    import sentence_transformers  # noqa: F401
    from qdrant_client import QdrantClient
    from medrag.index.embedder import BGEM3Embedder
    qdrant = QdrantClient(url="http://localhost:6333", timeout=30)
    embedder = BGEM3Embedder(device="cpu")
    print("[verify] retrieval ready", flush=True)

    client = _make_client()

    def get_retrieval_ranks(question: str, gold_chunk_ids: list[str]) -> dict:
        """Return {chunk_id: rank} for each gold chunk (None if not in top-k)."""
        enc = embedder.encode([question])
        vec = enc["dense"][0].tolist()
        results = qdrant.query_points(
            collection_name="medrag_text",
            query=vec,
            using="dense",
            limit=retrieval_top_k,
            with_payload=True,
        )
        retrieved_ids = [p.payload.get("chunk_id", "") for p in results.points]
        ranks: dict[str, int | None] = {}
        for cid in gold_chunk_ids:
            ranks[cid] = None
            for j, rid in enumerate(retrieved_ids, 1):
                if rid == cid:
                    ranks[cid] = j
                    break
        return ranks

    def assign_difficulty_band(ranks: dict) -> str:
        """Assign band from worst-case rank among gold chunks."""
        valid_ranks = [r for r in ranks.values() if r is not None]
        if not valid_ranks:
            return "hard"   # not found → hardest
        worst = max(valid_ranks)
        if worst <= EASY_RANK_MAX:
            return "easy"
        if worst >= HARD_RANK_MIN:
            return "hard"
        return "medium"

    def check_faithfulness(chunks: list[dict], claims: list[dict]) -> dict:
        chunks_block = "\n\n".join(
            f"[{ch['chunk_id']}]:\n{ch['text'][:1200]}" for ch in chunks
        )
        try:
            raw_resp = _chat(
                client, model, FAITHFULNESS_SYSTEM,
                FAITHFULNESS_USER_TMPL.format(
                    chunks_block=chunks_block,
                    claims_json=json.dumps(claims, ensure_ascii=False),
                ),
                max_tokens=1000, temp=0.1,
            )
            parsed = _parse_json(raw_resp)
            if parsed:
                faithful = bool(parsed.get("overall_faithful", False))
                conf = float(parsed.get("confidence", 0.5))
                return {
                    "faithful": faithful,
                    "faithfulness_confidence": conf,
                    "faithfulness_issues": str(parsed.get("issues", "")),
                    "faithfulness_pass": faithful and conf >= 0.75,
                }
        except Exception as exc:
            print(f"  [faithfulness] error: {exc}", file=sys.stderr)
        return {"faithful": False, "faithfulness_confidence": 0.0,
                "faithfulness_issues": "api_error", "faithfulness_pass": False}

    def check_parametric(question: str) -> dict:
        try:
            resp = _chat(
                client, model, PARAMETRIC_SYSTEM,
                PARAMETRIC_USER_TMPL.format(question=question),
                max_tokens=400, temp=0.1,
            )
            is_insuff = "INSUFFICIENT" in resp.upper()
            return {"is_parametric": not is_insuff, "parametric_answer": resp[:200]}
        except Exception as exc:
            print(f"  [parametric] error: {exc}", file=sys.stderr)
        return {"is_parametric": False, "parametric_answer": "error"}

    stats = {"leaked": 0, "unfaithful": 0, "parametric": 0, "saved": 0}

    with VERIFIED_FILE.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(raw):
            cid = rec["cluster_id"]
            if cid in done_ids:
                continue

            q      = rec["question"]
            chunks = rec["chunks"]
            claims = rec["claims"]
            gold_ids = list({cl["chunk_id"] for cl in claims})

            print(f"[verify] {i+1}/{len(raw)} | {cid[:60]}", flush=True)

            # 3a. LLM paraphrase check (whether question leaks source terms)
            passages_combined = " ".join(ch["text"] for ch in chunks)
            leak_score = rec.get("lexical_leak_score", _lexical_leak_score(q, passages_combined))
            if leak_score > 0.45:
                # Run LLM paraphrase check for high-risk cases
                try:
                    raw_resp = _chat(
                        client, model, PARAPHRASE_CHECK_SYSTEM,
                        PARAPHRASE_CHECK_USER_TMPL.format(
                            question=q,
                            passages_combined=passages_combined[:2000],
                        ),
                        max_tokens=300, temp=0.1,
                    )
                    pp = _parse_json(raw_resp)
                    has_leak = bool(pp.get("has_leak", False)) if pp else False
                    leaked_phrases = pp.get("leaked_phrases", []) if pp else []
                except Exception:
                    has_leak, leaked_phrases = False, []

                if has_leak:
                    print(f"  [paraphrase] SKIP — verbatim leak: {leaked_phrases[:2]}",
                          flush=True)
                    stats["leaked"] += 1
                    # Still save but mark as leaky so finalize can deprioritize
                    rec["verification"] = {"paraphrase_leak": True, "overall_pass": False,
                                           "leaked_phrases": leaked_phrases}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    time.sleep(sleep_s)
                    continue

            # 3b. P1 retrieval ranks (RECORD, not filter)
            ranks = get_retrieval_ranks(q, gold_ids)
            band  = assign_difficulty_band(ranks)
            print(f"  [retrieval] ranks={ranks} → band={band}", flush=True)

            # 3c. Faithfulness
            time.sleep(sleep_s)
            faith = check_faithfulness(chunks, claims)
            if not faith["faithfulness_pass"]:
                stats["unfaithful"] += 1
                print(f"  [faithful] FAIL conf={faith['faithfulness_confidence']:.2f} "
                      f"| {faith['faithfulness_issues'][:80]}", flush=True)

            # 3d. Parametric check (only if faithful)
            time.sleep(sleep_s)
            if faith["faithfulness_pass"]:
                param = check_parametric(q)
                if param["is_parametric"]:
                    stats["parametric"] += 1
                    print(f"  [parametric] WARN — answerable from general knowledge", flush=True)
            else:
                param = {"is_parametric": False, "parametric_answer": "skipped"}

            overall_pass = faith["faithfulness_pass"] and not param.get("is_parametric", False)
            if overall_pass:
                stats["saved"] += 1

            verified_rec = {
                **rec,
                "verification": {
                    "retrieval_rank_p1": ranks,
                    "difficulty_band": band,
                    "paraphrase_leak": False,
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
    print(f"  processed      : {total}", flush=True)
    print(f"  paraphrase leak: {stats['leaked']}", flush=True)
    print(f"  unfaithful     : {stats['unfaithful']}", flush=True)
    print(f"  parametric     : {stats['parametric']}", flush=True)
    print(f"  overall pass   : {stats['saved']}", flush=True)
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
    print(f"[finalize] loaded {len(verified)} verified items", flush=True)

    passed = [r for r in verified if r["verification"].get("overall_pass", False)]
    print(f"[finalize] overall_pass=True: {len(passed)}", flush=True)

    if len(passed) < n_final:
        print(f"[finalize] WARNING: only {len(passed)} passed, need {n_final}. "
              f"Including borderline items.", file=sys.stderr, flush=True)
        borderline = [
            r for r in verified
            if not r["verification"].get("overall_pass")
            and not r["verification"].get("paraphrase_leak")
            and r["verification"].get("faithfulness_confidence", 0) >= 0.65
        ]
        borderline.sort(key=lambda r: -r["verification"].get("faithfulness_confidence", 0))
        passed = passed + borderline

    # ── Three-dimensional balancing: category × question_difficulty × band ─────
    rng = random.Random(seed)

    # Group by (category, difficulty, difficulty_band)
    groups: dict[tuple, list] = defaultdict(list)
    for r in passed:
        v = r["verification"]
        band = v.get("difficulty_band", "medium")
        key  = (r["category"], r["difficulty"], band)
        groups[key].append(r)

    # Compute quota per cell (proportional)
    total_cat  = sum(CATEGORY_WEIGHTS.values())
    total_diff = sum(DIFFICULTY_WEIGHTS.values())
    total_band = sum(BAND_WEIGHTS.values())

    quotas: dict[tuple, float] = {}
    for cat, cw in CATEGORY_WEIGHTS.items():
        for diff, dw in DIFFICULTY_WEIGHTS.items():
            for band, bw in BAND_WEIGHTS.items():
                q = n_final * (cw / total_cat) * (dw / total_diff) * (bw / total_band)
                quotas[(cat, diff, band)] = q

    selected: list[dict] = []

    # Round 1: fill floor(quota) per cell, prefer non-parametric + high faithfulness
    remaining: dict[tuple, list] = {}
    for key, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda r: (
                int(r["verification"].get("is_parametric", False)),
                -r["verification"].get("faithfulness_confidence", 0),
            ),
        )
        remaining[key] = items_sorted

    for key, quota in sorted(quotas.items(), key=lambda x: -x[1]):
        take = int(quota)
        pool = remaining.get(key, [])
        for item in pool[:take]:
            selected.append(item)
        remaining[key] = pool[take:]

    # Round 2: fill remaining slots greedily
    leftover = [item for pool in remaining.values() for item in pool]
    rng.shuffle(leftover)
    for item in leftover:
        if len(selected) >= n_final:
            break
        selected.append(item)

    selected = selected[:n_final]
    rng.shuffle(selected)

    # ── Print balance summary ──────────────────────────────────────────────────
    from collections import Counter
    print(f"\n[finalize] selected {len(selected)} questions:", flush=True)
    cats  = Counter(r["category"]   for r in selected)
    diffs = Counter(r["difficulty"] for r in selected)
    bands = Counter(r["verification"].get("difficulty_band", "?") for r in selected)
    print(f"  Categories      : {dict(cats)}", flush=True)
    print(f"  Q-Difficulties  : {dict(diffs)}", flush=True)
    print(f"  Retrieval bands : {dict(bands)}", flush=True)
    n_param = sum(1 for r in selected if r["verification"].get("is_parametric"))
    print(f"  Parametric      : {n_param}", flush=True)
    avg_faith = sum(r["verification"].get("faithfulness_confidence", 0)
                    for r in selected) / len(selected)
    print(f"  Avg faithfulness: {avg_faith:.2f}", flush=True)

    # ── Write markdown ─────────────────────────────────────────────────────────
    md_lines = [
        "# MedRAG-Agent Golden Dataset v2\n\n",
        "> Multi-claim, clinically-paraphrased, difficulty-spread.\n",
        "> Each question synthesises ≥2 facts from ≥2 chunks of the same document.\n",
        "> Difficulty bands: easy (P1 rank ≤3) / medium (rank 4–15) / hard (rank ≥16).\n\n",
        "---\n\n",
    ]
    for idx, rec in enumerate(selected, 1):
        qid = f"Q{idx:03d}"
        v   = rec["verification"]
        band = v.get("difficulty_band", "?")
        faith = v.get("faithfulness_confidence", 0)
        gold_ids = [cl["chunk_id"] for cl in rec.get("claims", [])]
        ranks = v.get("retrieval_rank_p1", {})
        md_lines += [
            f"## {qid}\n\n",
            f"**Category**: {rec['category']}  |  "
            f"**Q-Difficulty**: {rec['difficulty']}  |  "
            f"**Retrieval band**: {band}\n\n",
            f"**Question**: {rec['question']}\n\n",
            f"**Answer**: {rec['answer']}\n\n",
            "**Claims**:\n",
        ]
        for cl in rec.get("claims", []):
            md_lines.append(f"  - [{cl['chunk_id']}] {cl['text']}\n")
        md_lines += [
            f"\n**Source**: {rec['source']}:{rec['doc_id']}  |  "
            f"faithfulness={faith:.2f}  |  P1 ranks={ranks}\n\n",
            "---\n\n",
        ]
    FINAL_MD.write_text("".join(md_lines), encoding="utf-8")
    print(f"[finalize] written → {FINAL_MD}", flush=True)

    # ── Write JSONL ────────────────────────────────────────────────────────────
    with FINAL_JSONL.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(selected, 1):
            qid  = f"Q{idx:03d}"
            v    = rec["verification"]
            gold_chunk_ids = list({cl["chunk_id"] for cl in rec.get("claims", [])})
            entry = {
                "id":                      qid,
                "category":                rec["category"],
                "difficulty":              rec["difficulty"],
                "difficulty_band":         v.get("difficulty_band", "medium"),
                "question":                rec["question"],
                "answer":                  rec["answer"],
                "gold_chunk_ids":          gold_chunk_ids,
                "claims":                  rec.get("claims", []),
                "source":                  rec["source"],
                "doc_id":                  rec["doc_id"],
                # backward compat for 08_eval_retrieval.py
                "source_chunk_id":         gold_chunk_ids[0] if gold_chunk_ids else "",
                "retrieval_rank_p1":       v.get("retrieval_rank_p1", {}),
                "faithfulness_confidence": v.get("faithfulness_confidence", 0.0),
                "is_parametric":           v.get("is_parametric", False),
                "notes":                   rec.get("notes", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[finalize] written → {FINAL_JSONL}", flush=True)

    # Write SHA256
    import hashlib
    digest = hashlib.sha256(FINAL_JSONL.read_bytes()).hexdigest()
    sha_file = FINAL_JSONL.with_suffix(FINAL_JSONL.suffix + ".sha256")
    sha_file.write_text(f"{digest}  golden_dataset.jsonl\n", encoding="utf-8")
    print(f"[finalize] checksum → {sha_file}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",
                    choices=["sample", "generate", "verify", "finalize", "all"],
                    default="all")
    ap.add_argument("--n-candidates", type=int, default=150,
                    help="Number of chunk clusters to sample (phase: sample)")
    ap.add_argument("--n-final",      type=int, default=50,
                    help="Target number of golden questions (phase: finalize)")
    ap.add_argument("--model",        default="mimo-v2.5-pro",
                    help="MiMo model name")
    ap.add_argument("--sleep",        type=float, default=1.0,
                    help="Seconds to sleep between API calls")
    ap.add_argument("--retrieval-top-k", type=int, default=50,
                    help="Top-k for P1 rank recording in verify phase")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

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
            phase_verify(args.model,
                         retrieval_top_k=args.retrieval_top_k,
                         sleep_s=args.sleep)
        elif phase == "finalize":
            phase_finalize(args.n_final, seed=args.seed)


if __name__ == "__main__":
    main()
