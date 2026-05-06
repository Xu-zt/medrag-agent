# Week 4 Tutorial: Evaluation Framework & Deployment

## Overview

Week 4 closes the project loop: we take the 50-question Golden Dataset generated in Week 3
and use it to rigorously measure how well each retrieval pipeline performs, then package
the entire system for deployment.

**Deliverables**
| Script / File | Purpose |
|---|---|
| `scripts/08_eval_retrieval.py` | Recall@K + MRR@20 for P1–P5 |
| `scripts/09_eval_answer.py` | Faithfulness / Relevance / Correctness (MiMo judge) |
| `scripts/10_eval_report.py` | Combined Markdown report |
| `docker-compose.yml` + `Dockerfile` | One-command local deployment |
| `data/eval/eval_report.md` | Final evaluation report |

---

## Part 1 — Why These Metrics?

### 1.1 Retrieval: Recall@K and MRR

A RAG system has two failure modes:
1. **Retrieval failure** — the right document is never fetched.
2. **Generation failure** — the right document was fetched but the LLM ignored it.

We measure retrieval separately from generation so we can diagnose *which* component fails.

**Recall@K**: does the gold source chunk appear in the top-K results?

```
Recall@K = (# questions where source_chunk rank ≤ K) / (# questions total)
```

We use K ∈ {1, 3, 5, 10, 20}:
- R@1 → exact-match precision (is the very first result the right one?)
- R@5 → practical LLM context window (we send top-5 to the generator)
- R@20 → upper bound (did we even retrieve it at all?)

**MRR@20** (Mean Reciprocal Rank):

```
MRR@20 = (1/N) × Σ  1 / rank_i      (rank_i = 21 if not found)
```

MRR rewards systems that rank the source chunk higher, not just find it somewhere in the top-20.

### 1.2 Answer Quality: Three-Dimensional Scoring

Even when retrieval succeeds, the generator can produce hallucinated or irrelevant answers.
We use MiMo-V2.5-Pro as a judge (LLM-as-a-judge pattern) across three dimensions:

| Dimension | What it measures | Prompt design |
|---|---|---|
| **Faithfulness** | Every claim grounded in retrieved chunks? | Compare generated answer vs context, ignore medical ground truth |
| **Relevance** | Does the answer address the question? | Compare question vs answer only |
| **Correctness** | Does it match the golden reference answer? | EQUIV comparison: generated vs golden |

**Composite score** = arithmetic mean of all three (0–1 scale).

This separates retrieval-caused failures (faithfulness low despite good context)
from generation failures (relevance/correctness low despite faithful grounding).

---

## Part 2 — Evaluation Architecture

### 2.1 Subprocess Isolation (same pattern as Week 3)

`08_eval_retrieval.py` runs each pipeline in a separate subprocess to avoid OOM
when BGE-M3 + BGE-Reranker are both in memory simultaneously:

```
main process (lightweight)
    ├── subprocess: P1 worker (BGE-M3 dense only)  → stdout JSON
    ├── subprocess: P2 worker (BGE-M3 + BM25)      → stdout JSON
    └── subprocess: P3 worker (BGE-M3 + Reranker)  → stdout JSON
```

Each worker:
1. Loads its models
2. Runs all 50 questions
3. Records `rank` of the source chunk in top-20
4. Prints JSON to stdout, exits

The main process parses the JSON and computes Recall@K / MRR.

### 2.2 Auto-detection of conda Python

The script automatically finds the right Python executable:

```python
def _find_python_exe() -> str:
    candidates = [sys.executable,
                  ~/.conda/envs/medrag/python.exe, ...]
    for exe in candidates:
        if Path(exe).exists():
            result = subprocess.run([exe, "-c", "import qdrant_client"], ...)
            if result.returncode == 0:
                return exe
```

This is necessary because `qdrant_client` is only installed in the conda env,
while the script itself may be launched from the system Python.

### 2.3 Chunk-ID Matching

Each golden question has a `source_chunk_id` (e.g. `pubmed:41962469:0`).
After retrieval we check whether this ID appears in the returned chunk payloads:

```python
retrieved_ids = [c.payload.get("chunk_id", "") for c in chunks]
rank = next((j+1 for j, cid in enumerate(retrieved_ids) if cid == source_id), None)
```

This exact-match approach is strict but unambiguous — partial credit is not given.

### 2.4 Answer Evaluation: Resume-Safe

`09_eval_answer.py` saves results incrementally after each question and tracks
already-evaluated IDs, so it can be safely interrupted and restarted:

```python
done = {r["id"]: r for r in existing["results"]}   # load already-scored
for item in golden:
    if item["id"] in done:
        continue                                     # skip
    ...
    out_path.write_text(json.dumps(payload), ...)    # save immediately
```

---

## Part 3 — Running the Evaluation

### Full run (all 5 pipelines)

```bash
# Retrieval eval — P1, P2, P3 (fast, no Ollama needed)
PYTHONPATH=src python scripts/08_eval_retrieval.py --pipelines p1,p2,p3

# Retrieval eval — P4, P5 (requires Ollama running)
PYTHONPATH=src python scripts/08_eval_retrieval.py --pipelines p4,p5

# Answer quality eval on P3 (requires Ollama + MiMo API)
PYTHONPATH=src python scripts/09_eval_answer.py --pipeline p3

# Generate combined report
python scripts/10_eval_report.py
```

### Output files

```
data/eval/
  retrieval_eval.json    # per-question ranks + aggregate metrics
  answer_eval.json       # per-question faithfulness/relevance/correctness
  eval_report.md         # human-readable combined report
```

---

## Part 4 — Interpreting Results

### What good numbers look like

| Metric | Poor | Acceptable | Good |
|---|---|---|---|
| R@5 | < 40% | 40–60% | > 60% |
| R@20 | < 60% | 60–80% | > 80% |
| MRR@20 | < 0.30 | 0.30–0.55 | > 0.55 |
| Faithfulness | < 0.60 | 0.60–0.80 | > 0.80 |
| Correctness | < 0.40 | 0.40–0.65 | > 0.65 |

### Common failure patterns

**Low R@1, acceptable R@5** → Dense retrieval finds the document but ranks it 2nd–5th.
Cross-encoder reranking (P3) typically fixes this.

**Low R@20** → The source chunk is simply not retrieved at all. Possible causes:
- Short query → expand with Multi-Query (P5)
- Technical jargon → sparse BM25 fusion helps (P2)
- Chunk granularity mismatch (chunk too small/large)

**High Faithfulness, low Correctness** → Generator correctly cites the retrieved
context, but the context doesn't contain enough information to match the golden answer.
Consider increasing k (retrieve more chunks).

**Low Faithfulness** → Generator is hallucinating. Check whether the question type
(complex mechanism questions) exceeds what the retrieved context can support.

---

## Part 5 — Docker Deployment

### Architecture

```
docker-compose.yml
  ├── qdrant        (port 6333) — vector store, persists in volume
  ├── ollama        (port 11434) — Qwen3-8B inference, auto-pulls on first start
  └── medrag-ui     (port 8501) — Streamlit frontend
```

### One-command startup

```bash
docker compose up -d
# Wait ~5 min for Ollama to pull Qwen3-8B on first run
docker compose logs -f ollama   # watch model download progress
```

Then open http://localhost:8501

### Rebuilding the UI image

```bash
docker compose build ui
docker compose up -d ui
```

### Production considerations

1. **GPU support**: Add `deploy.resources.reservations.devices` to the ollama service
   for NVIDIA GPU passthrough.
2. **Persistent data**: The `qdrant_data` volume survives container restarts.
   On first deployment, re-run the indexing scripts to populate it.
3. **Secrets**: Never commit `.env` with API keys. Use Docker secrets or
   environment variables injected at runtime.

---

## Part 6 — Debugging Lessons

### Lesson 1: chunk_id format consistency

The Qdrant payload `chunk_id` must match what the golden dataset stores as `source_chunk_id`.
Both come from `chunks.jsonl`. If you re-index with a different `chunk_id` scheme,
the evaluation metrics will all be zero (a silent failure that looks like bad retrieval).

**Diagnostic**: print a few retrieved `chunk_id` values and compare to golden `source_chunk_id`.

### Lesson 2: conda vs system Python for subprocesses

`qdrant_client`, `FlagEmbedding`, and related packages are only in the conda env.
Subprocess workers must use the conda Python, not `sys.executable` if that points to
the system Python. The `_find_python_exe()` helper probes each candidate with a quick
`import qdrant_client` check.

### Lesson 3: MiMo reasoning tokens eat into max_tokens

MiMo-V2.5-Pro is a reasoning model: its internal chain-of-thought (reasoning_tokens)
counts against the `max_tokens` budget. With `max_tokens=800`, all tokens were consumed
by reasoning and `message.content` was empty (`finish_reason=length`).

**Fix**: set `max_tokens=3000`. Reasoning uses ~900 tokens, leaving ~2100 for output.
The `_chat()` helper auto-retries with doubled `max_tokens` if it detects an empty response.

### Lesson 4: PowerShell stderr → NativeCommandError

When a Python subprocess writes to stderr (e.g., tqdm progress bars, warnings),
PowerShell captures these as `NativeCommandError` objects and sets `$?=False`.
This is cosmetic — the script is still running correctly.

**Diagnostic**: always check `proc.returncode` (in Python) or the last few lines of
the output file rather than relying on PowerShell's exit code display.
