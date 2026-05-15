# MedRAG-Agent — Architecture Reference

> Stack: LangGraph 0.2 · FastMCP 2.x · MiMo V2.5 / V2.5-Pro (API) · BGE-M3 · BGE-Reranker-v2-m3 · Qdrant · sentence_transformers

---

## 1. System Overview

MedRAG-Agent is a retrieval-augmented generation system for medical literature QA. It combines a vector database of PubMed/PMC abstracts with a LangGraph agentic loop that **retrieves, grades, rewrites, generates, and verifies** answers — terminating only when the answer is grounded in the retrieved evidence.

Two access paths:

```
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│  Browser (React + TypeScript)    │    │  Claude Desktop / Claude Code    │
│  http://localhost:5173           │    │  (MCP client)                    │
└──────────────────┬───────────────┘    └──────────────────┬───────────────┘
                   │  REST / WebSocket                      │  FastMCP 2.x
                   │  http://localhost:8000                 │  (stdio / SSE)
                   ▼                                        ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│  FastAPI Backend                 │    │  MedRAG MCP Server               │
│                                  │    │                                  │
│  /api/ask   WebSocket            │    │  Security Middleware (5 layers)  │
│  /api/search, /document, …       │    │  auth → rate_limit →             │
│  CORSMiddleware (all origins)    │    │  injection_guard → pii → audit   │
└──────────────────┬───────────────┘    └──────────────────┬───────────────┘
                   │                                        │
                   └──────────────┬─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph Agentic Loop                            │
│                   (CompiledStateGraph + SqliteSaver)                │
│                                                                     │
│  START → route → retrieve → rerank → grade ──────► generate        │
│                    ▲           │               │         │          │
│                    │      (relevant)      (not rel,      │          │
│                    │           │           iter<1)       │          │
│                    │           ▼               │         ▼          │
│                    └───── rewrite ◄────────────┘      check        │
│                                                          │          │
│                                             (faithful) ──► END      │
│                                         (unfaithful,               │
│                                           smart gate) ──► END       │
│                                         (unfaithful,               │
│                                           regen<1)  ──► inc_regen  │
│                                                          │          │
│                                                     ─► generate    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Retrieval Pipelines                                │
│                                                                     │
│  P2 Hybrid:  BGE-M3 dense (1024-d) ──► RRF fusion                  │
│  P3 Reranker: P2 candidates ──► BGE-Reranker cross-encoder         │
│                                                                     │
│  Qdrant (localhost:6333) · collection: medrag_text                  │
│  ~186k chunks from PubMed abstracts + PMC full text                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. LangGraph Node Reference

| Node | LLM | Thinking | Responsibility |
|------|-----|----------|----------------|
| `route` | `llm_fast` | OFF | Classify query: factual / synthesis / multihop |
| `retrieve` | — | — | Hybrid RRF retrieval, returns top-20 candidates |
| `rerank` | — | — | BGE cross-encoder → shrink to top-5 |
| `grade` | `llm_think` | ON | Score chunk relevance 0–1; set `rewrite_hint` |
| `rewrite` | `llm_think` | ON | Rewrite failed query (MeSH synonyms, sub-questions) |
| `generate` | `llm_fast` | OFF | Structured JSON answer: {answer, citations, confidence} |
| `check` | `llm_think` | ON | Binary faithfulness audit |
| `inc_regen` | — | — | Increment `regen_count` before re-generation |
| `append_history` | — | — | Persist completed Q&A turn to `state["history"]` |
| `summarize_gate` | — | — | Decide whether to compress history |
| `summarize` | `llm_fast` | OFF | Compress history to ≤200-word rolling summary |

### 2.1 Dual-LLM Strategy

```
llm_fast  (mimo-v2.5, thinking=OFF, temp=0.2)
  → route, generate, summarize
  → Low latency (~0.5–2 s), deterministic output

llm_think (mimo-v2.5-pro, thinking=ON, temp=0.6)
  → grade, rewrite, check
  → Deep reasoning (+1–3 s), better faithfulness auditing
```

Configured via env vars `MIMO_MODEL_FAST` and `MIMO_MODEL_THINK` (or `LLM_BACKEND=ollama`).

### 2.2 Conditional Routing

```
After grade:
  relevance_score ≥ threshold  →  generate
    thresholds: factual=0.5 · synthesis=0.6 · multihop=0.7
  score < threshold, iterations < MAX_REWRITES (1)  →  rewrite
  score < threshold, iterations ≥ MAX_REWRITES      →  generate (cap hit)

After check:
  faithful = True                                    →  append_history → END
  unfaithful, confidence ≥ 0.3 AND has citations    →  append_history → END (smart gate)
  unfaithful, regen_count < MAX_REGEN (1)            →  inc_regen → generate
  unfaithful, regen_count ≥ MAX_REGEN                →  append_history → END
```

Constants (`nodes.py`): `MAX_REWRITES=1`, `MAX_REGEN=1`, `GRADE_THRESHOLD=0.6`, `REGEN_CONFIDENCE_SKIP=0.3`, `CANDIDATE_K=20`, `TOP_K=5`.

---

## 3. Two-Tier Memory Architecture

```
L1 — LangGraph SqliteSaver
  Storage:  data/checkpoints/agent.db (SQLite)
  Purpose:  crash recovery, multi-turn conversation continuity
  Scope:    full AgentState snapshot per step
  Key:      thread_id (set by client per user session)

L2 — Rolling Summarisation
  Trigger:  every 10 conversation turns (HISTORY_SUMMARIZE_EVERY=10)
  LLM:      llm_fast (thinking=OFF)
  Output:   ≤200-word summary in state["summary"]
  Purpose:  prevent context-window overflow in long sessions
```

---

## 4. Retrieval Pipeline

### 4.1 Embedding — sentence_transformers

Both the embedder and reranker use `sentence_transformers` rather than `FlagEmbedding`. FlagEmbedding's decoder-only reranker triggers a `STATUS_ACCESS_VIOLATION` crash on Windows (access violation in `modeling_minicpm_reranker.py`).

```python
# embedder.py
SentenceTransformer("BAAI/bge-m3", device=device)
# Returns dense 1024-d float32 normalized vectors.
# Sparse vectors (SPLADE-style) are NOT produced — dense-only RRF.

# reranker.py
CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
# fp16 on CUDA, float32 on CPU
```

Device is auto-detected (`EMBEDDER_DEVICE` / `RERANKER_DEVICE` env vars, default `auto` → `cuda` if available, else `cpu`).

**Critical import order** (`app.py`): `import sentence_transformers` must appear before any `qdrant_client` import. On Windows, qdrant_client's gRPC C++ runtime conflicts with PyTorch if PyTorch loads after it. Pre-importing `sentence_transformers` at startup loads PyTorch first.

### 4.2 P2 Hybrid Retrieval

```
Query
  │
  └──► BGE-M3 encode (dense 1024-d)
         │
         ├──► Qdrant dense search  →  top-20 by cosine similarity
         │
         └──► RRF fusion (k=60)   →  top-20 fused candidates
```

(Sparse search is skipped when `sentence_transformers` returns empty sparse weights.)

### 4.3 P3 Hybrid + Reranker

```
P2 output (top-20 candidates)
  │
  └──► BGE-Reranker-v2-m3 (cross-encoder, batch_size=8)
       Input:  [query, chunk_text] pairs
       Output: top-5 by reranker score
```

### 4.4 Evaluation Results (50-question golden dataset)

| Pipeline | R@5 | MRR@20 | Latency |
|----------|-----|--------|---------|
| P1 Dense | 98.0% | 0.963 | 0.48 s |
| P2 Hybrid | **100.0%** | **1.000** | 0.55 s |
| P3 Hybrid+Reranker | **100.0%** | **1.000** | ~65 s |
| P4 HyDE | 88.0% | 0.810 | 8.97 s |
| P5 Multi-Query | 96.0% | 0.936 | 8.09 s |

P3 is used for `/api/ask` (quality-critical). P2 is used for `/api/search` (speed-critical).

---

## 5. API Contract

### 5.1 REST — OpenAPI

FastAPI auto-generates `/openapi.json`. TypeScript types are generated from it:

```bash
python scripts/export_openapi.py      # write openapi.json
cd frontend && npm run generate-types  # write src/types/api.gen.ts
```

Key schemas: `ChunkOut`, `SearchResponse`, `DocumentResponse`, `ChunkContextResponse`, `CorpusStats`, `AnswerOut`.

### 5.2 WebSocket — AgentEvent (manual sync)

`/api/ask` is a WebSocket endpoint — not in the OpenAPI schema. Events are a Pydantic discriminated union (`models.py`) mirrored manually in `frontend/src/types/ws.ts`.

```
node_start        { event, node }
node_end          { event, node, data: NodeEndData }
chunk_retrieved   { event, node, data: ChunkRetrievedData }
done              { event: "done", node: null, data: AnswerOut }
error             { event: "error", node: null, data: { message } }
```

Answer arrives once in the `done` event (no incremental token streaming).

---

## 6. Corpus Statistics

| Metric | Value |
|--------|-------|
| Sources | PubMed abstracts + PMC full-text (Open Access) |
| Total chunks | ~186,000 |
| Avg chunk length | ~300 tokens |
| Chunk overlap | 64 tokens |
| Embedding model | BAAI/bge-m3 (dense 1024-d, sentence_transformers) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Vector DB | Qdrant (single-node, localhost:6333) |

---

## 7. Directory Structure

```
medrag-agent/
├── src/medrag/
│   ├── agent/
│   │   ├── graph.py        # StateGraph + SqliteSaver
│   │   ├── nodes.py        # 10 node functions
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── prompts.py      # LLM prompt templates
│   │   ├── llms.py         # Dual-LLM factory
│   │   └── utils.py        # strip_thinking()
│   ├── index/
│   │   ├── embedder.py     # BGEM3Embedder (sentence_transformers)
│   │   └── indexer.py      # Qdrant upsert pipeline
│   ├── retrieval/
│   │   ├── hybrid.py       # HybridRetriever (RRF fusion)
│   │   ├── reranker.py     # BGEReranker (CrossEncoder)
│   │   ├── retriever.py    # DenseRetriever + RetrievedChunk
│   │   ├── hyde.py         # HyDERetriever
│   │   └── multi_query.py  # MultiQueryRetriever
│   └── mcp_server/
│       ├── server.py       # FastMCP server + 4 tools
│       └── security/       # auth, rate_limit, injection_guard, pii, audit
├── frontend/               # React + TypeScript (served separately)
│   ├── src/types/
│   │   ├── api.gen.ts      # Generated from openapi.json — do not edit
│   │   ├── index.ts        # Re-exports
│   │   └── ws.ts           # WebSocket types (manual mirror of models.py)
│   ├── .env.local          # VITE_API_URL=http://localhost:8000 (gitignored)
│   └── vite.config.ts      # No proxy — direct CORS to :8000
├── data/
│   ├── golden/             # 50-question evaluation dataset
│   ├── eval/               # Evaluation outputs
│   └── checkpoints/        # LangGraph SqliteSaver state
├── scripts/                # Numbered pipeline scripts (01–14)
├── openapi.json            # FastAPI OpenAPI schema
├── .env.example            # Required environment variables
├── start_ui.ps1            # Backend launcher (Windows)
└── start_mcp.ps1           # MCP server launcher (Windows)
```
