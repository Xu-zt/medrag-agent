# MedRAG-Agent — Architecture Reference

> Version: Week 5 (2026-05-06)  
> Stack: LangGraph 0.2 · FastMCP 2.x · MiMo V2.5/V2.5-Pro (API) · BGE-M3 · BGE-Reranker-v2-m3 · Qdrant

---

## 1. System Overview

MedRAG-Agent is a retrieval-augmented generation (RAG) system for medical literature QA. It combines a vector database of PubMed/PMC abstracts with an agentic LangGraph loop that **retrieves, grades, rewrites, generates, and verifies** answers — terminating only when the answer is grounded in the retrieved evidence.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude Desktop / Claude Code                 │
│                              (MCP client)                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  FastMCP 2.x  (stdio / SSE)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MedRAG MCP Server                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Security Middleware (5 layers)                             │   │
│  │  auth → rate_limit → injection_guard → pii → audit         │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │  Tools: search_literature · ask_agent · evaluate_query      │   │
│  │         search_visual (stub)                                 │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph Agentic Loop                            │
│                   (CompiledStateGraph + SqliteSaver)                │
│                                                                     │
│  START → route → retrieve → rerank → grade ──────► generate        │
│                    ▲           │               │         │          │
│                    │      (relevant)      (not rel,      │          │
│                    │           │           iter<2)       │          │
│                    │           ▼               │         ▼          │
│                    └───── rewrite ◄────────────┘      check        │
│                                                          │          │
│                                             (faithful) ──► END      │
│                                        (unfaithful,               │
│                                          regen<1) ──► inc_regen    │
│                                                          │          │
│                                                     ─► generate    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Retrieval Pipelines                                │
│                                                                     │
│  P2 Hybrid: BGE-M3 dense + BGE-M3 sparse ──► RRF fusion            │
│  P3 Reranker: P2 candidates ──► BGE-Reranker cross-encoder         │
│                                                                     │
│  Qdrant (localhost:6333)                                            │
│  Collection: medrag_text                                            │
│  Vectors: dense (1024-d) + sparse (SPLADE-style)                    │
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
| `inc_regen` | — | — | Increment `regen_count` before re-generation loop |
| `append_history` | — | — | Persist completed Q&A turn to `state["history"]` |
| `summarize_gate` | — | — | Passthrough: decide whether to compress history |
| `summarize` | `llm_fast` | OFF | Compress history to ≤200-word rolling summary |

### 2.1 Dual-LLM Strategy

```
llm_fast  (mimo-v2.5, thinking=OFF, temp=0.2, ctx=4096)
  → route, generate, summarize
  → Low latency (~0.5–2 s), deterministic output

llm_think (mimo-v2.5-pro, thinking=ON, temp=0.6, ctx=6144)
  → grade, rewrite, check
  → Deep reasoning (+1–3 s), better at:
     - detecting insufficient context
     - suggesting MeSH-aware rewrites
     - cross-referencing claims vs. chunks
```

### 2.2 Conditional Routing

```
After grade:
  relevance_score ≥ threshold (factual=0.5/synthesis=0.6/multihop=0.7)  →  generate
  score < threshold, iterations < 1  →  rewrite
  score < threshold, iterations ≥ 1  →  generate (best-effort, cap hit)

After check:
  faithful = True              →  append_history  →  summarize_gate  →  END
  unfaithful, first-gen has citations + confidence ≥ 0.3  →  append_history (smart gate)
  unfaithful, regen_count < 1  →  inc_regen  →  generate
  unfaithful, regen_count ≥ 1  →  append_history  →  summarize_gate  →  END
```

---

## 3. Two-Tier Memory Architecture

```
L1 Memory — LangGraph SqliteSaver
  Storage: data/checkpoints/agent.db (SQLite)
  Purpose: crash recovery, multi-turn conversation continuity
  Scope: full AgentState snapshot per step
  Key: thread_id (set by MCP client per user session)

L2 Memory — Rolling Summarisation
  Trigger: every 10 conversation turns (len(history) % 10 == 0)
  LLM: llm_fast (thinking=OFF)
  Output: ≤200-word summary stored in state["summary"]
  Purpose: prevent context-window overflow in long sessions
  Trade-off: loses turn-level detail, preserves medically relevant facts
```

---

## 4. Retrieval Pipeline Details

### 4.1 P2 Hybrid Retrieval (production default)

```
Query
  │
  ├──► BGE-M3 encode (dense 1024-d float32 + sparse SPLADE-style)
  │
  ├──► Qdrant dense search  →  top-20 by cosine similarity
  ├──► Qdrant sparse search →  top-20 by dot product
  │
  └──► RRF fusion (k=60)   →  top-20 fused candidates
```

### 4.2 P3 Hybrid + Reranker (highest quality)

```
P2 output (top-20 candidates)
  │
  └──► BGE-Reranker-v2-m3 (cross-encoder, CPU)
       Input:  [query, chunk_text] pairs
       Output: relevance scores (batch_size=8)
       Result: top-5 by reranker score
```

### 4.3 Evaluation Results (50-question golden dataset)

| Pipeline | R@5 | MRR@20 | Latency |
|----------|-----|--------|---------|
| P1 Dense | 98.0% | 0.963 | 0.48 s |
| P2 Hybrid | **100.0%** | **1.000** | 0.55 s |
| P3 Hybrid+Reranker | **100.0%** | **1.000** | 64.7 s |
| P4 HyDE | 88.0% | 0.810 | 8.97 s |
| P5 Multi-Query | 96.0% | 0.936 | 8.09 s |

**Selected**: P3 for `ask_agent` (quality-critical), P2 for `search_literature` (speed-critical).

---

## 5. Data Flow: `ask_agent` Tool Call

```
1. MCP client sends: ask_agent(query="...", thread_id="t1")

2. Security middleware:
   auth.verify_token()          — check MEDRAG_LOCAL_TOKEN
   rate_limit.check(generate=T) — consume global + generate buckets
   injection_guard.sanitise()   — block injection patterns, escape tokens
   audit.log_tool_call()        — write SHA-256(query) + latency to audit.jsonl

3. LangGraph app.invoke(state, config={"thread_id": "t1"})

4. Node execution:
   route    → classify query type (no external calls)
   retrieve → Qdrant hybrid query (2 HTTP calls, ~0.5 s)
   rerank   → BGE-Reranker inference (~0.2 s GPU / ~15 s CPU for 20 pairs)
   grade    → MiMo-V2.5-Pro thinking call (~1–3 s)
   [rewrite → retrieve → rerank → grade  × up to 1 rewrite]
   generate → MiMo-V2.5 fast call (~1–2 s)
   check    → MiMo-V2.5-Pro thinking call (~1–3 s)
   [inc_regen → generate → check  × 1 regen if unfaithful]
   append_history → persist Q&A turn

5. SqliteSaver persists state snapshot to data/checkpoints/agent.db

6. Return: {answer, citations, confidence, faithful, iterations, regen_count}
```

---

## 6. Corpus Statistics

| Metric | Value |
|--------|-------|
| Sources | PubMed abstracts + PMC full-text (Open Access) |
| Total chunks | ~186,000 |
| Avg chunk length | ~300 tokens |
| Chunk overlap | 64 tokens |
| Embedding model | BAAI/bge-m3 (dense 1024-d + sparse) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Vector DB | Qdrant (single-node, localhost:6333) |

---

## 7. Directory Structure

```
medrag-agent/
├── src/medrag/
│   ├── agent/
│   │   ├── graph.py        # StateGraph assembly + SqliteSaver
│   │   ├── nodes.py        # 10 node functions
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── prompts.py      # All LLM prompt templates
│   │   ├── llms.py         # Dual LLM factory
│   │   ├── generator.py    # Simple baseline generator (Week 1)
│   │   └── utils.py        # strip_thinking()
│   ├── index/
│   │   ├── embedder.py     # BGEM3Embedder (dense + sparse)
│   │   └── indexer.py      # Qdrant upsert pipeline
│   ├── retrieval/
│   │   ├── retriever.py    # DenseRetriever + RetrievedChunk
│   │   ├── hybrid.py       # HybridRetriever (RRF fusion)
│   │   ├── reranker.py     # BGEReranker (cross-encoder)
│   │   ├── hyde.py         # HyDERetriever
│   │   └── multi_query.py  # MultiQueryRetriever
│   └── mcp_server/
│       ├── server.py       # FastMCP server + 4 tools
│       └── security/
│           ├── auth.py           # Token auth
│           ├── rate_limit.py     # Token-bucket limiter
│           ├── audit.py          # JSON-Lines audit log
│           ├── pii.py            # PII redaction
│           └── injection_guard.py # Prompt injection defense
├── data/
│   ├── golden/golden_dataset.jsonl  # 50-question eval set
│   ├── eval/                         # Evaluation outputs
│   └── checkpoints/agent.db          # LangGraph SqliteSaver
├── scripts/                          # Numbered pipeline scripts
└── docs/                             # Architecture, security, tutorials
```

---

*MedRAG-Agent Week 5 — Architecture Reference*
