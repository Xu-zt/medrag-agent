# VeritasMed

Self-verifying medical literature QA. Retrieves evidence from PubMed and PMC, runs a LangGraph agentic loop to grade relevance and rewrite queries, generates a structured answer, then audits every claim for faithfulness before returning.

**Composite score (50-question golden dataset): 0.818**  
Retrieval recall@5: 100% (P2/P3) · Faithfulness: 90%+ · Answer quality: 0.80+

---

## Architecture

```
Browser (React + TypeScript)
  │  REST (Axios)     WebSocket (streaming events)
  ▼                   ▼
FastAPI :8000
  ├── /api/ask        WebSocket — LangGraph streaming
  ├── /api/search     GET       — hybrid retrieval
  ├── /api/document   GET       — document detail
  ├── /api/chunk      GET       — chunk + context window
  ├── /api/history    GET       — conversation history
  └── /api/corpus     GET       — corpus stats
           │
    LangGraph StateGraph (SqliteSaver → data/checkpoints/agent.db)
           │
    route → retrieve → rerank → grade → [rewrite] → generate → check
           │
    HybridRetriever (BGE-M3 dense, RRF) + BGEReranker (cross-encoder)
           │
    Qdrant :6333  ·  ~186k chunks (PubMed abstracts + PMC full-text)
```

Two LLMs:
- `llm_fast` — route, generate, summarize (low-latency)
- `llm_think` — grade, rewrite, check (deep reasoning, thinking mode)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.12 | conda env `medrag` |
| Node.js ≥ 18 | for the frontend |
| Qdrant | `docker run -d -p 6333:6333 qdrant/qdrant:latest` |
| LLM API key | MiMo (default) or Ollama — see `.env.example` |
| Indexed corpus | run scripts 01–07 to build the vector index |

---

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate medrag
```

```powershell
copy .env.example .env   # Windows
```
```bash
cp .env.example .env     # Unix
```

Edit `.env` — fill in `OPENAI_API_KEY` (MiMo endpoint) and `QDRANT_URL`.

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### 3. Build the index (first time only)

```bash
# With medrag env active:
python scripts/01_download_pubmed.py
python scripts/02_download_pmc.py
python scripts/03_chunk.py
python scripts/04_build_index.py
```

### 4. Run — two terminals

**Terminal 1 — backend**
```powershell
.\start_ui.ps1
# Backend at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Terminal 2 — frontend**
```bash
cd frontend
# First time only:
npm install
# Ensure frontend/.env.local contains: VITE_API_URL=http://localhost:8000
npm run dev
# Frontend at http://localhost:5173
```

Open **http://localhost:5173**.

---

## Dev Workflow

### Regenerate frontend types after API changes

REST types are generated from the FastAPI OpenAPI schema — do not edit `api.gen.ts` by hand:

```bash
python scripts/export_openapi.py        # write openapi.json at project root
cd frontend && npm run generate-types   # openapi-typescript → src/types/api.gen.ts
```

WebSocket event types (`frontend/src/types/ws.ts`) mirror `src/medrag/api/models.py` — keep in sync manually. They are not part of the OpenAPI schema.

### Tests

```bash
pytest tests/ -v
```

### Lint

```bash
ruff check src/
```

---

## MCP Server

VeritasMed also exposes a FastMCP server for Claude Desktop / Claude Code:

```powershell
.\start_mcp.ps1
```

Tools: `search_literature`, `ask_agent`, `evaluate_query`, `search_visual` (stub).  
Security: 5-layer middleware — auth → rate_limit → injection_guard → pii → audit.

---

## Project Structure

```
medrag-agent/
├── src/medrag/
│   ├── agent/
│   │   ├── graph.py        # LangGraph StateGraph + SqliteSaver
│   │   ├── nodes.py        # 10 node functions (route/retrieve/…/summarize)
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── prompts.py      # LLM prompt templates
│   │   ├── llms.py         # Dual-LLM factory (fast + think)
│   │   └── utils.py        # strip_thinking()
│   ├── index/
│   │   ├── embedder.py     # BGEM3Embedder (sentence_transformers, dense 1024-d)
│   │   └── indexer.py      # Qdrant upsert pipeline
│   ├── retrieval/
│   │   ├── hybrid.py       # HybridRetriever (dense + RRF)
│   │   ├── reranker.py     # BGEReranker (CrossEncoder)
│   │   ├── retriever.py    # DenseRetriever + RetrievedChunk
│   │   ├── hyde.py         # HyDERetriever
│   │   └── multi_query.py  # MultiQueryRetriever
│   ├── api/
│   │   ├── app.py          # FastAPI entry point (CORS, import order)
│   │   ├── models.py       # Pydantic models — single source of truth
│   │   └── routes/         # ask, search, document, chunk, history, corpus
│   └── mcp_server/
│       ├── server.py       # FastMCP server + 4 tools
│       └── security/       # 5-layer middleware
├── frontend/
│   ├── src/
│   │   ├── types/
│   │   │   ├── api.gen.ts  # Generated from openapi.json (do not edit)
│   │   │   ├── index.ts    # Re-exports from api.gen.ts
│   │   │   └── ws.ts       # WebSocket event types (hand-maintained)
│   │   ├── hooks/useAgentStream.ts  # WebSocket hook
│   │   ├── store/index.ts  # Zustand global state
│   │   ├── components/     # AgentTimeline, AnswerPanel, EvidencePanel, …
│   │   └── pages/          # AnswerPage, ExplorerPage, DocumentPage
│   ├── .env.local          # VITE_API_URL=http://localhost:8000 (gitignored)
│   └── vite.config.ts      # No proxy — direct CORS requests to :8000
├── data/
│   ├── golden/             # 50-question evaluation dataset
│   ├── eval/               # Evaluation outputs
│   └── checkpoints/        # LangGraph SQLite state
├── scripts/                # Numbered pipeline scripts (01–14)
├── openapi.json            # FastAPI OpenAPI schema (source for api.gen.ts)
├── .env.example            # Required env vars template
├── start_ui.ps1            # One-click backend launcher (Windows)
└── start_mcp.ps1           # One-click MCP server launcher (Windows)
```

---

## Key Design Decisions

**sentence_transformers instead of FlagEmbedding**  
FlagEmbedding's decoder-only reranker triggers a `STATUS_ACCESS_VIOLATION` crash on Windows. Both the embedder (`SentenceTransformer("BAAI/bge-m3")`) and reranker (`CrossEncoder("BAAI/bge-reranker-v2-m3")`) use `sentence_transformers`. Dense retrieval is unaffected; sparse vectors are not produced (dense-only RRF).

**Import order in app.py**  
`import sentence_transformers` must appear before any `qdrant_client` import. On Windows, qdrant_client's gRPC native runtime conflicts with PyTorch if PyTorch loads after it. Pre-importing `sentence_transformers` at startup loads PyTorch first.

**Frontend-backend separation**  
The backend does not serve the frontend. In dev, Vite runs at `:5173` and calls `:8000` via `VITE_API_URL`. In production, deploy them as two separate containers behind a reverse proxy.

**OpenAPI-first types**  
REST types are generated from `openapi.json` via `openapi-typescript`. WebSocket event types (`AgentEvent` discriminated union) are defined in both `models.py` (Pydantic) and `ws.ts` (TypeScript) and kept in sync manually.

---

## Evaluation

| Metric | P2 Hybrid | P3 +Reranker |
|--------|-----------|--------------|
| Recall@5 | 100% | 100% |
| MRR@20 | 1.000 | 1.000 |
| Composite (end-to-end) | — | **0.818** |
| Faithfulness | — | 90%+ |
| Latency (retrieval) | ~0.55 s | ~65 s |

P2 is used for `/api/search` (speed). P3 is used for `/api/ask` (quality).

Full details: [`docs/evaluation_report.md`](docs/evaluation_report.md)
