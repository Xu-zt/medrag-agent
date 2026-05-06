# MedRAG-Agent

> Agentic Medical RAG with Secure MCP Interface · Local-first · GDPR-conscious
>
> 5-week personal project · Status: **Week 5 / 5 — Complete** ✅

## What It Does

MedRAG-Agent retrieves peer-reviewed PubMed/PMC literature and generates grounded answers using a **self-correcting agentic loop**: if retrieved chunks are insufficient, it rewrites the query (up to 2×) and tries again; if the generated answer introduces facts not in the context, it regenerates once. Every answer comes with inline citations and a faithfulness flag.

```
User query
    │
    ▼
[Hybrid Retrieval]  ─── BGE-M3 dense + sparse, RRF fusion
    │
    ▼
[Cross-encoder Rerank]  ─── BGE-Reranker-v2-m3
    │
    ▼
[Grade relevance]  ─── Qwen3-8B (thinking ON)
    │ score < 0.6?
    ▼
[Rewrite query]  ──────────────► loop back (max 2×)
    │ score ≥ 0.6
    ▼
[Generate answer]  ─── Qwen3-8B (thinking OFF), JSON output
    │
    ▼
[Check faithfulness]  ─── Qwen3-8B (thinking ON)
    │ unfaithful?
    ▼
[Regenerate]  ─────────────────► once
    │
    ▼
Answer + citations + confidence + faithful flag
```

---

## Evaluation Results

**Retrieval** (50-question golden dataset, top-5):

| Pipeline | R@5 | MRR@20 | Latency |
|----------|-----|--------|---------|
| P1 Dense | 98.0% | 0.963 | 0.48 s |
| **P2 Hybrid** | **100.0%** | **1.000** | 0.55 s |
| **P3 Hybrid+Reranker** | **100.0%** | **1.000** | 64.7 s |
| P4 HyDE | 88.0% | 0.810 | 8.97 s |
| P5 Multi-Query | 96.0% | 0.936 | 8.09 s |

**Answer quality** (P3, judge: MiMo-V2.5-Pro):

| Dimension | Score |
|-----------|-------|
| Faithfulness | 0.405 |
| Relevance | 0.996 |
| Correctness | 0.916 |
| **Composite** | **0.772** |

---

## Quick Start

### Prerequisites

- NVIDIA GPU ≥ 8 GB VRAM (RTX 4060 tested — BGE-M3 and reranker run on CPU)
- [Ollama](https://ollama.com) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- Python 3.12 via conda

### Setup

```bash
git clone https://github.com/<your-handle>/medrag-agent
cd medrag-agent
conda env create -f environment.yml
conda activate medrag
pip install -e .
```

### Start services

```bash
ollama pull qwen3:8b

# PowerShell (Windows)
docker run -d --name medrag-qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v "D:\Desktop\Agent\medrag-agent\qdrant_storage:/qdrant/storage" `
  qdrant/qdrant:latest
```

### Build corpus index (one-off, ~45–120 min on CPU)

```bash
python scripts/01_ingest_pubmed.py   # ~1500–2000 PubMed abstracts
python scripts/02_ingest_pmc.py      # ~300 PMC OA full texts
python scripts/04_build_index.py     # embed + upsert into Qdrant
```

### Ask a question

```python
# Direct agent API
from medrag.agent.graph import app

config = {"configurable": {"thread_id": "my-session"}}
result = app.invoke({
    "query": "What is the mechanism of action of aspirin?",
    "rewritten_queries": [], "retrieved_chunks": [],
    "relevance_score": 0.0, "grade_reason": "", "rewrite_hint": "",
    "iterations": 0, "answer": "", "citations": [], "confidence": 0.0,
    "faithful": False, "faithfulness_issues": "", "regen_count": 0,
    "history": [], "summary": "",
}, config=config)

print(result["answer"])
print("Citations:", result["citations"])
print("Faithful:", result["faithful"])
```

```bash
# Via MCP server (Claude Desktop / Claude Code)
mcp dev src/medrag/mcp_server/server.py
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `ask_agent` | Full agentic loop: hybrid retrieval → rerank → grade/rewrite → generate → faithfulness check |
| `search_literature` | Fast hybrid retrieval (P2/P3), returns document snippets |
| `evaluate_query` | Grade how well provided context answers a query |
| `search_visual` | Stub for future image/figure search |

---

## Status

- [x] **Week 1**: Dense RAG — PubMed + PMC corpus, Qwen3-8B, BGE-M3 embeddings
- [x] **Week 2**: Hybrid retrieval (dense+sparse RRF) + BGE-Reranker + 50-Q golden dataset
- [x] **Week 3**: 5 retrieval pipelines (Dense, Hybrid, Hybrid+Reranker, HyDE, Multi-Query); retrieval eval P1–P5
- [x] **Week 4**: FastMCP server; LLM-as-judge answer evaluation; Docker compose; eval report
- [x] **Week 5**: LangGraph agentic loop (grade→rewrite→generate→check); 5-layer security middleware; 49 unit tests; architecture + security docs

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system diagram and node reference.

**Key design choices:**
- **Dual-LLM strategy**: `llm_fast` (thinking=OFF) for router + generate; `llm_think` (thinking=ON) for grade + rewrite + check
- **Two-tier memory**: L1 SqliteSaver (crash recovery) + L2 rolling summarisation every 10 turns
- **5-layer security**: auth → rate_limit → injection_guard → pii → audit (see [`docs/mcp_security.md`](docs/mcp_security.md))

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | Qwen3-8B via Ollama (Q4_K_M, ~5.2 GB VRAM) |
| Agent framework | LangGraph 0.2 (StateGraph + SqliteSaver) |
| Embedding | BAAI/bge-m3 (CPU, dense 1024-d + sparse) |
| Reranker | BAAI/bge-reranker-v2-m3 (CPU, cross-encoder) |
| Vector DB | Qdrant (Docker, localhost:6333) |
| MCP server | FastMCP 2.x (stdio transport) |
| Python | 3.12 + conda |

---

## Project Structure

```
src/medrag/
├── agent/
│   ├── graph.py        # LangGraph StateGraph + SqliteSaver
│   ├── nodes.py        # 10 node functions
│   ├── state.py        # AgentState TypedDict
│   ├── prompts.py      # All LLM prompt templates
│   ├── llms.py         # Dual LLM factory (fast/think)
│   └── generator.py    # Baseline generator (Week 1)
├── index/              # BGE-M3 embedder + Qdrant upsert
├── retrieval/          # Dense, Hybrid, Reranker, HyDE, Multi-Query
└── mcp_server/
    ├── server.py       # FastMCP server + 4 tools
    └── security/       # 5-layer middleware
        ├── auth.py
        ├── rate_limit.py
        ├── audit.py
        ├── pii.py
        └── injection_guard.py
data/
├── golden/             # 50-question golden dataset
├── eval/               # Retrieval + answer eval results
└── checkpoints/        # LangGraph SqliteSaver
docs/
├── architecture.md     # System diagram + node reference
└── mcp_security.md     # Threat model + middleware details
tests/
├── test_agent.py       # Graph topology + routing + node tests (19 tests)
└── test_mcp_security.py # Security middleware tests (30 tests)
scripts/
├── 01–07_*.py          # Ingest, index, retrieval pipelines
├── 08_eval_retrieval.py
├── 09_eval_answer.py
└── 10_smoke_test_agent.py
```

## License

Apache 2.0
