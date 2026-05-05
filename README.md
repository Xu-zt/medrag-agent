# MedRAG-Agent

> Agentic Medical RAG with secure MCP Interface · Local-first · GDPR-compliant
>
> 5-week personal project · Status: **Week 1 / 5** — basic dense RAG

## Status

- [x] Week 1: Basic dense RAG — PubMed + PMC corpus, local Qwen3-8B, BGE-M3 embeddings
- [ ] Week 2: Hybrid retrieval (dense + sparse) + reranker + 50-Q golden dataset
- [ ] Week 3: LangGraph agentic loop with self-correction + session memory
- [ ] Week 4: FastMCP server with 5-layer security middleware
- [ ] Week 5: Full evaluation pipeline, demo video, report

## Quick Start (Week 1 baseline)

### Prerequisites

- NVIDIA GPU (≥ 8 GB VRAM recommended)
- [Ollama](https://ollama.com) installed
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- Python 3.12 via conda or uv

### Setup

```bash
# 1. Clone and set up Python environment
git clone https://github.com/<your-handle>/medrag-agent
cd medrag-agent

# conda path
conda env create -f environment.yml
conda activate medrag
pip install -e .

# uv path (alternative)
# uv venv && source .venv/Scripts/activate  # Windows: .venv\Scripts\Activate.ps1
# uv sync
```

### Run services

```bash
# Pull LLM
ollama pull qwen3:8b

# Start Qdrant (bash / Git Bash)
docker run -d --name medrag-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ${PWD}/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Windows PowerShell equivalent:
# docker run -d --name medrag-qdrant `
#   -p 6333:6333 -p 6334:6334 `
#   -v "D:\Desktop\Agent\medrag-agent\qdrant_storage:/qdrant/storage" `
#   qdrant/qdrant:latest
```

### Ingest + index (one-off, ~45-120 min)

```bash
python scripts/01_ingest_pubmed.py   # ~1500-2000 PubMed abstracts
python scripts/02_ingest_pmc.py      # ~300 PMC OA full texts
python scripts/04_build_index.py     # embed + upsert into Qdrant (CPU, Plan B)
```

### Ask a question

```bash
python scripts/quick_demo.py "What is the typical resolution of 3T MRI?"
python scripts/quick_demo.py "What contrast agents are used for cardiac MRI?"
```

## Tech Stack (Week 1)

| Component | Choice | Notes |
|---|---|---|
| LLM | Qwen3-8B via Ollama (Q4_K_M) | ~5.2 GB VRAM, hybrid thinking |
| Embedding | BGE-M3 (FlagEmbedding) | **CPU** — leaves VRAM for LLM |
| Vector DB | Qdrant (Docker, local) | Dense cosine search |
| Data sources | PubMed abstracts + PMC OA full texts | ~13k chunks |
| Python | 3.12 + conda | pyarrow preload required on Windows+CUDA |

## Hardware (Plan B)

RTX 4060 8 GB VRAM — embed/rerank on CPU, GPU reserved for Qwen3-8B.

| Component | VRAM |
|---|---|
| Qwen3-8B Q4_K_M | ~5.2 GB |
| BGE-M3 (CPU) | 0 GB |
| KV cache (4096 ctx) | ~0.8 GB |
| **Total** | **~6.0 GB ✅** |

## Project Structure

```
src/medrag/
├── ingest/       pubmed.py, pmc.py, chunker.py
├── index/        embedder.py, qdrant_setup.py, indexer.py
├── retrieval/    retriever.py
├── agent/        generator.py, utils.py
├── mcp_server/   (Week 4)
└── eval/         (Week 5)
scripts/
├── 01_ingest_pubmed.py
├── 02_ingest_pmc.py
├── 04_build_index.py
└── quick_demo.py
docs/
└── hardware_smoke_test.md
```

## Roadmap

See `MedRAG-Agent_项目书_v1_1.md` for the full design document.

## License

Apache 2.0
