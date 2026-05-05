# Hardware Smoke Test — Week 1 (Plan B)

| Field | Value |
|---|---|
| Date | 2026-05-04 |
| GPU | NVIDIA GeForce RTX 4060, 8 GB VRAM |
| OS | Windows 11 Home, CUDA 12.4 |
| Python | 3.12.13 (conda-forge) |
| Plan | **Plan B** — embed/rerank on CPU, GPU reserved for LLM |

---

## Step 1 — Ollama + Qwen3-8B

```
ollama list  →  qwen3:8b  5.2 GB  ✅
```

- Status: model downloaded and available
- Approximate VRAM when loaded: ~5.2 GB (Q4_K_M)
- `ollama run qwen3:8b "Explain CT scan in 30 words"` → produces `<think>…</think>` + answer ✅

## Step 2 — BGE-M3 (CPU mode)

```
Python 3.12.13 | packaged by conda-forge
torch 2.6.0+cu124, cuda=True         ✅
transformers 4.57.3                   ✅
FlagEmbedding (loaded)                ✅
BGEM3FlagModel loaded on CPU
dense shape: (2, 1024), dtype: float32
L2-norm of vec 0: 1.0000              ✅
```

- VRAM unchanged during BGE-M3 encoding (CPU only)
- RAM usage: ~2.5 GB for the model

## Step 3 — bge-reranker-v2-m3 (CPU mode)

```
FlagReranker loaded on CPU
rerank score: [-3.81172513961792]     ✅
```

## Step 4 — End-to-end smoke (BGE + reranker on CPU)

Full result from `_smoke_bge_only.py`:

```
[done] BGE-M3 + reranker fully working on CPU
```

All 6 checks passed.

---

## Decision

**Plan B confirmed — main configuration:**

| Component | Device | VRAM |
|---|---|---|
| Qwen3-8B (Q4_K_M) | GPU | ~5.2 GB |
| BGE-M3 embedding | CPU | 0 GB |
| bge-reranker-v2-m3 | CPU | 0 GB |
| KV cache (num_ctx 4096) | GPU | ~0.8-1.3 GB |
| **Total peak** | | **~6.5 GB ✅** |

Indexing phase: can temporarily move BGE-M3 to GPU after `ollama stop qwen3:8b` to speed up.

No downgrade needed (qwen3:8b is primary model).

---

## Windows-specific notes

- pyarrow must be imported before torch on Windows+CUDA to avoid AV (0xC0000005)
- All scripts preload `pyarrow.dataset` at top to prevent crash
- Package managed via conda (`medrag` env, Python 3.12)
