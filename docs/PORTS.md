# Port & URL reference

| Port | Service | Config |
|------|---------|--------|
| **6333** | Qdrant REST | `.env` → `QDRANT_URL=http://localhost:6333` |
| **8000** | FastAPI backend | `start_dev.ps1 -BackendPort` |
| **5173** | Vite frontend (dev) | `npm run dev` in `frontend/` — config in `vite.config.ts` (do not pass `--port` from PowerShell; PS breaks `--`) |
| **5173→80** | Frontend (Docker Compose) | `docker compose up` |
| **11434** | Ollama (optional) | `LLM_BACKEND=ollama` |

## Startup scripts

| Script | Who | What |
|--------|-----|------|
| `start_dev.ps1` | You (daily dev) | Docker Qdrant + backend + frontend windows |
| `start_setup.ps1` | New machine / beginner | conda (Python only) + `pip install -e .` from `pyproject.toml`, data/index checks, then same as dev |
| `start_mcp.ps1` | Claude MCP | Config help / `mcp dev` / install |

All Qdrant URLs read from `QDRANT_URL` via `medrag.config.qdrant_url()` — do not hardcode ports in new code.

PowerShell scripts always resolve **conda env `medrag`** via `conda run -n medrag` (not the currently active `base` env).
