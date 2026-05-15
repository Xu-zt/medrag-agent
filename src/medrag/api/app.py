"""
VeritasMed FastAPI application.

Start with:
    uvicorn src.medrag.api.app:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from project root before any module reads os.environ
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# sentence_transformers MUST be imported before qdrant_client — importing
# qdrant_client first initialises grpcio's native C++ runtime which conflicts
# with PyTorch's internal threads and causes a segfault when the model is
# loaded later.  The route modules pull in qdrant_client at import time via
# medrag.retrieval.retriever, so we pre-empt that here.
import sentence_transformers  # noqa: F401 — import order matters

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from medrag.api.routes import ask, chunk, corpus, document, history, search

app = FastAPI(
    title="VeritasMed API",
    description="Self-verifying medical literature QA backend",
    version="1.0.0",
)

# ── CORS (allow frontend dev server) ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ───────────────────────────────────────────────────────────────
app.include_router(ask.router)
app.include_router(search.router)
app.include_router(document.router)
app.include_router(chunk.router)
app.include_router(history.router)
app.include_router(corpus.router)

# ── Serve built frontend (production) ───────────────────────────────────────
_DIST = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
