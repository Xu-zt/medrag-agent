"""Project-wide configuration helpers (env vars, paths)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/medrag/config.py → repo root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "medrag_text"


def load_project_env() -> None:
    """Load .env from repo root (idempotent)."""
    load_dotenv(PROJECT_ROOT / ".env")


def qdrant_url() -> str:
    """Qdrant REST base URL from QDRANT_URL env var."""
    load_project_env()
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")
