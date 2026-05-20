"""Inspect Qdrant collection (point count + sample payload)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient

from medrag.config import COLLECTION_NAME, qdrant_url

q = QdrantClient(url=qdrant_url(), timeout=10)
info = q.get_collection(COLLECTION_NAME)
print("url:", qdrant_url())
print("collection:", COLLECTION_NAME)
print("points_count:", info.points_count)
pts, _ = q.scroll(COLLECTION_NAME, limit=2, with_payload=True)
print("payload keys:", list(pts[0].payload.keys()))
p = {k: v for k, v in pts[0].payload.items() if k != "text"}
print("non-text payload:", p)
print("text snippet:", pts[0].payload["text"][:120])
