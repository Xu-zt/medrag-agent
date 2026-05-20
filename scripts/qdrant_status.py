"""Print Qdrant collection status (for setup scripts). Exit 0 if indexed, 1 otherwise."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient

from medrag.config import COLLECTION_NAME, qdrant_url


def main() -> int:
    url = qdrant_url()
    try:
        client = QdrantClient(url=url, timeout=10)
        if not client.collection_exists(COLLECTION_NAME):
            print(f"collection_missing:{COLLECTION_NAME}", flush=True)
            return 1
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(f"ok:{count}", flush=True)
        return 0 if count > 0 else 1
    except Exception as exc:
        print(f"error:{exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
