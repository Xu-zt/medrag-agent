"""Quick CLI demo: retrieve + answer a medical question."""
# Windows + CUDA: preload pyarrow before torch to avoid access violation
import pyarrow.dataset  # noqa: F401

import sys
import io
# Force UTF-8 output on Windows (avoids GBK codec errors with Unicode text)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from medrag.agent.generator import generate_answer
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever


def main(query: str, k: int = 5) -> None:
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")
    retriever = DenseRetriever(qdrant, embedder)

    chunks = retriever.retrieve(query, k=k)
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c.score:.3f}  {c.citation}")
        print(f"    {c.text[:200]}...")
    print("=" * 60)
    print("ANSWER:\n")
    print(generate_answer(query, chunks))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the typical resolution of 3T MRI in clinical practice?"
    main(q)
