"""Quick CLI demo: retrieve + answer a medical question.

Usage:
    python scripts/quick_demo.py "What is 3T MRI resolution?" --mode p3
    python scripts/quick_demo.py "..." --mode p1   # dense-only (P1)
    python scripts/quick_demo.py "..." --mode p2   # hybrid RRF (P2)
    python scripts/quick_demo.py "..." --mode p3   # hybrid + reranker (P3, default)
    python scripts/quick_demo.py "..." --mode p4   # HyDE (P4)
    python scripts/quick_demo.py "..." --mode p5   # Multi-Query (P5)
"""
# Windows + CUDA: preload pyarrow before torch to avoid access violation
import pyarrow.dataset  # noqa: F401

import argparse
import sys
import io

# Force UTF-8 output on Windows (avoids GBK codec errors with Unicode text)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from medrag.agent.generator import generate_answer
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.retrieval.hyde import HyDERetriever
from medrag.retrieval.multi_query import MultiQueryRetriever


def main(query: str, k: int = 5, mode: str = "p3") -> None:
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")

    extra_info = None

    if mode == "p1":
        retriever = DenseRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        label = "P1: dense-only"

    elif mode == "p2":
        retriever = HybridRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        label = "P2: hybrid RRF"

    elif mode == "p3":
        retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
        candidates = retriever.retrieve(query, k=20)
        reranker = BGEReranker()
        chunks = reranker.rerank(query, candidates, top_k=k)
        label = "P3: hybrid + reranker"

    elif mode == "p4":
        retriever = HyDERetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        extra_info = f"HyDE hypothesis: {retriever.last_hypothesis}"
        label = "P4: HyDE"

    else:  # p5
        retriever = MultiQueryRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        extra_info = "Sub-queries: " + " | ".join(retriever.last_queries)
        label = "P5: Multi-Query"

    print("=" * 60)
    print(f"Query : {query}")
    print(f"Mode  : {label}")
    if extra_info:
        print(f"        {extra_info}")
    print("=" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c.score:.4f}  {c.citation}")
        print(f"    {c.text[:200]}...")
    print("=" * 60)
    print("ANSWER:\n")
    print(generate_answer(query, chunks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    parser.add_argument("--mode", choices=["p1", "p2", "p3", "p4", "p5"], default="p3")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    q = " ".join(args.query) or "What is the typical resolution of 3T MRI in clinical practice?"
    main(q, k=args.k, mode=args.mode)
