"""Build Qdrant index from all ingested data (PubMed + PMC).

Two-phase design:
  Phase 1 — embed: load chunks, encode with BGE-M3, save dense.npy + chunks.jsonl
  Phase 2 — index: load saved arrays, upsert into Qdrant

Run with --phase=embed, --phase=index, or --phase=all (default).
If Phase 1 was completed previously (dense.npy exists), --phase=all skips re-embedding.
"""
# Windows + CUDA: preload pyarrow before torch to avoid access violation (0xC0000005)
import pyarrow.dataset  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

from medrag.ingest.chunker import chunk_pubmed_record, chunk_pmc_record
from medrag.index.qdrant_setup import create_collection
from medrag.index.indexer import index_chunks

CACHE_DIR = Path("data/index_cache")
DENSE_FILE = CACHE_DIR / "dense.npy"
CHUNKS_FILE = CACHE_DIR / "chunks.jsonl"


def load_chunks() -> list:
    chunks = []
    pubmed_path = Path("data/raw/pubmed/abstracts.jsonl")
    pmc_path = Path("data/raw/pmc/full_texts.jsonl")
    if pubmed_path.exists():
        with pubmed_path.open(encoding="utf-8") as f:
            for line in f:
                chunks.extend(chunk_pubmed_record(json.loads(line)))
        print(f"[index] pubmed chunks: {len(chunks)}", flush=True)
    if pmc_path.exists():
        n = len(chunks)
        with pmc_path.open(encoding="utf-8") as f:
            for doc_idx, line in enumerate(f):
                rec = json.loads(line)
                # BioC XML often omits article-id_pmc; fall back to line index
                if not rec.get("pmcid"):
                    rec["pmcid"] = f"doc{doc_idx}"
                chunks.extend(chunk_pmc_record(rec))
        print(f"[index] pmc chunks: {len(chunks) - n}", flush=True)
    print(f"[index] total chunks: {len(chunks)}", flush=True)
    return chunks


def phase_embed(chunks) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[embed] loading BGE-M3 on cuda...", flush=True)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    print("[embed] model loaded", flush=True)
    texts = [c.text for c in chunks]
    print(f"[embed] encoding {len(texts)} texts ...", flush=True)
    out = model.encode(
        texts,
        batch_size=4,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        max_length=512,
    )
    dense = np.array(out["dense_vecs"], dtype=np.float32)
    print(f"[embed] dense shape: {dense.shape}", flush=True)
    np.save(DENSE_FILE, dense)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    print(f"[embed] saved to {CACHE_DIR}", flush=True)
    return dense


def phase_index(chunks, dense: np.ndarray) -> None:
    print("[qdrant] connecting (timeout=120s)...", flush=True)
    client = QdrantClient(url="http://localhost:6333", timeout=120)
    create_collection(client, "medrag_text", recreate=True)
    print(f"[qdrant] upserting {len(chunks)} points...", flush=True)
    index_chunks(client, chunks, dense, collection="medrag_text", batch=256)
    count = client.count(collection_name="medrag_text").count
    print(f"[done] qdrant points: {count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["embed", "index", "all"], default="all")
    args = parser.parse_args()

    if args.phase in ("embed", "all"):
        if args.phase == "all" and DENSE_FILE.exists():
            print(f"[skip] {DENSE_FILE} already exists, loading from cache", flush=True)
            chunks = load_chunks()
            dense = np.load(DENSE_FILE)
            print(f"[embed] loaded dense shape: {dense.shape}", flush=True)
        else:
            chunks = load_chunks()
            if not chunks:
                raise SystemExit("[error] No chunks found.")
            dense = phase_embed(chunks)

    if args.phase == "index":
        chunks = load_chunks()
        dense = np.load(DENSE_FILE)
        print(f"[index] loaded dense shape: {dense.shape}", flush=True)

    if args.phase in ("index", "all"):
        phase_index(chunks, dense)


if __name__ == "__main__":
    main()
