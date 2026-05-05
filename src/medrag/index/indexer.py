"""Upsert chunks with dense vectors into Qdrant."""

from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from medrag.ingest.chunker import Chunk


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vecs: np.ndarray,
    collection: str = "medrag_text",
    batch: int = 256,
) -> None:
    points: list[PointStruct] = []
    for c, vec in zip(chunks, dense_vecs):
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector={"dense": vec.tolist()},
            payload={
                "chunk_id": c.chunk_id,
                "source": c.source,
                "doc_id": c.doc_id,
                "text": c.text,
                **c.metadata,
            },
        ))
        if len(points) >= batch:
            client.upsert(collection_name=collection, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points)


__all__ = ["index_chunks"]
