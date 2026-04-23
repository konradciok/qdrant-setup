"""QdrantUpserter — embed and upsert vault chunks into Qdrant.

Public API
----------
upsert_chunks(client, dense_embedder, bm25_embedder, doc, chunks) -> None
    Embed all chunks for a document and batch-upsert them.
    Skips the document if its doc_hash matches the stored hash.

delete_orphans(client, active_file_paths) -> int
    Scroll all points; delete those whose file_path is not in active_file_paths.
    Returns the count of deleted points.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from os.path import dirname
from typing import TYPE_CHECKING

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from vault_qdrant.collection import VAULT_COLLECTION
from vault_qdrant.embedder import BM25Embedder, DenseEmbedder

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

_SCROLL_BATCH = 100


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def _chunk_id(file_path: str, h2: str | None, h3: str | None, chunk_index: int) -> str:
    """Return a deterministic 32-char hex ID from SHA-256(file_path + h2 + h3 + chunk_index)."""
    raw = (file_path + (h2 or "") + (h3 or "") + str(chunk_index)).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Hash check
# ---------------------------------------------------------------------------


def _existing_doc_hash(client: "QdrantClient", file_path: str) -> str | None:
    """Return the stored doc_hash for file_path, or None if not found."""
    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_path),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return points[0].payload.get("doc_hash")


# ---------------------------------------------------------------------------
# Point construction
# ---------------------------------------------------------------------------


def _build_point(
    doc: dict,
    chunk: dict,
    dense_vector: list[float],
    sparse_vector,
    modified_at: str,
) -> PointStruct:
    """Build a single PointStruct from doc + chunk + embeddings."""
    return PointStruct(
        id=_chunk_id(
            doc["file_path"],
            chunk.get("h2"),
            chunk.get("h3"),
            chunk.get("chunk_index", 0),
        ),
        vector={
            "fast-bge-large-en-v1.5": dense_vector,
            "sparse": sparse_vector,
        },
        payload={
            "file_path": doc["file_path"],
            "folder": dirname(doc["file_path"]) or ".",
            "doc_type": doc.get("type"),
            "type_source": doc.get("type_source", "inferred"),
            "tags": doc.get("tags", []),
            "modified_at": modified_at,
            "status": doc.get("status"),
            "h1": chunk.get("h1"),
            "h2": chunk.get("h2"),
            "h3": chunk.get("h3"),
            "forward_links": chunk.get("forward_links", []),
            "chunk_index": chunk.get("chunk_index"),
            "is_title_chunk": chunk.get("chunk_index", 0) == 0,
            "doc_hash": doc["doc_hash"],
            "text": chunk.get("text", ""),
        },
    )


# ---------------------------------------------------------------------------
# Public: upsert_chunks
# ---------------------------------------------------------------------------


def upsert_chunks(
    client: "QdrantClient",
    dense_embedder: DenseEmbedder,
    bm25_embedder: BM25Embedder,
    doc: dict,
    chunks: list[dict],
) -> None:
    """Embed all chunks for a document and upsert them into Qdrant.

    Skips the document entirely when the stored doc_hash matches doc['doc_hash'].

    Args:
        client: Connected QdrantClient instance.
        dense_embedder: Dense embedding model.
        bm25_embedder: Sparse BM25 embedding model.
        doc: Scanner record (file_path, content, tags, type, status, doc_hash, ...).
        chunks: Chunker records (text, h1, h2, h3, forward_links, chunk_index).
    """
    existing_hash = _existing_doc_hash(client, doc["file_path"])
    if existing_hash == doc["doc_hash"]:
        return

    modified_at = datetime.now(tz=timezone.utc).isoformat()

    points: list[PointStruct] = []
    for chunk in chunks:
        dense = dense_embedder.embed(chunk["text"])
        sparse = bm25_embedder.embed(chunk["text"])
        point = _build_point(doc, chunk, dense, sparse, modified_at)
        points.append(point)

    client.upsert(
        collection_name=VAULT_COLLECTION,
        points=points,
    )


# ---------------------------------------------------------------------------
# Public: delete_orphans
# ---------------------------------------------------------------------------


def delete_orphans(client: "QdrantClient", active_file_paths: set[str]) -> int:
    """Delete all Qdrant points whose file_path is not in active_file_paths.

    Args:
        client: Connected QdrantClient instance.
        active_file_paths: Set of currently active vault file paths.

    Returns:
        Number of points deleted.
    """
    orphan_ids: list[str] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=VAULT_COLLECTION,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            file_path = point.payload.get("file_path")
            if file_path not in active_file_paths:
                orphan_ids.append(point.id)

        if offset is None:
            break

    if orphan_ids:
        client.delete(
            collection_name=VAULT_COLLECTION,
            points_selector=PointIdsList(points=orphan_ids),
        )

    return len(orphan_ids)
