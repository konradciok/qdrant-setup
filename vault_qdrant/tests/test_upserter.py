"""Tests for QdrantUpserter — written BEFORE implementation (TDD).

Uses unittest.mock to mock QdrantClient; no real Qdrant connection needed.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import SparseVector

from vault_qdrant.upserter import _chunk_id, delete_orphans, upsert_chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_doc() -> dict:
    """Minimal scanner document record."""
    return {
        "file_path": "projects/alpha.md",
        "content": "# Alpha\nSome content.",
        "tags": ["project", "alpha"],
        "type": "project",
        "created": "2024-01-01",
        "status": "active",
        "projects": ["medusa"],
        "doc_hash": "abc123deadbeef",
    }


@pytest.fixture()
def sample_chunks() -> list[dict]:
    """Two minimal chunker records."""
    return [
        {
            "text": "Alpha intro text",
            "h1": "Alpha",
            "h2": "Overview",
            "h3": None,
            "forward_links": ["beta.md"],
            "chunk_index": 0,
        },
        {
            "text": "Alpha details",
            "h1": "Alpha",
            "h2": "Details",
            "h3": "Sub",
            "forward_links": [],
            "chunk_index": 1,
        },
    ]


@pytest.fixture()
def mock_client() -> MagicMock:
    """QdrantClient mock that returns no existing scroll results by default."""
    client = MagicMock()
    # scroll returns (points, next_page_offset); empty by default
    client.scroll.return_value = ([], None)
    return client


@pytest.fixture()
def mock_dense_embedder() -> MagicMock:
    """DenseEmbedder mock returning a 1024-dim dense vector."""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 1024
    return embedder


@pytest.fixture()
def mock_bm25_embedder() -> MagicMock:
    """BM25Embedder mock returning a minimal SparseVector."""
    embedder = MagicMock()
    embedder.embed.return_value = SparseVector(indices=[1, 2], values=[0.5, 0.3])
    return embedder


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _expected_id(file_path: str, h2: str | None, h3: str | None, chunk_index: int = 0) -> str:
    """Deterministic chunk ID as first-32-chars of SHA-256(file_path + h2 + h3 + chunk_index)."""
    raw = (file_path + (h2 or "") + (h3 or "") + str(chunk_index)).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Test 1: upsert_chunks calls client.upsert with correct collection name
# ---------------------------------------------------------------------------


def test_upsert_calls_upsert_points(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """upsert_chunks() must call client.upsert() with the vault collection name."""
    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )

    assert mock_client.upsert.called, "client.upsert() was never called"
    upsert_call_args = mock_client.upsert.call_args
    assert upsert_call_args is not None

    kwargs = upsert_call_args.kwargs
    args = upsert_call_args.args
    collection_name = kwargs.get("collection_name") or (args[0] if args else None)
    assert collection_name == "vault", f"Expected 'vault', got {collection_name!r}"


# ---------------------------------------------------------------------------
# Test 2: upserted points contain all required payload fields
# ---------------------------------------------------------------------------


def test_upsert_payload_fields(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """Each upserted point must carry the required payload fields."""
    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )

    assert mock_client.upsert.called
    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[1]

    required_fields = {
        "file_path",
        "doc_type",
        "tags",
        "folder",
        "modified_at",
        "status",
        "h1",
        "h2",
        "h3",
        "forward_links",
        "chunk_index",
        "doc_hash",
    }

    assert points, "No points were upserted"
    for point in points:
        missing = required_fields - set(point.payload.keys())
        assert not missing, f"Point missing payload fields: {missing}"


# ---------------------------------------------------------------------------
# Test 3: each point has dense and sparse named vectors
# ---------------------------------------------------------------------------


def test_upsert_has_dense_and_sparse_vectors(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """Each upserted point must have a 'dense' named vector and a 'sparse' sparse vector."""
    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )

    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[1]

    assert points
    for point in points:
        vectors = point.vector
        assert "fast-bge-large-en-v1.5" in vectors, "Missing 'fast-bge-large-en-v1.5' vector"
        assert "sparse" in vectors, "Missing 'sparse' vector"


# ---------------------------------------------------------------------------
# Test 4: chunk ID is SHA-256(file_path + h2)[:32]
# ---------------------------------------------------------------------------


def test_chunk_id_is_sha256_of_file_path_and_h2(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """Point IDs must be deterministic SHA-256 hashes of file_path + h2 + h3."""
    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )

    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[1]

    for point, chunk in zip(points, sample_chunks):
        expected = _expected_id(sample_doc["file_path"], chunk["h2"], chunk.get("h3"), chunk.get("chunk_index", 0))
        assert point.id == expected, (
            f"Expected ID {expected!r}, got {point.id!r} for h2={chunk['h2']!r} h3={chunk.get('h3')!r}"
        )

    # Two chunks sharing the same H2 but with different H3 must produce different IDs
    id_same_h2_no_h3 = _expected_id("file.md", "Section", None)
    id_same_h2_with_h3 = _expected_id("file.md", "Section", "SubSection")
    assert id_same_h2_no_h3 != id_same_h2_with_h3, (
        "Chunks with same H2 but different H3 must not collide"
    )


# ---------------------------------------------------------------------------
# Test 5: skip upsert when existing doc_hash matches
# ---------------------------------------------------------------------------


def test_skip_unchanged_doc_hash(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """If existing Qdrant point has the same doc_hash, upsert must be skipped."""
    existing_point = MagicMock()
    existing_point.payload = {"doc_hash": sample_doc["doc_hash"]}
    mock_client.scroll.return_value = ([existing_point], None)

    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )

    mock_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: delete_orphans removes points whose file_path is not in active set
# ---------------------------------------------------------------------------


def test_delete_orphan_points(mock_client):
    """delete_orphans() must call client.delete() for orphaned file_paths."""
    active_paths = {"projects/alpha.md", "projects/beta.md"}

    # First scroll page: two points, one orphan
    p1 = MagicMock()
    p1.id = "id-orphan-1"
    p1.payload = {"file_path": "archive/old.md"}  # orphan

    p2 = MagicMock()
    p2.id = "id-active-1"
    p2.payload = {"file_path": "projects/alpha.md"}  # active

    # Second scroll page: another orphan
    p3 = MagicMock()
    p3.id = "id-orphan-2"
    p3.payload = {"file_path": "deleted/gone.md"}  # orphan

    # Two scroll pages then done
    mock_client.scroll.side_effect = [
        ([p1, p2], "cursor-1"),
        ([p3], None),
    ]

    count = delete_orphans(mock_client, active_paths)

    assert mock_client.delete.called, "client.delete() was never called"
    assert count == 2, f"Expected 2 orphans deleted, got {count}"

    # Verify the delete call included orphan IDs (not active ones)
    deleted_ids: set[str] = set()
    for c in mock_client.delete.call_args_list:
        points_selector = c.kwargs.get("points_selector") or (
            c.args[1] if len(c.args) > 1 else None
        )
        if points_selector is not None:
            ids = getattr(points_selector, "points", None)
            if ids:
                deleted_ids.update(ids)

    assert "id-orphan-1" in deleted_ids
    assert "id-orphan-2" in deleted_ids
    assert "id-active-1" not in deleted_ids


def test_duplicate_headings_different_chunk_index_no_collision() -> None:
    """Two chunks with identical h2/h3 but different chunk_index must produce different IDs."""
    id_first = _chunk_id("notes.md", "Notes", None, 0)
    id_second = _chunk_id("notes.md", "Notes", None, 1)
    assert id_first != id_second, (
        "Chunks with duplicate headings but different chunk_index must not collide"
    )


def test_chunk_id_includes_chunk_index() -> None:
    """chunk_index alone must change the resulting ID."""
    id_zero = _chunk_id("file.md", "Section", "Sub", 0)
    id_one = _chunk_id("file.md", "Section", "Sub", 1)
    assert id_zero != id_one


def test_first_chunk_is_title_chunk(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
):
    """chunk_index=0 must have is_title_chunk=True in payload."""
    upsert_chunks(
        mock_client, mock_dense_embedder, mock_bm25_embedder, sample_doc, sample_chunks
    )
    points = mock_client.upsert.call_args.kwargs.get("points") or \
             mock_client.upsert.call_args.args[1]

    title_chunk = next(p for p in points if p.payload.get("chunk_index") == 0)
    other_chunk = next(p for p in points if p.payload.get("chunk_index") == 1)

    assert title_chunk.payload["is_title_chunk"] is True
    assert other_chunk.payload["is_title_chunk"] is False


def test_type_source_propagated_to_payload(
    mock_client, mock_dense_embedder, mock_bm25_embedder, sample_chunks
):
    """type_source from doc must appear in every chunk payload."""
    doc = {
        "file_path": "projects/alpha.md",
        "content": "content",
        "tags": [],
        "type": "project",
        "type_source": "inferred",
        "created": None,
        "status": None,
        "projects": [],
        "doc_hash": "xyzabc",
    }
    upsert_chunks(mock_client, mock_dense_embedder, mock_bm25_embedder, doc, sample_chunks)
    points = mock_client.upsert.call_args.kwargs.get("points") or \
             mock_client.upsert.call_args.args[1]

    for p in points:
        assert p.payload["type_source"] == "inferred"
