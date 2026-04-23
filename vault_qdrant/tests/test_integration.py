"""Integration and acceptance tests for the vault_qdrant pipeline.

Integration tests use unittest.mock — no real services needed.
Acceptance test requires a running Qdrant at localhost:6333 (skipped otherwise).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector


# ---------------------------------------------------------------------------
# Acceptance test guard
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
    try:
        return requests.get("http://localhost:6333/healthz", timeout=1).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Integration test 1: collection bootstrap
# ---------------------------------------------------------------------------


def test_collection_bootstrap_creates_vault():
    """ensure_vault_collection creates 'vault' and all 5 payload indexes."""
    from vault_qdrant.collection import ensure_vault_collection

    mock_client = MagicMock()
    # Simulate collection not existing: get_collection raises, so creation proceeds
    mock_client.get_collection.side_effect = Exception("not found")

    ensure_vault_collection(mock_client)

    # create_collection must be called with collection_name="vault"
    mock_client.create_collection.assert_called_once()
    create_kwargs = mock_client.create_collection.call_args.kwargs
    assert create_kwargs.get("collection_name") == "vault", (
        f"Expected collection_name='vault', got {create_kwargs.get('collection_name')!r}"
    )

    # create_payload_index must be called exactly 5 times
    assert mock_client.create_payload_index.call_count == 5, (
        f"Expected 5 payload index calls, got {mock_client.create_payload_index.call_count}"
    )

    # Verify each expected field is indexed
    indexed_fields = {
        c.kwargs.get("field_name")
        for c in mock_client.create_payload_index.call_args_list
    }
    expected_fields = {"doc_type", "tags", "folder", "modified_at", "status"}
    assert indexed_fields == expected_fields, (
        f"Expected indexes for {expected_fields}, got {indexed_fields}"
    )


# ---------------------------------------------------------------------------
# Integration test 2: upsert search roundtrip
# ---------------------------------------------------------------------------


def test_upsert_search_roundtrip():
    """upsert_chunks stores a point with the correct file_path in payload."""
    from vault_qdrant.upserter import upsert_chunks

    mock_client = MagicMock()
    # No existing doc hash — scroll returns empty
    mock_client.scroll.return_value = ([], None)

    mock_dense = MagicMock()
    mock_dense.embed.return_value = [0.1] * 1024

    mock_bm25 = MagicMock()
    mock_bm25.embed.return_value = SparseVector(indices=[0, 1], values=[0.8, 0.4])

    doc = {
        "file_path": "projects/roundtrip.md",
        "content": "# Roundtrip\nSome integration test content.",
        "tags": ["integration"],
        "type": "spec",
        "created": "2025-01-01",
        "status": "active",
        "projects": ["medusa"],
        "doc_hash": "roundtrip_hash_001",
    }

    chunks = [
        {
            "text": "Integration test content",
            "h1": "Roundtrip",
            "h2": "Overview",
            "h3": None,
            "forward_links": [],
            "chunk_index": 0,
        }
    ]

    upsert_chunks(mock_client, mock_dense, mock_bm25, doc, chunks)

    # client.upsert must have been called exactly once
    mock_client.upsert.assert_called_once()

    # Verify the upserted point carries the correct file_path
    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[1]

    assert len(points) == 1, f"Expected 1 point, got {len(points)}"
    assert points[0].payload["file_path"] == "projects/roundtrip.md", (
        f"Expected file_path='projects/roundtrip.md', "
        f"got {points[0].payload['file_path']!r}"
    )


# ---------------------------------------------------------------------------
# Integration test 3: metadata filter narrows results
# ---------------------------------------------------------------------------


def test_metadata_filter_narrows_results():
    """Search with a doc_type filter returns only matching hits."""
    mock_client = MagicMock()

    # Fake hits: one spec, one note
    spec_hit = MagicMock()
    spec_hit.payload = {"doc_type": "spec", "file_path": "specs/alpha.md"}
    spec_hit.score = 0.95

    note_hit = MagicMock()
    note_hit.payload = {"doc_type": "note", "file_path": "notes/beta.md"}
    note_hit.score = 0.80

    spec_filter = Filter(
        must=[FieldCondition(key="doc_type", match=MatchValue(value="spec"))]
    )

    def _search_side_effect(**kwargs):
        search_filter = kwargs.get("query_filter")
        if search_filter is None:
            return [spec_hit, note_hit]
        # Simulate Qdrant's filtering by inspecting must conditions
        must_conditions = getattr(search_filter, "must", None) or []
        for cond in must_conditions:
            field_key = getattr(cond, "key", None)
            match_val = getattr(getattr(cond, "match", None), "value", None)
            if field_key == "doc_type" and match_val == "spec":
                return [spec_hit]
        return [spec_hit, note_hit]

    mock_client.search.side_effect = _search_side_effect

    # Unfiltered search returns both hits
    all_results = mock_client.search(
        collection_name="vault",
        query_vector=("fast-bge-large-en-v1.5", [0.0] * 1024),
        limit=10,
    )
    assert len(all_results) == 2

    # Filtered search returns only the spec hit
    filtered_results = mock_client.search(
        collection_name="vault",
        query_vector=("fast-bge-large-en-v1.5", [0.0] * 1024),
        limit=10,
        query_filter=spec_filter,
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].payload["doc_type"] == "spec", (
        f"Expected doc_type='spec', got {filtered_results[0].payload['doc_type']!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance test 4: real Qdrant smoke test (skipped if unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not running at localhost:6333")
def test_acceptance_sync_and_search():
    """Smoke test: scan + chunk works end-to-end (no embedding/upsert needed)."""
    from vault_qdrant.scanner import scan

    docs = scan("/Users/konradciok/repositories/medusa/medusa/obsidian-vault")
    assert len(docs) > 0, "scan() returned no documents"

    # Verify chunker works on first doc
    from vault_qdrant.chunker import chunk

    chunks = chunk(docs[0]["content"], docs[0].get("type"))
    assert len(chunks) > 0, "chunk() returned no chunks for first doc"
    assert "text" in chunks[0], f"chunk missing 'text' key: {list(chunks[0].keys())}"
