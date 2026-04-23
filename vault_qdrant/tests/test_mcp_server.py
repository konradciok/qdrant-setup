"""Tests for mcp_server helpers — no Qdrant connection needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from vault_qdrant.mcp_server import _format_hit, vault_search, vault_search_documents


def _make_hit(payload: dict, score: float = 0.75) -> MagicMock:
    hit = MagicMock()
    hit.payload = payload
    hit.score = score
    return hit


def test_format_hit_returns_600_char_text():
    long_text = "x" * 700
    hit = _make_hit({"text": long_text, "file_path": "a.md"})
    result = _format_hit(hit)
    assert len(result["text"]) == 600


def test_format_hit_includes_forward_links():
    hit = _make_hit({"text": "hello", "forward_links": ["a.md", "b.md"], "file_path": "c.md"})
    result = _format_hit(hit)
    assert result["forward_links"] == ["a.md", "b.md"]


def test_format_hit_missing_forward_links_defaults_to_empty():
    hit = _make_hit({"text": "hello", "file_path": "c.md"})
    result = _format_hit(hit)
    assert result["forward_links"] == []


def test_format_hit_score_rounded():
    hit = _make_hit({"text": "hi", "file_path": "a.md"}, score=0.123456)
    result = _format_hit(hit)
    assert result["score"] == 0.1235


def _make_search_hit(file_path: str, score: float, chunk_index: int = 0) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "file_path": file_path,
        "doc_type": "note",
        "h1": "Title",
        "h2": None,
        "h3": None,
        "tags": [],
        "forward_links": [],
        "text": "some text",
        "modified_at": "2026-01-01T00:00:00Z",
        "status": None,
        "is_title_chunk": chunk_index == 0,
        "chunk_index": chunk_index,
    }
    return hit


def test_vault_search_documents_groups_by_file():
    """Multiple chunks from same file must collapse to one document result."""
    hits = [
        _make_search_hit("file_a.md", 0.9, chunk_index=0),
        _make_search_hit("file_a.md", 0.7, chunk_index=1),
        _make_search_hit("file_b.md", 0.8, chunk_index=0),
    ]
    mock_result = MagicMock()
    mock_result.points = hits

    with patch("vault_qdrant.mcp_server._get_client") as mock_client_fn, \
         patch("vault_qdrant.mcp_server._get_dense") as mock_dense_fn, \
         patch("vault_qdrant.mcp_server._get_bm25") as mock_bm25_fn:

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        mock_client_fn.return_value = mock_client
        mock_dense_fn.return_value = MagicMock(embed=MagicMock(return_value=[0.1] * 1024))
        mock_bm25_fn.return_value = MagicMock(
            embed=MagicMock(return_value=MagicMock(indices=[1], values=[0.5]))
        )

        results = vault_search_documents("test query", limit=5)

    assert len(results) == 2, f"Expected 2 unique docs, got {len(results)}"
    assert results[0]["file_path"] == "file_a.md"
    assert results[0]["best_score"] == 0.9
    assert results[1]["file_path"] == "file_b.md"


def test_vault_search_documents_prefers_title_chunk_on_tie():
    """On score tie, is_title_chunk=True chunk must supply the result text."""
    hits = [
        _make_search_hit("x.md", 0.5, chunk_index=1),
        _make_search_hit("x.md", 0.5, chunk_index=0),
    ]
    mock_result = MagicMock()
    mock_result.points = hits

    with patch("vault_qdrant.mcp_server._get_client") as mock_client_fn, \
         patch("vault_qdrant.mcp_server._get_dense") as mock_dense_fn, \
         patch("vault_qdrant.mcp_server._get_bm25") as mock_bm25_fn:

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        mock_client_fn.return_value = mock_client
        mock_dense_fn.return_value = MagicMock(embed=MagicMock(return_value=[0.1] * 1024))
        mock_bm25_fn.return_value = MagicMock(
            embed=MagicMock(return_value=MagicMock(indices=[1], values=[0.5]))
        )

        results = vault_search_documents("query")

    assert results[0]["is_title_chunk"] is True


def test_vault_search_rerank_invokes_reranker():
    """When rerank=True, _get_reranker().rerank() must be called."""
    hits = [_make_search_hit("a.md", 0.5), _make_search_hit("b.md", 0.9)]
    mock_result = MagicMock()
    mock_result.points = hits

    with patch("vault_qdrant.mcp_server._get_client") as mock_client_fn, \
         patch("vault_qdrant.mcp_server._get_dense") as mock_dense_fn, \
         patch("vault_qdrant.mcp_server._get_bm25") as mock_bm25_fn, \
         patch("vault_qdrant.mcp_server._get_reranker") as mock_reranker_fn:

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        mock_client_fn.return_value = mock_client
        mock_dense_fn.return_value = MagicMock(embed=MagicMock(return_value=[0.1] * 1024))
        mock_bm25_fn.return_value = MagicMock(
            embed=MagicMock(return_value=MagicMock(indices=[1], values=[0.5]))
        )
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [_format_hit(h) for h in hits]
        mock_reranker_fn.return_value = mock_reranker

        vault_search("query", limit=2, rerank=True)

        mock_reranker.rerank.assert_called_once()


def test_vault_search_default_no_rerank():
    """When rerank=False (default), _get_reranker must not be called."""
    mock_result = MagicMock()
    mock_result.points = []

    with patch("vault_qdrant.mcp_server._get_client") as mock_client_fn, \
         patch("vault_qdrant.mcp_server._get_dense") as mock_dense_fn, \
         patch("vault_qdrant.mcp_server._get_bm25") as mock_bm25_fn, \
         patch("vault_qdrant.mcp_server._get_reranker") as mock_reranker_fn:

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        mock_client_fn.return_value = mock_client
        mock_dense_fn.return_value = MagicMock(embed=MagicMock(return_value=[0.1] * 1024))
        mock_bm25_fn.return_value = MagicMock(
            embed=MagicMock(return_value=MagicMock(indices=[1], values=[0.5]))
        )

        vault_search("query", limit=5)

        mock_reranker_fn.assert_not_called()
