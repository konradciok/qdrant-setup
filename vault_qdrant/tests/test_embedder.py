"""Tests for embedding models including CrossEncoderReranker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from vault_qdrant.embedder import CrossEncoderReranker


def test_reranker_reorders_hits_by_score():
    """CrossEncoderReranker must return hits sorted by cross-encoder score descending."""
    hits = [
        {"text": "low relevance text", "file_path": "a.md"},
        {"text": "highly relevant text", "file_path": "b.md"},
    ]

    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]
    reranker._model = mock_model

    result = reranker.rerank("query", hits)

    assert result[0]["file_path"] == "b.md"
    assert result[1]["file_path"] == "a.md"


def test_reranker_passes_correct_pairs_to_model():
    """rerank() must call model.predict() with (query, text) pairs for each hit."""
    hits = [
        {"text": "alpha text", "file_path": "a.md"},
        {"text": "beta text", "file_path": "b.md"},
    ]

    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.5]
    reranker._model = mock_model

    reranker.rerank("my query", hits)

    call_args = mock_model.predict.call_args[0][0]
    assert call_args == [("my query", "alpha text"), ("my query", "beta text")]


def test_reranker_lazy_loads_model():
    """CrossEncoderReranker must not load the model at construction time."""
    reranker = CrossEncoderReranker()
    assert reranker._model is None, "Model must be None before first use"
