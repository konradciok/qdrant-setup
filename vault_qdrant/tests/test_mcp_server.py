"""Tests for mcp_server helpers — no Qdrant connection needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from vault_qdrant.mcp_server import _format_hit


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
