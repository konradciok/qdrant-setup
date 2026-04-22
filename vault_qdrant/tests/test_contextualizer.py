"""Tests for contextualizer prompt caching behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from vault_qdrant.contextualizer import Contextualizer


def _make_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


class TestContextualizerCaching:
    def test_document_block_has_cache_control(self) -> None:
        """The document content block must carry cache_control=ephemeral."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("full document text", "just chunk text")

            create_kwargs = mock_client.messages.create.call_args.kwargs
            messages = create_kwargs["messages"]
            assert len(messages) == 1
            user_content = messages[0]["content"]
            doc_block = user_content[0]
            assert doc_block["cache_control"] == {"type": "ephemeral"}, (
                "Document block must have cache_control={'type': 'ephemeral'}"
            )

    def test_chunk_block_has_no_cache_control(self) -> None:
        """The chunk content block must NOT carry cache_control (it is volatile)."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("full document text", "just chunk text")

            create_kwargs = mock_client.messages.create.call_args.kwargs
            user_content = create_kwargs["messages"][0]["content"]
            chunk_block = user_content[1]
            assert "cache_control" not in chunk_block, (
                "Chunk block must not have cache_control — it changes per call"
            )

    def test_document_appears_before_chunk(self) -> None:
        """Document block must precede chunk block (prefix caching requirement)."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Context sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            c.contextualize("THE DOCUMENT", "THE CHUNK")

            user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
            assert "THE DOCUMENT" in user_content[0]["text"]
            assert "THE CHUNK" in user_content[1]["text"]

    def test_return_value_prepends_context(self) -> None:
        """contextualize() must return '<context>\\n\\n<original chunk>'."""
        with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _make_response("Intro sentence.")
            MockAnthropic.return_value = mock_client

            c = Contextualizer(api_key="test-key")
            result = c.contextualize("doc", "chunk text")

            assert result == "Intro sentence.\n\nchunk text"
