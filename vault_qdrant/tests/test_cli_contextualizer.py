"""Tests for vault_qdrant.contextualizer and vault_qdrant.cli modules."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vault_qdrant.cli import main
from vault_qdrant.contextualizer import Contextualizer


# ---------------------------------------------------------------------------
# contextualizer.py tests
# ---------------------------------------------------------------------------


def test_contextualizer_prepends_context() -> None:
    """contextualize() returns context text followed by the original chunk."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This chunk describes the auth flow.")]

    with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropicClass:
        mock_client = MagicMock()
        MockAnthropicClass.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        c = Contextualizer(api_key="test-key")
        result = c.contextualize("full document text", "just the chunk text")

    assert result.startswith("This chunk describes the auth flow.")
    assert "just the chunk text" in result


def test_contextualizer_uses_correct_model() -> None:
    """contextualize() passes the configured model and max_tokens=150 to the API."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="context")]

    with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropicClass:
        mock_client = MagicMock()
        MockAnthropicClass.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        c = Contextualizer(api_key="test-key", model="claude-haiku-4-5-20251001")
        c.contextualize("doc", "chunk")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"
        assert call_kwargs.kwargs.get("max_tokens") == 150


def test_contextualizer_default_model() -> None:
    """Contextualizer defaults to claude-haiku-4-5-20251001 when no model given."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ctx")]

    with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropicClass:
        mock_client = MagicMock()
        MockAnthropicClass.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        c = Contextualizer(api_key="sk-test")
        assert c.model == "claude-haiku-4-5-20251001"


def test_contextualizer_initialises_anthropic_with_api_key() -> None:
    """Contextualizer forwards the api_key argument to the Anthropic constructor."""
    test_token = "fake-token-for-testing"
    with patch("vault_qdrant.contextualizer.Anthropic") as MockAnthropicClass:
        MockAnthropicClass.return_value = MagicMock()
        Contextualizer(api_key=test_token)
        _args, _kwargs = MockAnthropicClass.call_args
        assert _kwargs.get("api_key") == test_token


# ---------------------------------------------------------------------------
# cli.py tests
# ---------------------------------------------------------------------------


def test_cli_sync_requires_vault_option() -> None:
    """sync command exits non-zero when --vault is not provided."""
    runner = CliRunner()
    result = runner.invoke(main, ["sync"])
    assert result.exit_code != 0


def test_cli_search_help() -> None:
    """search --help exits 0 and mentions the QUERY argument."""
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "query" in result.output.lower() or "QUERY" in result.output


def test_cli_main_help() -> None:
    """Top-level --help exits 0 and lists subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.output
    assert "status" in result.output
    assert "search" in result.output


def test_cli_status_command_exits_with_connection_error() -> None:
    """status exits non-zero when Qdrant is unavailable (connection error)."""
    runner = CliRunner()
    with patch("vault_qdrant.cli._load_env"), \
         patch("vault_qdrant.cli.QdrantClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        mock_instance.get_collection.side_effect = Exception("connection refused")

        result = runner.invoke(main, ["status"])

    assert result.exit_code != 0


def test_cli_status_success() -> None:
    """status prints collection name, status and points when Qdrant responds."""
    runner = CliRunner()

    mock_info = MagicMock()
    mock_info.status = "green"
    mock_info.points_count = 42
    mock_info.config.params.vectors_config = {}
    mock_info.config.params.sparse_vectors_config = {"sparse": MagicMock()}

    with patch("vault_qdrant.cli._load_env"), \
         patch("vault_qdrant.cli.QdrantClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        mock_instance.get_collection.return_value = mock_info

        result = runner.invoke(main, ["status"])

    assert result.exit_code == 0
    assert "42" in result.output
    assert "green" in result.output


def test_cli_sync_force_flag_parsed() -> None:
    """sync --force is accepted and runs without hitting real services."""
    runner = CliRunner()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("vault_qdrant.cli._load_env"), \
             patch("vault_qdrant.cli._make_client", return_value=MagicMock()), \
             patch("vault_qdrant.cli._make_ollama", return_value=MagicMock()), \
             patch("vault_qdrant.cli.BM25Embedder", return_value=MagicMock()), \
             patch("vault_qdrant.cli._make_contextualizer", return_value=MagicMock()), \
             patch("vault_qdrant.cli.ensure_vault_collection"), \
             patch("vault_qdrant.cli.scan", return_value=[]), \
             patch("vault_qdrant.cli.delete_orphans", return_value=0):
            result = runner.invoke(main, ["sync", "--vault", tmpdir, "--force"])

    assert result.exit_code == 0
    assert "Files scanned" in result.output


def test_cli_sync_without_force_flag() -> None:
    """sync without --force runs the normal (non-force) path."""
    runner = CliRunner()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("vault_qdrant.cli._load_env"), \
             patch("vault_qdrant.cli._make_client", return_value=MagicMock()), \
             patch("vault_qdrant.cli._make_ollama", return_value=MagicMock()), \
             patch("vault_qdrant.cli.BM25Embedder", return_value=MagicMock()), \
             patch("vault_qdrant.cli._make_contextualizer", return_value=MagicMock()), \
             patch("vault_qdrant.cli.ensure_vault_collection"), \
             patch("vault_qdrant.cli.scan", return_value=[]), \
             patch("vault_qdrant.cli.delete_orphans", return_value=0):
            result = runner.invoke(main, ["sync", "--vault", tmpdir])

    assert result.exit_code == 0
    assert "Orphans deleted" in result.output


def test_cli_search_no_results() -> None:
    """search prints 'No results found.' when both hit lists are empty."""
    runner = CliRunner()
    with patch("vault_qdrant.cli._load_env"), \
         patch("vault_qdrant.cli._make_client") as MockClient, \
         patch("vault_qdrant.cli._make_ollama") as MockOllama, \
         patch("vault_qdrant.cli.BM25Embedder") as MockBM25:
        mock_ollama_inst = MagicMock()
        mock_ollama_inst.embed.return_value = [0.1] * 384
        MockOllama.return_value = mock_ollama_inst

        mock_bm25_inst = MagicMock()
        mock_bm25_inst.embed.return_value = {}
        MockBM25.return_value = mock_bm25_inst

        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = []
        MockClient.return_value = mock_qdrant

        result = runner.invoke(main, ["search", "test query"])

    assert result.exit_code == 0
    assert "No results found." in result.output
