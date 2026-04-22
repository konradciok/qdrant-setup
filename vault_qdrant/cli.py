"""CLI for the Vault-Qdrant pipeline.

Commands
--------
sync  --vault <path> [--force]   Full pipeline: scan → chunk → contextualize → upsert
status                            Collection info
search <query>                    Hybrid dense+sparse search with RRF fusion
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch

from vault_qdrant.chunker import chunk
from vault_qdrant.collection import VAULT_COLLECTION, ensure_vault_collection
from vault_qdrant.contextualizer import Contextualizer
from vault_qdrant.embedder import BM25Embedder, OllamaEmbedder
from vault_qdrant.scanner import scan
from vault_qdrant.upserter import _existing_doc_hash, delete_orphans, upsert_chunks

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> None:
    load_dotenv(_ENV_FILE)


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        click.echo(f"ERROR: required environment variable {name!r} is not set.", err=True)
        sys.exit(1)
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        click.echo(f"ERROR: {name!r} must be an integer, got {raw!r}.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Client / embedder factory helpers
# ---------------------------------------------------------------------------


def _make_client() -> QdrantClient:
    port = _int_env("QDRANT_HTTP_PORT", 6333)
    return QdrantClient(host="localhost", port=port)


def _make_ollama() -> OllamaEmbedder:
    base_url = _require_env("OLLAMA_BASE_URL")
    model = _require_env("OLLAMA_EMBED_MODEL")
    return OllamaEmbedder(base_url=base_url, model=model)


def _make_contextualizer() -> Contextualizer:
    api_key = _require_env("ANTHROPIC_API_KEY")
    return Contextualizer(api_key=api_key)


# ---------------------------------------------------------------------------
# Sync pipeline helpers
# ---------------------------------------------------------------------------


def _contextualize_chunks(
    contextualizer: Contextualizer,
    doc_content: str,
    chunks: list[dict],
) -> list[dict]:
    """Return new chunk list with contextualized text. Originals are not mutated."""
    result = []
    for ch in chunks:
        new_text = contextualizer.contextualize(doc_content, ch["text"])
        result.append({**ch, "text": new_text})
    return result


def _sync_doc(
    client: QdrantClient,
    ollama: OllamaEmbedder,
    bm25: BM25Embedder,
    contextualizer: Contextualizer | None,
    doc: dict,
    force: bool,
) -> int:
    """Chunk, contextualize (if contextualizer provided), and upsert a single doc."""
    if not force and _existing_doc_hash(client, doc["file_path"]) == doc["doc_hash"]:
        return 0
    effective_doc = {**doc, "doc_hash": ""} if force else doc
    raw_chunks = chunk(doc["content"], doc.get("type"))
    ctx_chunks = (
        _contextualize_chunks(contextualizer, doc["content"], raw_chunks)
        if contextualizer is not None
        else raw_chunks
    )
    upsert_chunks(client, ollama, bm25, effective_doc, ctx_chunks)
    return len(ctx_chunks)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Vault-Qdrant pipeline CLI."""


# ---------------------------------------------------------------------------
# sync command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--vault", required=True, type=click.Path(exists=True), help="Path to Obsidian vault")
@click.option("--force", is_flag=True, default=False, help="Re-upsert even if doc_hash unchanged")
@click.option("--no-context", "no_context", is_flag=True, default=False, help="Skip Anthropic contextualisation")
def sync(vault: str, force: bool, no_context: bool) -> None:
    """Scan vault, chunk docs, contextualize, and upsert into Qdrant."""
    _load_env()
    client = _make_client()
    ollama = _make_ollama()
    bm25 = BM25Embedder()
    contextualizer = None if no_context else _make_contextualizer()

    ensure_vault_collection(client)

    docs = scan(vault)
    click.echo(f"Scanned {len(docs)} file(s) from {vault}")

    total_chunks = 0
    for doc in docs:
        n = _sync_doc(client, ollama, bm25, contextualizer, doc, force)
        total_chunks += n

    orphans = delete_orphans(client, {d["file_path"] for d in docs})

    click.echo(f"Files scanned  : {len(docs)}")
    click.echo(f"Chunks upserted: {total_chunks}")
    click.echo(f"Orphans deleted: {orphans}")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@main.command()
def status() -> None:
    """Print Qdrant collection info."""
    _load_env()
    client = _make_client()

    try:
        info = client.get_collection(VAULT_COLLECTION)
    except Exception as exc:
        click.echo(f"ERROR: could not retrieve collection {VAULT_COLLECTION!r}: {exc}", err=True)
        sys.exit(1)

    vectors_config = info.config.params.vectors_config or {}
    sparse_config = info.config.params.sparse_vectors_config or {}

    dense_size = _dense_size(vectors_config)
    sparse_info = _sparse_info(sparse_config)

    click.echo(f"Collection : {VAULT_COLLECTION}")
    click.echo(f"Status     : {info.status}")
    click.echo(f"Points     : {info.points_count}")
    click.echo(f"Dense dim  : {dense_size}")
    click.echo(f"Sparse     : {sparse_info}")


def _dense_size(vectors_config: Any) -> str:
    """Extract dense vector dimension from vectors config."""
    if hasattr(vectors_config, "__getitem__"):
        try:
            return str(vectors_config["dense"].size)
        except (KeyError, TypeError, AttributeError):
            pass
    if hasattr(vectors_config, "dense"):
        return str(vectors_config.dense.size)
    return "unknown"


def _sparse_info(sparse_config: Any) -> str:
    """Return a readable description of sparse vector config."""
    if not sparse_config:
        return "none"
    if hasattr(sparse_config, "keys"):
        return ", ".join(str(k) for k in sparse_config.keys())
    return str(sparse_config)


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
def search(query: str) -> None:
    """Hybrid semantic + BM25 search with server-side RRF fusion."""
    _load_env()
    client = _make_client()
    ollama = _make_ollama()
    bm25 = BM25Embedder()

    dense_vec = ollama.embed(query)
    sparse_vec = bm25.embed(query)

    hits = client.query_points(
        collection_name=VAULT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=20),
            Prefetch(query=sparse_vec, using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
        with_payload=True,
    ).points

    if not hits:
        click.echo("No results found.")
        return

    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text_preview = (payload.get("text") or "")[:200]
        click.echo(
            f"[{rank}] score={hit.score:.4f}  {payload.get('file_path', '')}  "
            f"h2={payload.get('h2', '')}  {text_preview!r}"
        )
