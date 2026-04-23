"""Vault MCP server — knowledge-base tools over the Qdrant vault collection.

Tools exposed:
  vault_search          Hybrid RRF search (dense + BM25), optional filters
  vault_search_filtered Explicit filter axes: doc_type, tags, folder, status, date
  vault_get_chunks      All chunks for a file in order
  vault_outline         Heading hierarchy (h1/h2/h3) for a file, no body text
  vault_find_backlinks  Notes that link to a given file via forward_links
  vault_list_recent     Most recently modified notes
  vault_list_by_tag     All files carrying a specific tag
  vault_stats           Collection summary (counts, doc_type breakdown, top tags)
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    OrderBy,
    Prefetch,
    Range,
)

from vault_qdrant.collection import VAULT_COLLECTION
from vault_qdrant.embedder import BM25Embedder, DenseEmbedder

DENSE_FIELD = "fast-bge-large-en-v1.5"
SPARSE_FIELD = "sparse"

mcp = FastMCP("vault")

# ---------------------------------------------------------------------------
# Shared singletons — embedders lazy-load on first call
# ---------------------------------------------------------------------------

_dense: DenseEmbedder | None = None
_bm25: BM25Embedder | None = None
_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
        _client = QdrantClient(host="localhost", port=port)
    return _client


def _get_dense() -> DenseEmbedder:
    global _dense
    if _dense is None:
        _dense = DenseEmbedder()
    return _dense


def _get_bm25() -> BM25Embedder:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Embedder()
    return _bm25


_reranker: "CrossEncoderReranker | None" = None


def _get_reranker() -> "CrossEncoderReranker":
    global _reranker
    if _reranker is None:
        from vault_qdrant.embedder import CrossEncoderReranker
        _reranker = CrossEncoderReranker()
    return _reranker


_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import os as _os
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=_os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filter(
    doc_type: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    status: str | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> Filter | None:
    must: list[Any] = []
    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
    if folder:
        must.append(FieldCondition(key="folder", match=MatchValue(value=folder)))
    if status:
        must.append(FieldCondition(key="status", match=MatchValue(value=status)))
    if modified_after or modified_before:
        range_params: dict[str, Any] = {}
        if modified_after:
            range_params["gte"] = modified_after
        if modified_before:
            range_params["lte"] = modified_before
        must.append(FieldCondition(key="modified_at", range=Range(**range_params)))
    return Filter(must=must) if must else None


def _format_hit(hit: Any) -> dict:
    p = hit.payload or {}
    return {
        "file_path": p.get("file_path"),
        "score": round(hit.score, 4) if hasattr(hit, "score") else None,
        "doc_type": p.get("doc_type"),
        "h1": p.get("h1"),
        "h2": p.get("h2"),
        "h3": p.get("h3"),
        "tags": p.get("tags", []),
        "forward_links": p.get("forward_links", []),
        "text": (p.get("text") or "")[:600],
        "modified_at": p.get("modified_at"),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def vault_search(
    query: str,
    limit: int = 5,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    status: str | None = None,
    rerank: bool = False,
) -> list[dict]:
    """Hybrid semantic + keyword search over the Obsidian vault.

    Combines dense vector similarity (BAAI/bge-large-en-v1.5) with BM25 sparse
    retrieval using server-side Reciprocal Rank Fusion. Returns ranked chunks.

    Optional filters: doc_type, tags, folder, status.
    Set rerank=True to apply cross-encoder reranking (~200ms extra latency).

    Example: vault_search("deployment infrastructure", doc_type="session", rerank=True)
    """
    client = _get_client()
    dense_vec = _get_dense().embed(query)
    sparse_vec = _get_bm25().embed(query)
    query_filter = _build_filter(doc_type=doc_type, tags=tags, folder=folder, status=status)

    fetch_limit = max(limit * 3, 20) if rerank else limit
    hits = client.query_points(
        collection_name=VAULT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_FIELD, limit=50),
            Prefetch(query=sparse_vec, using=SPARSE_FIELD, limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_limit,
        query_filter=query_filter,
        with_payload=True,
    ).points

    if rerank:
        formatted = [_format_hit(h) for h in hits]
        return _get_reranker().rerank(query, formatted)[:limit]

    return [_format_hit(h) for h in hits]


@mcp.tool()
def vault_search_filtered(
    query: str,
    limit: int = 5,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    status: str | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> list[dict]:
    """Hybrid search with full filter control including date range.

    Same hybrid RRF search as vault_search but exposes date range filters.
    Use when the user wants notes from a specific time period, e.g.
    "notes I wrote last month about planning" or "active projects in Q1".

    Date format for modified_after / modified_before: ISO 8601, e.g. "2025-01-01T00:00:00Z".
    """
    client = _get_client()
    dense_vec = _get_dense().embed(query)
    sparse_vec = _get_bm25().embed(query)
    query_filter = _build_filter(
        doc_type=doc_type,
        tags=tags,
        folder=folder,
        status=status,
        modified_after=modified_after,
        modified_before=modified_before,
    )

    hits = client.query_points(
        collection_name=VAULT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_FIELD, limit=50),
            Prefetch(query=sparse_vec, using=SPARSE_FIELD, limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [_format_hit(h) for h in hits]


@mcp.tool()
def vault_search_documents(
    query: str,
    limit: int = 5,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Document-level hybrid search — returns the best matching document per file.

    Unlike vault_search (which returns raw chunks), this groups results by source
    file and surfaces the highest-scoring chunk per document. Use this when you
    want to find which *notes* are most relevant, not which specific passages.

    Example: vault_search_documents("phase 0 deployment plan") returns the
    deployment plan file as rank #1, even if a session note discussing it
    ranks higher at the raw-chunk level.

    Optional filters: doc_type, tags, folder, status.
    """
    client = _get_client()
    dense_vec = _get_dense().embed(query)
    sparse_vec = _get_bm25().embed(query)
    query_filter = _build_filter(doc_type=doc_type, tags=tags, folder=folder, status=status)

    fetch_limit = max(limit * 3, 15)
    hits = client.query_points(
        collection_name=VAULT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_FIELD, limit=50),
            Prefetch(query=sparse_vec, using=SPARSE_FIELD, limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_limit,
        query_filter=query_filter,
        with_payload=True,
    ).points

    best: dict[str, Any] = {}
    for hit in hits:
        p = hit.payload or {}
        fp = p.get("file_path")
        if not fp:
            continue
        existing = best.get(fp)
        if existing is None:
            best[fp] = hit
        elif hit.score > existing.score:
            best[fp] = hit
        elif hit.score == existing.score and p.get("is_title_chunk"):
            best[fp] = hit

    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]

    return [
        {
            "file_path": (h.payload or {}).get("file_path"),
            "best_score": round(h.score, 4),
            "doc_type": (h.payload or {}).get("doc_type"),
            "h1": (h.payload or {}).get("h1"),
            "tags": (h.payload or {}).get("tags", []),
            "status": (h.payload or {}).get("status"),
            "modified_at": (h.payload or {}).get("modified_at"),
            "forward_links": (h.payload or {}).get("forward_links", []),
            "is_title_chunk": (h.payload or {}).get("is_title_chunk", False),
            "text": ((h.payload or {}).get("text") or "")[:600],
        }
        for h in ranked
    ]


@mcp.tool()
def vault_get_chunks(file_path: str) -> list[dict]:
    """Return all indexed chunks for a specific vault note in reading order.

    Use when the user wants to read a full note or understand its complete content.
    Chunks are sorted by chunk_index so they reconstruct the document flow.
    Each chunk includes its heading context (h1, h2, h3) and full text.
    """
    client = _get_client()
    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    chunks = sorted(
        points,
        key=lambda p: p.payload.get("chunk_index", 0) if p.payload else 0,
    )

    return [
        {
            "chunk_index": (p.payload or {}).get("chunk_index"),
            "h1": (p.payload or {}).get("h1"),
            "h2": (p.payload or {}).get("h2"),
            "h3": (p.payload or {}).get("h3"),
            "text": (p.payload or {}).get("text", ""),
            "forward_links": (p.payload or {}).get("forward_links", []),
        }
        for p in chunks
    ]


@mcp.tool()
def vault_outline(file_path: str) -> list[dict]:
    """Return the heading structure (h1/h2/h3) for a note without full body text.

    Use for a fast structural overview of a document — what sections it has —
    before deciding whether to read the full content with vault_get_chunks.
    """
    client = _get_client()
    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    chunks = sorted(
        points,
        key=lambda p: p.payload.get("chunk_index", 0) if p.payload else 0,
    )

    seen: list[dict] = []
    for p in chunks:
        payload = p.payload or {}
        entry = {"h1": payload.get("h1"), "h2": payload.get("h2"), "h3": payload.get("h3")}
        if not seen or seen[-1] != entry:
            seen.append(entry)

    return seen


@mcp.tool()
def vault_find_backlinks(file_path: str) -> list[str]:
    """Find all vault notes that link to the given file_path via wiki links.

    Use when the user asks "what links to this note?" or wants to understand
    a note's connections in the knowledge graph. Returns a deduplicated sorted
    list of file_paths whose forward_links field contains the target.
    """
    client = _get_client()
    note_name = file_path.split("/")[-1].replace(".md", "")

    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="forward_links",
                    match=MatchAny(any=[file_path, note_name]),
                )
            ]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    seen: set[str] = set()
    result: list[str] = []
    for p in points:
        fp = (p.payload or {}).get("file_path", "")
        if fp and fp not in seen:
            seen.add(fp)
            result.append(fp)

    return sorted(result)


@mcp.tool()
def vault_list_recent(limit: int = 10, doc_type: str | None = None) -> list[dict]:
    """List the most recently modified notes in the vault.

    Returns notes sorted by modified_at descending — useful for "what did I
    work on recently?" queries. Optionally filter to a specific doc_type
    (e.g. "meeting", "project", "note"). Returns one entry per document.
    """
    client = _get_client()
    scroll_filter = _build_filter(doc_type=doc_type)

    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=scroll_filter,
        limit=limit * 5,
        with_payload=True,
        with_vectors=False,
        order_by=OrderBy(key="modified_at", direction="desc"),
    )

    seen: set[str] = set()
    result: list[dict] = []
    for p in points:
        payload = p.payload or {}
        fp = payload.get("file_path", "")
        if fp and fp not in seen:
            seen.add(fp)
            result.append(
                {
                    "file_path": fp,
                    "doc_type": payload.get("doc_type"),
                    "tags": payload.get("tags", []),
                    "status": payload.get("status"),
                    "modified_at": payload.get("modified_at"),
                }
            )
            if len(result) >= limit:
                break

    return result


@mcp.tool()
def vault_list_by_tag(tag: str, limit: int = 50) -> list[str]:
    """Return all vault note file_paths that carry a specific tag.

    Use when the user wants to browse notes by topic or category, e.g.
    "show me all notes tagged medusa" or "list my meeting notes".
    Returns unique file_paths sorted alphabetically (one per document).
    """
    client = _get_client()

    points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="tags", match=MatchAny(any=[tag]))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    seen: set[str] = set()
    result: list[str] = []
    for p in points:
        fp = (p.payload or {}).get("file_path", "")
        if fp and fp not in seen:
            seen.add(fp)
            result.append(fp)
            if len(result) >= limit:
                break

    return sorted(result)


@mcp.tool()
def vault_ask(
    query: str,
    limit: int = 8,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    rerank: bool = True,
) -> dict:
    """Answer a question using knowledge from the vault.

    Retrieves relevant chunks via hybrid search, then calls Claude (Haiku) to
    synthesize a direct answer grounded in vault content. Returns the answer and
    the source notes used.

    Example: vault_ask("What decisions were made for the production hosting stack?")
    Returns: {"answer": "...", "sources": [{"file_path": "...", "h1": "...", "score": 0.9}]}
    """
    hits = vault_search(query, limit=limit, doc_type=doc_type, tags=tags, rerank=rerank)

    if not hits:
        return {"answer": "No relevant content found in the vault.", "sources": []}

    context_parts = []
    for hit in hits:
        fp = hit.get("file_path", "unknown")
        text = hit.get("text", "")
        context_parts.append(f"[{fp}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=(
            "You are a knowledge assistant with access to vault excerpts. "
            "Answer using only the provided excerpts. Cite sources by file_path in brackets. "
            "If the answer is not in the excerpts, say so explicitly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Vault excerpts:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )

    answer = response.content[0].text

    seen_fps: set[str] = set()
    sources: list[dict] = []
    for hit in hits:
        fp = hit.get("file_path")
        if fp and fp not in seen_fps:
            seen_fps.add(fp)
            sources.append({"file_path": fp, "h1": hit.get("h1"), "score": hit.get("score")})

    return {"answer": answer, "sources": sources}


@mcp.tool()
def vault_related_notes(file_path: str, limit: int = 10) -> list[dict]:
    """Find vault notes that are semantically similar to the given file.

    Uses the stored dense vector of the file's first chunk (chunk_index=0) to
    find similar documents — no re-embedding needed.

    Example: vault_related_notes("projects/medusa/deployment/phase-0.md")
    Returns notes about deployment, infrastructure, or related planning topics.
    """
    client = _get_client()

    title_points, _ = client.scroll(
        collection_name=VAULT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not title_points:
        return []

    title_point_id = title_points[0].id

    hits = client.query_points(
        collection_name=VAULT_COLLECTION,
        query=title_point_id,
        using=DENSE_FIELD,
        limit=limit * 5,
        query_filter=Filter(
            must_not=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
        ),
        with_payload=True,
    ).points

    best: dict[str, Any] = {}
    for hit in hits:
        p = hit.payload or {}
        fp = p.get("file_path")
        if not fp or fp == file_path:
            continue
        if fp not in best or hit.score > best[fp].score:
            best[fp] = hit

    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]

    return [
        {
            "file_path": (h.payload or {}).get("file_path"),
            "score": round(h.score, 4),
            "doc_type": (h.payload or {}).get("doc_type"),
            "h1": (h.payload or {}).get("h1"),
            "tags": (h.payload or {}).get("tags", []),
            "status": (h.payload or {}).get("status"),
            "modified_at": (h.payload or {}).get("modified_at"),
        }
        for h in ranked
    ]


@mcp.tool()
def vault_stats() -> dict:
    """Return a summary of the vault collection: total chunks, doc_type breakdown, top tags, quality metrics.

    The 'quality' section shows vault metadata health: notes missing frontmatter type,
    notes with no tags, inferred vs explicit type counts, and average chunks per document.
    """
    client = _get_client()

    info = client.get_collection(VAULT_COLLECTION)
    total_points = info.points_count or 0

    doc_type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    doc_tags: dict[str, list] = {}
    doc_type_source: dict[str, str] = {}
    doc_has_type: dict[str, bool] = {}
    chunk_counts: dict[str, int] = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=VAULT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            dt = payload.get("doc_type") or "unknown"
            doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1
            for tag in payload.get("tags") or []:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            fp = payload.get("file_path")
            if fp:
                chunk_counts[fp] = chunk_counts.get(fp, 0) + 1
                if fp not in doc_tags:
                    doc_tags[fp] = payload.get("tags") or []
                    doc_type_source[fp] = payload.get("type_source", "inferred")
                    doc_has_type[fp] = payload.get("doc_type") is not None

        if offset is None:
            break

    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    unique_docs = len(chunk_counts)
    avg_chunks = round(total_points / unique_docs, 2) if unique_docs else 0.0

    return {
        "total_chunks": total_points,
        "doc_type_breakdown": dict(
            sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        "quality": {
            "notes_without_type": sum(1 for has_type in doc_has_type.values() if not has_type),
            "notes_without_tags": sum(1 for tags in doc_tags.values() if not tags),
            "inferred_type_count": sum(1 for src in doc_type_source.values() if src == "inferred"),
            "avg_chunks_per_doc": avg_chunks,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
