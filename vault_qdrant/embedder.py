"""Embedding models for vault chunks.

Provides two embedders:
- DenseEmbedder: Dense embeddings (1024-dim) via fastembed
- BM25Embedder: Sparse embeddings via fastembed
"""

from __future__ import annotations

from qdrant_client.models import SparseVector

DENSE_MODEL = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"


class DenseEmbedder:
    """Dense text embeddings via fastembed (BAAI/bge-large-en-v1.5, 1024-dim)."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed not installed. Install with: pip install fastembed"
                ) from exc
            self._model = TextEmbedding(model_name=DENSE_MODEL)

    def embed(self, text: str) -> list[float]:
        """Embed text into a 1024-dim dense vector.

        Args:
            text: Text to embed

        Returns:
            Dense vector as list of floats (1024-dim)
        """
        self._ensure_model()
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()


class BM25Embedder:
    """Embed text using BM25 sparse embeddings via fastembed.

    Uses the "Qdrant/bm25" model from fastembed library.
    Returns sparse vectors (indices and values).
    """

    def __init__(self) -> None:
        """Initialize BM25 embedder (lazy-loads fastembed model)."""
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load fastembed SparseTextEmbedding model."""
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed not installed. Install with: pip install fastembed"
                ) from exc
            self._model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def embed(self, text: str) -> SparseVector:
        """Embed text into a sparse vector using BM25.

        Args:
            text: Text to embed

        Returns:
            SparseVector with indices and values

        Raises:
            ImportError: If fastembed is not installed
        """
        self._ensure_model()
        embeddings = list(self._model.embed([text]))
        sparse_embedding = embeddings[0]
        return SparseVector(indices=sparse_embedding.indices, values=sparse_embedding.values)


CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Reranks a list of hit dicts by (query, text) cross-encoder score.
    Model is lazy-loaded on first call and cached for the session.
    """

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> "CrossEncoder":
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. Run: uv sync"
            ) from exc
        return CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """Return hits re-sorted by cross-encoder relevance score (descending).

        Args:
            query: The search query string.
            hits: List of hit dicts, each with a "text" key.

        Returns:
            Same hits sorted by cross-encoder score, highest first.
        """
        if self._model is None:
            self._model = self._load_model()
        pairs = [(query, h.get("text", "")) for h in hits]
        scores = self._model.predict(pairs)
        return [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
