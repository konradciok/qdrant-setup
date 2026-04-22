"""Embedding models for vault chunks.

Provides two embedders:
- OllamaEmbedder: Dense embeddings (2560-dim) via Ollama API
- BM25Embedder: Sparse embeddings via fastembed
"""

from __future__ import annotations

import requests
from qdrant_client.models import SparseVector


class OllamaEmbedder:
    """Embed text using Ollama's embedding API.

    Calls POST {base_url}/api/embed with model and text.
    Returns dense vector (typically 2560-dim for qwen3-embedding:4b).
    """

    def __init__(self, base_url: str, model: str) -> None:
        """Initialize Ollama embedder.

        Args:
            base_url: Ollama API base URL (e.g. "http://localhost:11434")
            model: Model name (e.g. "qwen3-embedding:4b")
        """
        self.base_url = base_url
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Embed text into a dense vector.

        Args:
            text: Text to embed

        Returns:
            Dense vector as list of floats (2560-dim)

        Raises:
            requests.RequestException: If API call fails
        """
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


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
