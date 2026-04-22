#!/usr/bin/env python3
"""Smoke test: embed a string with Ollama and search it in Qdrant."""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
QDRANT_URL = f"http://localhost:{os.getenv('QDRANT_HTTP_PORT', '6333')}"
EMBED_DIM = int(os.getenv("EMBED_DIM", "2560"))
COLLECTION = "documents"


def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def upsert(point_id: int, vector: list[float], payload: dict) -> None:
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
        timeout=10,
    )
    resp.raise_for_status()


def search(vector: list[float], top_k: int = 3) -> list[dict]:
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json={"vector": vector, "limit": top_k, "with_payload": True},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["result"]


if __name__ == "__main__":
    test_text = "Qdrant is a vector similarity search engine."
    print(f"Embedding: '{test_text}'")

    vector = embed(test_text)
    print(f"  dim={len(vector)} ✓")

    if len(vector) != EMBED_DIM:
        print(f"  WARNING: expected dim={EMBED_DIM}, got {len(vector)}", file=sys.stderr)

    print("Upserting point id=1 …")
    upsert(1, vector, {"text": test_text, "source": "smoke-test"})
    print("  upserted ✓")

    print("Searching …")
    results = search(vector)
    for hit in results:
        print(f"  score={hit['score']:.4f}  text={hit['payload'].get('text', '')!r}")

    print("Smoke test passed.")
