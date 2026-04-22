#!/usr/bin/env python3
"""Create default Qdrant collections wired to the Ollama qwen3-embedding model."""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

QDRANT_URL = f"http://localhost:{os.getenv('QDRANT_HTTP_PORT', '6333')}"
EMBED_DIM = int(os.getenv("EMBED_DIM", "2560"))

COLLECTIONS = [
    {"name": "documents"},
]


def create_collection(name: str) -> None:
    url = f"{QDRANT_URL}/collections/{name}"
    payload = {
        "vectors": {
            "size": EMBED_DIM,
            "distance": "Cosine",
            "on_disk": False,
        },
        "optimizers_config": {"default_segment_number": 2},
        "replication_factor": 1,
    }
    resp = requests.put(url, json=payload, timeout=10)
    if resp.status_code in (200, 201):
        print(f"  created: {name}")
    elif resp.status_code == 409:
        print(f"  exists:  {name} (skipped)")
    else:
        print(f"  ERROR {resp.status_code} for {name}: {resp.text}", file=sys.stderr)
        sys.exit(1)


def wait_for_qdrant() -> None:
    import time
    for _ in range(15):
        try:
            r = requests.get(f"{QDRANT_URL}/healthz", timeout=3)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        print("  waiting for Qdrant…")
        time.sleep(2)
    print("Qdrant did not become healthy in time", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    wait_for_qdrant()
    print(f"Creating collections (dim={EMBED_DIM}):")
    for col in COLLECTIONS:
        create_collection(col["name"])
    print("Done.")
