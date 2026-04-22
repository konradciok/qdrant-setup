#!/usr/bin/env python3
"""Create default Qdrant collections wired to vault chunking pipeline."""

import os
import sys
import time
import requests
from pathlib import Path

# Add parent directory to path so we can import vault_qdrant
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from vault_qdrant.collection import ensure_vault_collection

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

QDRANT_URL = f"http://localhost:{os.getenv('QDRANT_HTTP_PORT', '6333')}"


def wait_for_qdrant(max_retries: int = 15) -> bool:
    """Wait for Qdrant to be healthy.

    Args:
        max_retries: Maximum number of health check attempts

    Returns:
        True if Qdrant became healthy, False if timeout
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{QDRANT_URL}/healthz", timeout=3)
            if r.status_code == 200:
                print("✓ Qdrant is ready")
                return True
        except requests.ConnectionError:
            pass

        if attempt < max_retries - 1:
            print(f"  waiting for Qdrant... ({attempt + 1}/{max_retries})")
            time.sleep(2)

    print("✗ Qdrant did not become ready", file=sys.stderr)
    return False


def main() -> int:
    """Initialize Qdrant collections."""
    print(f"Connecting to Qdrant at {QDRANT_URL}")

    if not wait_for_qdrant():
        return 1

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Create vault collection
    print("Bootstrapping vault collection...")
    try:
        ensure_vault_collection(client)
        print("✓ vault collection ready")
    except Exception as e:
        print(f"✗ Failed to create vault collection: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
