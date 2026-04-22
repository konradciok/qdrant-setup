"""Create and manage the `vault` Qdrant collection."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    CollectionStatus,
    FieldIndexParams,
    KeywordIndexParams,
    DatetimeIndexParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

VAULT_COLLECTION = "vault"


def ensure_vault_collection(client: QdrantClient) -> None:
    """Create the vault collection if it doesn't exist.

    Creates a hybrid search-enabled collection with:
    - Dense vector (2560 dim, cosine distance)
    - Sparse vector (BM25 index)
    - Payload indexes for filtering (doc_type, tags, folder, modified_at, status)
    - HNSW config: m=16, ef_construct=200
    - Optimizers: default_segment_number=2, on_disk_payload=True

    Idempotent: Does nothing if collection already exists.

    Args:
        client: QdrantClient instance connected to Qdrant server
    """
    # Check if collection exists
    try:
        collection_info = client.get_collection(VAULT_COLLECTION)
        if collection_info.status == CollectionStatus.GREEN:
            return  # Collection already exists and is healthy
    except Exception:
        pass  # Collection doesn't exist, proceed with creation

    # Create collection with hybrid schema
    client.create_collection(
        collection_name=VAULT_COLLECTION,
        vectors_config=VectorParams(size=2560, distance=Distance.COSINE),
        sparse_vectors_config={
            "sparse": SparseVectorParams(index={"full_scan_threshold": 10000})
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2, on_disk_payload=True
        ),
    )

    # Create payload indexes for filtering
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    """Create indexes on payload fields for efficient filtering.

    Indexes created:
    - doc_type (keyword): Document type filter
    - tags (keyword): Tag-based filtering
    - folder (keyword): Folder/path filtering
    - modified_at (datetime): Temporal filtering
    - status (keyword): Status filtering
    """
    indexes = {
        "doc_type": KeywordIndexParams(),
        "tags": KeywordIndexParams(),
        "folder": KeywordIndexParams(),
        "modified_at": DatetimeIndexParams(),
        "status": KeywordIndexParams(),
    }

    for field_name, index_params in indexes.items():
        try:
            client.create_payload_index(
                collection_name=VAULT_COLLECTION,
                field_name=field_name,
                field_schema=index_params,
            )
        except Exception:
            # Index may already exist; silently continue
            pass
