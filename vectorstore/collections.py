"""Qdrant collection management."""
from qdrant_client.models import Distance, VectorParams
from vectorstore.qdrant_client import get_qdrant_client
import config


def create_collection(collection_name: str = None, vector_size: int = 1536):
    """Create a Qdrant collection.
    
    Args:
        collection_name: Name of the collection
        vector_size: Size of the embedding vectors
    """
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    
    client = get_qdrant_client()
    
    # Check if collection already exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
        return
    except Exception:
        pass  # Collection doesn't exist, create it
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{collection_name}' with vector size {vector_size}")

