"""Qdrant client setup."""
from qdrant_client import QdrantClient
import config


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client.
    
    Returns:
        QdrantClient instance
    """
    if config.QDRANT_API_KEY:
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
    else:
        client = QdrantClient(url=config.QDRANT_URL)
    
    return client

