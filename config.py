"""Configuration settings for the agentic RAG system."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Collection Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agentic_rag")

# Model Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.3

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4")) 

