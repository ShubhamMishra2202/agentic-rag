"""Configuration settings for the agentic RAG system."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agentic_rag")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

