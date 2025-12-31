# Agentic RAG

A retrieval-augmented generation system with agentic workflows using LangChain and LangGraph.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

3. Run the application:
```bash
python main.py
```

## Project Structure

- `ingest/`: Document ingestion and processing pipeline
- `vectorstore/`: Vector database integration (Qdrant)
- `agents/`: Specialized agents for retrieval, answering, and query rewriting
- `memory/`: Conversation memory management
- `graph/`: LangGraph workflow definition
- `utils/`: Utility functions for relevance, logging, and stop conditions

