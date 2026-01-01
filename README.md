# Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph that uses agentic workflows to intelligently route queries, retrieve relevant documents, and generate context-aware answers with source citations.

## 🚀 Features

- **Intent Classification**: Automatically classifies queries to determine if document retrieval is needed or if a direct answer suffices
- **Agentic Retrieval**: Uses a specialized retrieval agent with planning capabilities to find the most relevant documents
- **Conversation Memory**: Maintains chat history for coherent follow-up questions and context-aware responses
- **Query Rewriting**: Intelligently rewrites queries to improve retrieval, especially for follow-up questions
- **Source Citations**: Provides transparent source attribution for all information in responses
- **Contradiction Detection**: Identifies and highlights contradictory information across different sources
- **Stop Conditions**: Smart conversation termination based on goodbye messages, repeated questions, and answer completeness
- **Multi-format Support**: Ingest documents from PDFs and web pages

## 🏗️ Architecture

The system uses a LangGraph-based workflow with the following components:

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Intent Classify │ ──► Direct Answer (greetings, meta-questions)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Node   │ ──► Retrieval Agent (with planning & search tools)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Node     │ ──► Answering Agent (with citations & chat history)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Response│
└─────────────────┘
```

### Key Components

1. **Intent Classifier**: Routes queries to either retrieval or direct answer paths
2. **Retrieval Agent**: Uses tools to plan searches and query the vector database
3. **Answering Agent**: Generates answers with citations, handling contradictions and conversation context
4. **Query Rewriter**: Enhances queries for better retrieval, especially for follow-ups
5. **Stop Conditions**: Manages conversation termination intelligently

## 📋 Prerequisites

- Python 3.10+
- Docker and Docker Compose (for Qdrant)
- OpenAI API key

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration (optional - defaults shown)
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=agentic_rag

# Retrieval Configuration (optional - defaults shown)
RELEVANCE_THRESHOLD=0.5
```

### 4. Start Qdrant Vector Database

Using Docker Compose:

```bash
docker-compose up -d
```

This will start Qdrant on:
- HTTP API: `http://localhost:6333`
- gRPC API: `http://localhost:6334`

Verify Qdrant is running:

```bash
curl http://localhost:6333/health
```

### 5. Ingest Documents

Before querying, you need to ingest documents into the vector database. The ingestion pipeline supports:

- **PDF files**: Local PDF documents
- **Web pages**: URLs to web content

Run the ingestion pipeline:

```bash
python3 -m ingest.ingest_pipeline
```

**Note**: You may need to modify the `ingest_pipeline.py` file to specify your own document sources (PDF paths or web URLs) in the `if __name__ == "__main__"` block.

### 6. Run the Application

```bash
python main.py
```

The application will start in interactive mode. Type your questions and press Enter. Type `quit` or `exit` to end the session.

## 📁 Project Structure

```
agentic-rag/
├── agents/                  # Specialized agents
│   ├── answering_agent.py   # Answer generation with citations
│   ├── intent_classifier.py # Query intent classification
│   ├── query_rewriter.py    # Query enhancement for retrieval
│   └── retrieval_agent.py   # Document retrieval agent
├── graph/                   # LangGraph workflow
│   ├── nodes.py             # Graph node implementations
│   ├── rag_graph.py         # Main graph definition
│   └── state.py             # State schema
├── ingest/                  # Document ingestion pipeline
│   ├── chunking.py          # Text chunking strategies
│   ├── ingest_pipeline.py   # Main ingestion pipeline
│   ├── pdf_loader.py        # PDF document loader
│   ├── text_cleaner.py      # Text cleaning utilities
│   └── web_loader.py        # Web page loader
├── utils/                   # Utility functions
│   ├── logging.py           # Logging configuration
│   ├── relevance.py         # Relevance scoring
│   └── stop_conditions.py   # Conversation stop conditions
├── vectorstore/             # Vector database integration
│   ├── collections.py       # Collection management
│   └── qdrant_client.py     # Qdrant client setup
├── config.py                # Configuration settings
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Qdrant service configuration
└── README.md                # This file
```

## 🔧 Configuration

Key configuration options in `config.py`:

- **MODEL_NAME**: LLM model to use (default: `gpt-4o-mini`)
- **TEMPERATURE**: Model temperature (default: `0.3`)
- **CHUNK_SIZE**: Document chunk size (default: `1000`)
- **CHUNK_OVERLAP**: Chunk overlap size (default: `200`)
- **RELEVANCE_THRESHOLD**: Minimum relevance score (default: `0.5`)

## 💡 Usage Examples

### Basic Query

```
You: What is self-attention?
Assistant: [Answer with citations]
```

### Follow-up Question

```
You: How does it work?
Assistant: [Answer that references previous context]
```

### Direct Answer (No Retrieval)

```
You: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking...
```

## 🎯 Key Features Explained

### Intent Classification

The system automatically determines if a query needs document retrieval:
- **Retrieval Required**: Questions about documents, factual queries, technical questions
- **Direct Answer**: Greetings, meta-questions, general conversation

### Agentic Retrieval

The retrieval agent uses a two-tool approach:
1. **create_retrieval_plan**: Analyzes complex queries and creates search strategies
2. **search_vectorstore**: Performs semantic similarity search in the vector database

### Conversation Memory

The system maintains chat history to:
- Understand follow-up questions
- Provide coherent, context-aware responses
- Detect repeated questions

### Contradiction Detection

The answering agent identifies and highlights contradictory information:
- Detects opposing factual claims
- Presents both perspectives
- Attributes sources to each claim
- Explains potential reasons for contradictions

### Stop Conditions

Conversations can end due to:
- **Goodbye messages**: User indicates they're done
- **Repeated questions**: Same question asked again
- **Answer completeness**: System determines the answer is sufficient

## 🔍 How It Works

1. **Query Input**: User submits a query
2. **Intent Classification**: System determines if retrieval is needed
3. **Routing**: Query is routed to either direct answer or retrieval path
4. **Retrieval** (if needed):
   - Query may be rewritten for better retrieval
   - Retrieval agent plans and executes searches
   - Relevant documents are retrieved with similarity scores
5. **Answer Generation**:
   - Answering agent uses retrieved context and chat history
   - Generates answer with source citations
   - Detects and highlights contradictions
6. **Response**: Final answer is returned with sources

## 🧪 Testing

The system includes comprehensive logging. To see detailed logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Dependencies

Key dependencies:
- `langchain` & `langchain-core`: Core LangChain functionality
- `langchain-openai`: OpenAI integration
- `langchain-community`: Community integrations
- `langgraph`: Graph-based workflow orchestration
- `qdrant-client`: Qdrant vector database client
- `sentence-transformers`: Embedding models
- `langchain-huggingface`: HuggingFace model integration
- `langchain-qdrant`: Qdrant vector store integration

See `requirements.txt` for the complete list.
