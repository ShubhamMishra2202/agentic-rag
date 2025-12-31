"""Document ingestion pipeline."""
from typing import List
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from ingest.pdf_loader import load_pdf
from ingest.web_loader import load_web
from ingest.text_cleaner import clean_text
from ingest.chunking import chunk_documents
from vectorstore.qdrant_client import get_qdrant_client
from vectorstore.collections import create_collection
import config

def ingest_documents(source: str, source_type: str = "pdf") -> List[Document]:
    """Run the complete ingestion pipeline.
    
    Args:
        source: Path to file or URL
        source_type: Type of source ("pdf" or "web")
        
    Returns:
        List of processed Document objects
    """
    # Load documents
    if source_type == "pdf":
        documents = load_pdf(source)
        source_name = os.path.basename(source)
    elif source_type == "web":
        documents = load_web(source)
        source_name = source
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    # Clean text
    documents = clean_text(documents)
    
    # Chunk documents
    chunks = chunk_documents(documents)

    # Add metadata to the chunks
    for chunk in chunks:
        chunk.metadata = {
            "source": source_name,
            "source_type": source_type,
            "chunk_id": chunks.index(chunk),
            "chunk_length": len(chunk.page_content)
        }

    # Embed the chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create collection if it doesn't exist (all-MiniLM-L6-v2 has 384 dimensions)
    collection_name = "agentic_rag_docs"
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        client.get_collection(collection_name)
    except Exception:
        # Collection doesn't exist, create it
        create_collection(collection_name=collection_name, vector_size=384)
    
    # Create vectorstore and add documents
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    # Add documents to the vectorstore
    vectorstore.add_documents(chunks)
    
    print(f"Ingested {len(chunks)} chunks into Qdrant")
    print(f"Vectorstore: {vectorstore}")
    return chunks


#test the ingest pipeline
if __name__ == "__main__":
    chunks = ingest_documents("https://scrapfly.io/blog/posts/build-a-documentation-chatbot-that-works-on-any-website", "web")
    
    

