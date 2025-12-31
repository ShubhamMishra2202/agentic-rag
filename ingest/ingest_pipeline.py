"""Document ingestion pipeline."""
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
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
    elif source_type == "web":
        documents = load_web(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    # Clean text
    documents = clean_text(documents)
    
    # Chunk documents
    chunks = chunk_documents(documents)

    # Add metadata to the chunks
    for chunk in chunks:
        chunk.metadata = {
            "source": source,
            "source_type": source_type,
            "chunk_index": chunks.index(chunk)
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
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    # Add documents to the vectorstore
    vectorstore.add_documents(chunks)
    
    print(f"Ingested {len(chunks)} chunks into Qdrant")
    print(f"Vectorstore: {vectorstore}")
    return chunks


#test the ingest pipeline
if __name__ == "__main__":
    chunks = ingest_documents("/home/shubhammishra/Downloads/attention_is_all_you_need.pdf", "pdf")
    
    

