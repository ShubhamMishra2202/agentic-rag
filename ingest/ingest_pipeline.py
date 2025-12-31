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
    
    qdrant_db_client = get_qdrant_client()
    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=qdrant_db_client,
        collection_name="agentic_rag_docs"
    )
    
    print(f"Ingested {len(chunks)} chunks into Qdrant")
    print(f"Vectorstore: {vectorstore}")
    return chunks



    

