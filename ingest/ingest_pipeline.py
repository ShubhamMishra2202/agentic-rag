"""Document ingestion pipeline."""
from typing import List
from langchain_core.documents import Document
from ingest.pdf_loader import load_pdf
from ingest.web_loader import load_web
from ingest.text_cleaner import clean_text
from ingest.chunking import chunk_documents


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
    
    return chunks

