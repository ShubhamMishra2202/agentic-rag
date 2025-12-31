"""Text chunking utilities."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import config


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

