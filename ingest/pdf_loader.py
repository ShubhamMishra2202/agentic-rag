"""PDF document loader."""
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    """Load and parse a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
