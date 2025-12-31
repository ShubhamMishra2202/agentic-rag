"""Web document loader."""
from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain.schema import Document


def load_web(url: str) -> List[Document]:
    """Load and parse a web page.
    
    Args:
        url: URL of the web page
        
    Returns:
        List of Document objects
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

