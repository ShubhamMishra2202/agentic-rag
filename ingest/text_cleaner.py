"""Text cleaning utilities."""
from langchain.schema import Document
from typing import List
import re


def clean_text(documents: List[Document]) -> List[Document]:
    """Clean text in documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of cleaned Document objects
    """
    cleaned = []
    for doc in documents:
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', doc.page_content)
        cleaned_text = cleaned_text.strip()
        
        doc.page_content = cleaned_text
        cleaned.append(doc)
    
    return cleaned

