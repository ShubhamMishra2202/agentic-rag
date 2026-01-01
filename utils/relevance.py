"""Relevance scoring utilities."""

from typing import List, Dict


def calculate_relevance_score(query: str, document: str) -> float:
    """Calculate relevance score between query and document.

    Args:
        query: User query
        document: Document text

    Returns:
        Relevance score between 0 and 1
    """
    # TODO: Implement proper relevance scoring
    # This is a placeholder
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())

    if not query_words:
        return 0.0

    intersection = query_words.intersection(doc_words)
    score = len(intersection) / len(query_words)

    return min(score, 1.0)


def rank_documents(query: str, documents: List[str]) -> List[Dict]:
    """Rank documents by relevance to query.

    Args:
        query: User query
        documents: List of document texts

    Returns:
        List of documents with relevance scores, sorted by score
    """
    scored_docs = []
    for doc in documents:
        score = calculate_relevance_score(query, doc)
        scored_docs.append({"document": doc, "score": score})

    # Sort by score descending
    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    return scored_docs
