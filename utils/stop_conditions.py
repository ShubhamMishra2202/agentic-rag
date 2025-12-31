"""Stop conditions for agentic workflows."""
from graph.state import GraphState
from typing import Callable
import re
from langchain_core.messages import HumanMessage


def has_answer(state: GraphState) -> bool:
    """Check if state has an answer.
    
    Args:
        state: Current graph state
        
    Returns:
        True if answer exists, False otherwise
    """
    return bool(state.get("answer"))


def has_context(state: GraphState) -> bool:
    """Check if state has context.
    
    Args:
        state: Current graph state
        
    Returns:
        True if context exists, False otherwise
    """
    context = state.get("context", [])
    return bool(context and len(context) > 0)


def is_goodbye_message(query: str) -> bool:
    """Check if the user query indicates they want to end the conversation.
    
    Args:
        query: User's query string
        
    Returns:
        True if query contains goodbye/thanks/done phrases, False otherwise
    """
    if not query:
        return False
    
    # Normalize query to lowercase for comparison
    query_lower = query.lower().strip()
    
    # Goodbye phrases
    goodbye_patterns = [
        r'\b(goodbye|bye|see you|farewell)\b',
        r'\b(thanks|thank you|thx)\b',
        r'\b(done|finished|that\'?s all|that\'?s it)\b',
        r'\b(no more|nothing else|that\'?s everything)\b',
        r'\b(exit|quit|stop)\b',
    ]
    
    for pattern in goodbye_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False


def is_repeated_question(query: str, messages: list, similarity_threshold: float = 0.85) -> bool:
    """Check if the current question is a repeat of a previous question.
    
    Args:
        query: Current user query
        messages: List of previous messages in the conversation
        similarity_threshold: Threshold for considering questions similar (0.0-1.0)
        
    Returns:
        True if question is repeated, False otherwise
    """
    if not query or not messages:
        return False
    
    # Normalize current query
    query_normalized = _normalize_text(query)
    
    # Check against previous human messages
    previous_queries = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
            previous_queries.append(_normalize_text(msg.content))
    
    # If no previous queries, can't be a repeat
    if not previous_queries:
        return False
    
    # Check similarity with previous queries
    for prev_query in previous_queries:
        similarity = _calculate_similarity(query_normalized, prev_query)
        if similarity >= similarity_threshold:
            return True
    
    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text (lowercase, stripped, punctuation removed)
    """
    if not text:
        return ""
    
    # Convert to lowercase and strip
    normalized = text.lower().strip()
    
    # Remove punctuation and extra whitespace
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts using word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Split into words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def should_stop_conversation(state: GraphState) -> bool:
    """Check if conversation should stop based on all stopping conditions.
    
    Args:
        state: Current graph state
        
    Returns:
        True if conversation should stop, False otherwise
    """
    # Check if should_stop flag is already set
    if state.get("should_stop", False):
        return True
    
    return False


def max_iterations_reached(iterations: int, max_iter: int = 10) -> Callable:
    """Create a stop condition for maximum iterations.
    
    Args:
        iterations: Current iteration count
        max_iter: Maximum allowed iterations
        
    Returns:
        Callable that checks if max iterations reached
    """
    def check(state: GraphState) -> bool:
        # TODO: Track iterations in state
        return iterations >= max_iter
    
    return check

