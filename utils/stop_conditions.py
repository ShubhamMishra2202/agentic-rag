"""Stop conditions for agentic workflows."""

from graph.state import GraphState
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
import logging

logger = logging.getLogger(__name__)


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
        r"\b(goodbye|bye|see you|farewell)\b",
        r"\b(thanks|thank you|thx)\b",
        r"\b(done|finished|that\'?s all|that\'?s it)\b",
        r"\b(no more|nothing else|that\'?s everything)\b",
        r"\b(exit|quit|stop)\b",
    ]

    for pattern in goodbye_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


def is_repeated_question(
    query: str, messages: list, similarity_threshold: float = 0.85
) -> bool:
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
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
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
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

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


def is_answer_complete(question: str, answer: str) -> bool:
    """Check if the answer fully resolves the question using LLM.

    Args:
        question: User's original question
        answer: Generated answer text (without Sources section)

    Returns:
        True if answer is complete/fully resolved, False otherwise
    """
    if not question or not answer:
        return False

    return _is_answer_complete_llm(question, answer)


def _is_answer_complete_llm(question: str, answer: str) -> bool:
    """Use LLM to determine if answer is complete.

    Args:
        question: User's original question
        answer: Generated answer text

    Returns:
        True if answer is complete, False otherwise
    """
   

    try:
        llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0.1,  # Low temperature for consistent evaluation
            api_key=config.OPENAI_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an evaluator that determines if an answer fully resolves a question.

Analyze whether the answer:
1. Directly addresses all parts of the question
2. Provides sufficient detail and completeness
3. Doesn't leave critical aspects unanswered
4. Is conclusive rather than partial or tentative

Respond with ONLY "YES" if the answer is complete and fully resolved, or "NO" if it's incomplete or partial.""",
                ),
                (
                    "human",
                    """Question: {question}

Answer: {answer}

Is this answer complete and fully resolved? (YES/NO):""",
                ),
            ]
        )

        chain = prompt | llm
        result = chain.invoke({"question": question, "answer": answer})

        response = (
            result.content.strip().upper()
            if hasattr(result, "content")
            else str(result).strip().upper()
        )
        is_complete = response.startswith("YES")

        logger.debug(
            f"Answer completeness check: {'Complete' if is_complete else 'Incomplete'}"
        )
        return is_complete

    except Exception as e:
        logger.warning(f"Error checking answer completeness with LLM: {e}")
        # On error, assume incomplete to be safe (don't add closing message)
        return False
