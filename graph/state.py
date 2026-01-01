"""State definition for the RAG graph."""

from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import BaseMessage


def add_messages(
    left: List[BaseMessage], right: List[BaseMessage]
) -> List[BaseMessage]:
    """Reducer function to add messages to state."""
    return left + right


class GraphState(TypedDict):
    """State schema for the RAG graph."""

    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    answer: str
    needs_fallback: bool  # Flag to indicate if fallback message should be used
    should_stop: bool  # Flag to indicate if conversation should stop
    stop_reason: str  # Reason for stopping (e.g., "goodbye", "repeated_question", "no_relevant_chunks")
    intent: Literal[
        "retrieval_required", "direct_answer", ""
    ]  # Query intent classification
