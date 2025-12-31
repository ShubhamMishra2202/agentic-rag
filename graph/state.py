"""State definition for the RAG graph."""
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer function to add messages to state."""
    return left + right


class GraphState(TypedDict):
    """State schema for the RAG graph."""
    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    answer: str

