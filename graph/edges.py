"""Graph edges for the RAG workflow."""
from graph.state import GraphState


def should_continue(state: GraphState) -> str:
    """Determine the next node to execute.
    
    Args:
        state: Current graph state
        
    Returns:
        Name of the next node
    """
    # Simple linear flow: retrieve -> answer
    if not state.get("context"):
        return "retrieve"
    return "answer"


def should_end(state: GraphState) -> str:
    """Determine if the workflow should end.
    
    Args:
        state: Current graph state
        
    Returns:
        "end" if workflow should terminate, otherwise "continue"
    """
    if state.get("answer"):
        return "end"
    return "continue"

