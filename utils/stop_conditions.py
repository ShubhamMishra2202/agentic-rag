"""Stop conditions for agentic workflows."""
from graph.state import GraphState
from typing import Callable


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

