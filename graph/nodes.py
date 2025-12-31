"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query
from agents.answering_agent import create_answering_agent


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with context
    """
    # TODO: Implement retrieval logic
    rewritten_query = rewrite_query(state["query"])
    state["context"] = []  # Placeholder
    return state


def answer_node(state: GraphState) -> GraphState:
    """Generate answer from context.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer
    """
    answering_agent = create_answering_agent()
    
    context_str = "\n".join(state.get("context", []))
    result = answering_agent.invoke({
        "context": context_str,
        "question": state["query"]
    })
    
    state["answer"] = result.content
    return state

