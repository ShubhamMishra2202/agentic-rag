"""Main RAG graph definition."""
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import retrieve_node, answer_node
from utils.stop_conditions import should_stop_conversation


def check_should_continue(state: GraphState) -> str:
    """Check if workflow should continue or end.
    
    Args:
        state: Current graph state
        
    Returns:
        "end" if should stop, "answer" if should continue
    """
    if should_stop_conversation(state):
        return "end"
    return "answer"


def create_rag_graph():
    """Create and compile the RAG graph.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add conditional edge from retrieve to check if we should stop
    workflow.add_conditional_edges(
        "retrieve",
        check_should_continue,
        {
            "end": END,
            "answer": "answer"
        }
    )
    
    # Add edge from answer to end
    workflow.add_edge("answer", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app

