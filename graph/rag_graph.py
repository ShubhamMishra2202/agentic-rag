"""Main RAG graph definition."""
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import retrieve_node, answer_node


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
    
    # Add edges
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app

