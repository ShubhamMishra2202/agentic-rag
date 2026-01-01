"""Main RAG graph definition."""

from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    retrieve_node,
    answer_node,
    intent_classify_node,
    direct_answer_node,
)
from utils.stop_conditions import should_stop_conversation


def route_by_intent(state: GraphState) -> str:
    """Route to appropriate node based on intent classification.

    Args:
        state: Current graph state

    Returns:
        "retrieve" if retrieval is required,
        "direct_answer" if no retrieval needed
    """
    intent = state.get("intent", "")

    if intent == "direct_answer":
        return "direct_answer"
    else:
        # Default to retrieval_required if intent is empty or retrieval_required
        return "retrieve"


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
    """Create and compile the RAG graph with intent routing.

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("intent_classify", intent_classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("direct_answer", direct_answer_node)

    # Set entry point to intent classification
    workflow.set_entry_point("intent_classify")

    # Route based on intent
    workflow.add_conditional_edges(
        "intent_classify",
        route_by_intent,
        {"retrieve": "retrieve", "direct_answer": "direct_answer"},
    )

    # Add conditional edge from retrieve to check if we should stop
    workflow.add_conditional_edges(
        "retrieve", check_should_continue, {"end": END, "answer": "answer"}
    )

    # Add edge from answer to end
    workflow.add_edge("answer", END)

    # Add edge from direct_answer to end
    workflow.add_edge("direct_answer", END)

    # Compile graph
    app = workflow.compile()

    return app
