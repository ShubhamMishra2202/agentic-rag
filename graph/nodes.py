"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query
from agents.retrieval_agent import create_retrieval_agent_graph
from langchain_core.messages import HumanMessage, ToolMessage


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents using LangGraph agent.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with context
    """
    # Get the retrieval agent
    retrieval_agent = create_retrieval_agent_graph()
    
    # Prepare messages for the agent
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        # Add the query as a human message if not already present
        messages.append(HumanMessage(content=state["query"]))
    
    # Invoke the LangGraph agent
    result = retrieval_agent.invoke({"messages": messages})
    
    # Extract the retrieved documents from tool results, not from agent's summary
    context_list = []
    if result.get("messages"):
        # Look for tool result messages that contain actual chunks
        # Tool results are typically ToolMessage objects
        for msg in result["messages"]:
            # Check if this is a tool result from search_vectorstore
            if isinstance(msg, ToolMessage):
                # This is a tool result - extract the chunks
                if hasattr(msg, 'content') and msg.content:
                    context_list.append(msg.content)
            # Also check for messages with tool_call_id (another way tool results are represented)
            elif hasattr(msg, 'tool_call_id') and hasattr(msg, 'content'):
                context_list.append(msg.content)
        
        # If no tool results found, try to extract from last message
        # (fallback - but this shouldn't happen if agent follows instructions)
        if not context_list:
            last_message = result["messages"][-1]
            retrieved_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            # Only add if it looks like document content, not an answer
            if retrieved_content and "Document" in retrieved_content:
                context_list = [retrieved_content]
        
        state["context"] = context_list
        state["messages"] = result["messages"]  # Update messages with agent's conversation
    else:
        state["context"] = []
    
    return state


def answer_node(state: GraphState) -> GraphState:
    """Generate answer from context with citations.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer
    """
    from agents.answering_agent import generate_answer
    
    context = state.get("context", [])
    question = state.get("query", "")
    
    # Generate answer using the enhanced answering agent
    answer = generate_answer(question, context)
    
    state["answer"] = answer
    return state

