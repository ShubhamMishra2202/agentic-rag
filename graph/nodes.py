"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query
from agents.answering_agent import create_answering_agent
from agents.retrieval_agent import create_retrieval_agent_graph
from langchain_core.messages import HumanMessage


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
    
    # Extract the retrieved documents from the agent's response
    # The agent's final message should contain the retrieved documents
    if result.get("messages"):
        last_message = result["messages"][-1]
        retrieved_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Parse the retrieved documents and add to context
        # This assumes the agent returns formatted document strings
        context_list = [retrieved_content] if retrieved_content else []
        state["context"] = context_list
        state["messages"] = result["messages"]  # Update messages with agent's conversation
    else:
        state["context"] = []
    
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

