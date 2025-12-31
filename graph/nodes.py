"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query
from agents.retrieval_agent import create_retrieval_agent_graph
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents using LangGraph agent with optional query rewriting.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with context
    """
    # Get the retrieval agent
    retrieval_agent = create_retrieval_agent_graph()
    
    # Prepare messages for the agent
    messages = state.get("messages", [])
    query = state.get("query", "")
    
    # This helps with follow-up questions like "How does it work?" after "What is X?"
    if len(messages) > 0:
        # Extract past conversations for query rewriting
        past_conversations = []
        for msg in messages[-10:]:  # Last 10 messages
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                past_conversations.append({
                    "role": msg.type,
                    "content": msg.content
                })
        
        # Only rewrite if we have conversation history
        if past_conversations:
            try:
                rewritten_query = rewrite_query(query, past_conversations)
                logger.info(f"📝 Query rewriting: '{query}' -> '{rewritten_query}'")
                query = rewritten_query
            except Exception as e:
                logger.warning(f"⚠️  Query rewriting failed, using original query: {e}")
    
    # Add the query as a human message
    if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != query:
        messages.append(HumanMessage(content=query))
    
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
        
        # Filter messages to only keep user messages and final answers (no internal reaoning)
        # Keep only HumanMessage and AIMessage without tool_calls
        filtered_messages = []
        for msg in messages:  # Keep existing messages (user questions and final answers)
            if isinstance(msg, HumanMessage):
                filtered_messages.append(msg)
            elif isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                # Only keep AIMessage that don't have tool calls (final answers, not internal reasoning)
                filtered_messages.append(msg)
        
        state["messages"] = filtered_messages  # Only store user messages and final answers
    else:
        state["context"] = []
    
    return state


def answer_node(state: GraphState) -> GraphState:
    """Generate answer from context with citations and chat history.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer and updated messages
    """
    from agents.answering_agent import generate_answer
    from langchain_core.messages import HumanMessage, AIMessage
    
    context = state.get("context", [])
    question = state.get("query", "")
    messages = state.get("messages", [])
    
    # Filter messages to only keep user messages and final answers
    # Remove any tool messages or internal reasoning that might have slipped through
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            # Only keep AIMessage without tool_calls (final answers, not internal reasoning)
            filtered_messages.append(msg)
    messages = filtered_messages
    
    # Add user question to messages if not already present
    if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != question:
        messages.append(HumanMessage(content=question))
    
    # Generate answer using the enhanced answering agent with chat history
    answer = generate_answer(question, context, chat_history=messages)
    
    # Add AI response to messages (this is a final answer, not internal reasoning)
    messages.append(AIMessage(content=answer))
    
    state["answer"] = answer
    state["messages"] = messages  # Persist chat history (only user messages and final answers)
    return state

