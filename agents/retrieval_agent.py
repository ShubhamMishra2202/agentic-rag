
"""Retrieval agent for finding relevant documents."""
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Optional, Literal
from vectorstore.qdrant_client import get_qdrant_client
import config
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Collection name used in ingestion pipeline
COLLECTION_NAME = "agentic_rag_docs"


def format_conversation_memory(past_conversations: List[Dict]) -> List[BaseMessage]:
    """Convert conversation dicts to LangChain BaseMessage objects.
    
    Args:
        past_conversations: List of conversation dicts with "role" and "content" keys
        
    Returns:
        List of BaseMessage objects (HumanMessage or AIMessage)
    """
    messages = []
    for conv in past_conversations:
        role = conv.get("role", "").lower()
        content = conv.get("content", "")
        
        if role == "human" or role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "ai" or role == "assistant":
            messages.append(AIMessage(content=content))
    
    return messages


def read_input_and_memory(user_input: str, conversation_memory: Optional[List[Dict]] = None) -> Dict:
    """Read and format user input along with conversation memory.
    
    This function combines the user's current query with past conversation
    history to provide context for retrieval.
    
    Args:
        user_input: Current user query/input
        conversation_memory: Optional list of past conversation dicts
        
    Returns:
        Dictionary with formatted input and chat_history
    """
    chat_history = []
    if conversation_memory:
        chat_history = format_conversation_memory(conversation_memory)
    
    return {
        "input": user_input,
        "chat_history": chat_history
    }


def get_vectorstore():
    """Get or create the Qdrant vectorstore instance.
    
    Returns:
        Qdrant vectorstore instance
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    return vectorstore


@tool
def create_retrieval_plan(user_query: str, conversation_context: str = "") -> str:
    """Create a retrieval plan based on the user query and conversation context.
    
    This tool analyzes the query and creates a structured plan for what to search,
    including key concepts, sub-queries, and search strategy.
    
    Args:
        user_query: The user's current query
        conversation_context: Optional summary of past conversation context
        
    Returns:
        A structured retrieval plan with search queries and strategy
    """
    start_time = time.time()
    
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    context_part = f"\n\nConversation Context: {conversation_context}" if conversation_context else ""
    
    prompt = f"""Analyze the following query and create a retrieval plan.

User Query: {user_query}{context_part}

Create a structured retrieval plan that includes:
1. Key concepts and terms to search for
2. Potential sub-queries if the query is complex
3. Search strategy (e.g., broad search first, then narrow down)
4. Expected number of documents needed

Format your response as a clear, actionable plan."""
    
    response = llm.invoke(prompt)
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("📋 RETRIEVAL PLAN CREATED")
    logger.info(f"   Query: '{user_query}'")
    logger.info(f"   Time taken: {elapsed_time:.3f} seconds")
    logger.info("=" * 80)
    
    return response.content


@tool
def search_vectorstore(query: str, k: int = 5) -> str:
    """Search the vector database for relevant documents.
    
    This tool searches the document collection using semantic similarity
    to find the most relevant chunks for a given query.
    
    Args:
        query: The search query to find relevant documents
        k: Number of documents to retrieve (default: 5)
        
    Returns:
        A formatted string containing the retrieved documents with their content
    """
    start_time = time.time()
    
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        
        elapsed_time = time.time() - start_time
        num_chunks = len(results) if results else 0
        
        # Log the search details
        logger.info("=" * 80)
        logger.info("🔍 VECTOR SEARCH EXECUTED")
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Chunks requested (k): {k}")
        logger.info(f"   Chunks returned: {num_chunks}")
        logger.info(f"   Time taken: {elapsed_time:.3f} seconds")
        logger.info("=" * 80)
        
        if not results:
            return "No relevant documents found."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            formatted_results.append(
                f"Document {i} (Source: {source}):\n{content}\n"
            )
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error searching vectorstore after {elapsed_time:.3f}s: {str(e)}")
        return f"Error searching vectorstore: {str(e)}"


def create_retrieval_agent_graph():
    """Create a LangGraph agent for retrieval using StateGraph.
    
    Returns:
        Compiled LangGraph agent
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    # Bind tools to LLM
    tools = [create_retrieval_plan, search_vectorstore]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create prompt with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval agent that finds relevant documents from a vector database.

CRITICAL REQUIREMENTS:
1. You MUST call search_vectorstore tool for EVERY query - this is MANDATORY.
2. You CANNOT answer questions directly - you must retrieve documents first.
3. After retrieving documents, return them as-is. DO NOT summarize, DO NOT answer questions.
4. The create_retrieval_plan tool is optional - use it only for complex queries that need planning.

Workflow:
1. (Optional) Call create_retrieval_plan if the query is complex and needs analysis
2. (MANDATORY) Call search_vectorstore with the query - you MUST do this for every request
3. Return the RAW retrieved documents exactly as returned by search_vectorstore
4. DO NOT summarize, DO NOT provide explanations, DO NOT answer the question

Your ONLY job is to retrieve relevant documents from the vector database. Return the raw document chunks with their sources."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Define the agent node
    def agent_node(state: MessagesState):
        """Call the LLM with tools."""
        messages = state["messages"]
        
        # Log agent invocation
        if len(messages) > 0:
            last_user_msg = [m for m in messages if isinstance(m, HumanMessage)]
            if last_user_msg:
                logger.info(f"🤖 Agent processing query: '{last_user_msg[-1].content[:100]}...'")
        
        formatted_messages = prompt.invoke({"messages": messages})
        response = llm_with_tools.invoke(formatted_messages.messages)
        
        # Log if tool calls were made
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
            logger.info(f"🔧 Agent calling tools: {', '.join(tool_names)}")
        
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Define conditional edge function
    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if search_vectorstore has been called at least once
        has_searched = False
        for msg in messages:
            # Check for tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get('name') == 'search_vectorstore':
                        has_searched = True
                        break
            # Check for tool result messages
            if hasattr(msg, 'name') and msg.name == 'search_vectorstore':
                has_searched = True
                break
            if hasattr(msg, 'tool_call_id'):
                # This might be a tool result
                has_searched = True
                break
        
        # If there are tool calls, go to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # If we haven't searched yet and no tool calls, force agent to search
        # (The prompt should handle this, but we log a warning)
        if not has_searched:
            logger.warning("⚠️  Agent ended without calling search_vectorstore! This should not happen.")
        
        return "end"
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

