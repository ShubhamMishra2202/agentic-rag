
"""Retrieval agent for finding relevant documents."""
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Optional, Literal
from vectorstore.qdrant_client import get_qdrant_client
import config


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
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            return "No relevant documents found."
        print(f"Results: {results}")
        
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
        ("system", "You are a retrieval agent that finds relevant documents. "
         "Follow these steps:\n"
         "1. First, use create_retrieval_plan to analyze the user's query and conversation history, "
         "then create a structured retrieval plan.\n"
         "2. Based on the plan, use search_vectorstore to execute searches for relevant documents.\n"
         "3. You may need to perform multiple searches if the query is complex or requires different aspects.\n"
         "4. Adjust the number of results (k parameter) based on query complexity - use higher k for broad queries, lower for specific ones.\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Define the agent node
    def agent_node(state: MessagesState):
        """Call the LLM with tools."""
        messages = state["messages"]
        formatted_messages = prompt.invoke({"messages": messages})
        response = llm_with_tools.invoke(formatted_messages.messages)
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Define conditional edge function
    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        # If there are tool calls, go to tools, otherwise end
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
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


#test the retrieval agent
if __name__ == "__main__":
    retrieval_agent = create_retrieval_agent_graph()
    user_query = "What is self-attention?"
    past_conversations = [
        {"role": "human", "content": "What is transformer model?"}, 
        {"role": "ai", "content": "The transformer model is a type of neural network architecture that is used to process sequential data."}
    ]
    
    # Format conversation memory
    chat_history = format_conversation_memory(past_conversations)
    
    # Invoke the LangGraph agent
    # LangGraph agents expect messages format
    messages = chat_history + [HumanMessage(content=user_query)]
    
    result = retrieval_agent.invoke({"messages": messages})
    print(result)

  # test the search_vectorstore tool
#   result=search_vectorstore.invoke("What is self-attention?")
#   print(result)