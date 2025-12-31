"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query, refine_query_for_retry
from agents.retrieval_agent import create_retrieval_agent_graph
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import logging
import re
import config

logger = logging.getLogger(__name__)


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents with relevance checking and retry logic.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with context
    """
    # Get the retrieval agent
    retrieval_agent = create_retrieval_agent_graph()
    
    # Prepare messages for the agent
    messages = state.get("messages", [])
    original_query = state.get("query", "")
    query = original_query
    
    # Initial query rewriting for follow-up questions
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
    
    # First retrieval attempt
    search_messages = [HumanMessage(content=query)]
    result = retrieval_agent.invoke({"messages": search_messages})
    
    # Extract chunks and scores from tool results
    context_list = []
    chunks_with_scores = []
    
    if result.get("messages"):
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                content = msg.content
                if content and "Document" in content:
                    context_list.append(content)
                    # Parse scores from the formatted string
                    # Format: "Document X (Source: Y, Score: 0.1234):\ncontent"
                    score_matches = re.findall(r'Score: ([\d.]+)', content)
                    if score_matches:
                        for score_str in score_matches:
                            chunks_with_scores.append(float(score_str))
    
    # Check relevance: sort scores and check top score
    relevance_threshold = getattr(config, 'RELEVANCE_THRESHOLD', 0.7)
    is_relevant = False
    
    if chunks_with_scores:
        # Sort scores in descending order
        sorted_scores = sorted(chunks_with_scores, reverse=True)
        top_score = sorted_scores[0]
        
        logger.info(f"📊 Relevance Check:")
        logger.info(f"   Top similarity score: {top_score:.4f}")
        logger.info(f"   Threshold: {relevance_threshold:.4f}")
        
        if top_score >= relevance_threshold:
            is_relevant = True
            logger.info(f"✅ Chunks are relevant (score {top_score:.4f} >= {relevance_threshold:.4f})")
        else:
            logger.warning(f"⚠️  Chunks are NOT relevant (score {top_score:.4f} < {relevance_threshold:.4f})")
    elif context_list:
        # If we have chunks but no scores, assume relevant (fallback)
        logger.warning("⚠️  No scores found in chunks, assuming relevant")
        is_relevant = True
    
    # If not relevant, refine query and retry once
    if not is_relevant and context_list:
        logger.info("🔄 Refining query and retrying...")
        
        try:
            refined_query = refine_query_for_retry(original_query, context_list)
            logger.info(f"📝 Refined query: '{original_query}' -> '{refined_query}'")
            
            # Retry with refined query
            retry_messages = [HumanMessage(content=refined_query)]
            retry_result = retrieval_agent.invoke({"messages": retry_messages})
            
            # Extract retry chunks and scores
            retry_context = []
            retry_scores = []
            
            for msg in retry_result.get("messages", []):
                if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                    content = msg.content
                    if content and "Document" in content:
                        retry_context.append(content)
                        score_matches = re.findall(r'Score: ([\d.]+)', content)
                        if score_matches:
                            for score_str in score_matches:
                                retry_scores.append(float(score_str))
            
            # Check relevance of retry results
            if retry_scores:
                retry_sorted = sorted(retry_scores, reverse=True)
                retry_top_score = retry_sorted[0]
                
                logger.info(f"📊 Retry Relevance Check:")
                logger.info(f"   Top similarity score: {retry_top_score:.4f}")
                logger.info(f"   Threshold: {relevance_threshold:.4f}")
                
                if retry_top_score >= relevance_threshold:
                    is_relevant = True
                    context_list = retry_context
                    logger.info(f"✅ Retry successful - chunks are relevant (score {retry_top_score:.4f})")
                else:
                    logger.warning(f"⚠️  Retry still NOT relevant (score {retry_top_score:.4f} < {relevance_threshold:.4f})")
                    context_list = []  # Clear for fallback
            elif retry_context:
                # Retry got chunks but no scores - use them
                context_list = retry_context
                is_relevant = True
            else:
                # No results on retry
                context_list = []
        except Exception as e:
            logger.error(f"❌ Query refinement/retry failed: {e}")
            context_list = []  # Fallback
    
    # Set context and flag for fallback
    state["context"] = context_list
    
    # Mark if we need fallback (no relevant chunks found)
    if not is_relevant or not context_list:
        state["needs_fallback"] = True
    else:
        state["needs_fallback"] = False
    
    # Filter messages to only keep user messages and final answers
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            filtered_messages.append(msg)
    
    state["messages"] = filtered_messages
    return state


def answer_node(state: GraphState) -> GraphState:
    """Generate answer from context with citations and chat history.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer and updated messages
    """
    from agents.answering_agent import generate_answer
    
    context = state.get("context", [])
    question = state.get("query", "")
    messages = state.get("messages", [])
    needs_fallback = state.get("needs_fallback", False)
    
    # Check if we need fallback (no relevant chunks found)
    if needs_fallback or not context:
        logger.warning("⚠️  Using fallback message - no relevant chunks found")
        fallback_answer = "I don't have enough information in the knowledge base to answer."
        
        # Add AI response to messages
        if not messages or not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=fallback_answer))
        
        state["answer"] = fallback_answer
        state["messages"] = messages
        return state
    
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

