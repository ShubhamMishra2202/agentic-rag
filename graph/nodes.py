"""Graph nodes for the RAG workflow."""
from graph.state import GraphState
from agents.query_rewriter import rewrite_query, refine_query_for_retry
from agents.retrieval_agent import create_retrieval_agent_graph
from agents.answering_agent import format_final_response
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from utils.stop_conditions import is_goodbye_message, is_repeated_question
import logging
import re
import config

logger = logging.getLogger(__name__)


def intent_classify_node(state: GraphState) -> GraphState:
    """Classify query intent to determine routing.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with intent classification
    """
    from agents.intent_classifier import classify_intent
    
    query = state.get("query", "")
    
    if not query:
        logger.warning("⚠️  Empty query, defaulting to retrieval_required")
        state["intent"] = "retrieval_required"
        return state
    
    # Classify intent
    intent = classify_intent(query)
    state["intent"] = intent
    
    logger.info(f"🎯 Intent classification: '{query[:50]}...' -> {intent}")
    
    return state


def direct_answer_node(state: GraphState) -> GraphState:
    """Generate direct answer for queries that don't require retrieval.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    import config
    
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    # Build conversation context
    conversation_context = ""
    if messages:
        recent_history = messages[-6:]  # Last 6 messages
        for msg in recent_history:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    conversation_context += f"User: {msg.content}\n"
                elif msg.type == "ai":
                    conversation_context += f"Assistant: {msg.content}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question directly without needing to retrieve documents.

This query has been classified as not requiring document retrieval, so you can answer based on:
- General knowledge
- Conversational context
- System capabilities
- Common sense

Be friendly, concise, and helpful. If the question is about the system itself, explain what you can do."""),
        ("human", """Previous conversation:
{conversation_history}

Current question: {query}

Answer:""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "query": query,
        "conversation_history": conversation_context if conversation_context else "No previous conversation."
    })
    
    answer = result.content if hasattr(result, 'content') else str(result)
    
    # Update messages
    if not messages or not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=query))
    messages.append(AIMessage(content=answer))
    
    state["answer"] = answer
    state["messages"] = messages
    state["context"] = []  # No context for direct answers
    state["needs_fallback"] = False
    state["should_stop"] = False
    
    logger.info(f"💬 Direct answer generated for query: '{query[:50]}...'")
    
    return state


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents with relevance checking and retry logic.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with context
    """
    # Initialize stopping flags
    state["should_stop"] = False
    state["stop_reason"] = ""
    
    # Get the retrieval agent
    retrieval_agent = create_retrieval_agent_graph()
    
    # Prepare messages for the agent
    messages = state.get("messages", [])
    original_query = state.get("query", "")
    query = original_query
    
    # Check for goodbye messages
    if is_goodbye_message(query):
        logger.info("👋 Goodbye message detected - stopping conversation")
        state["should_stop"] = True
        state["stop_reason"] = "goodbye"
        state["answer"] = "Thank you! Have a great day!"
        state["context"] = []
        state["needs_fallback"] = False
        return state
    
    # Check for repeated questions
    if is_repeated_question(query, messages):
        logger.info("🔄 Repeated question detected - stopping conversation")
        state["should_stop"] = True
        state["stop_reason"] = "repeated_question"
        state["answer"] = "I've already answered this question. Is there anything else you'd like to know?"
        state["context"] = []
        state["needs_fallback"] = False
        return state
    
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
        # After retry, if still no relevant chunks, stop the conversation
        logger.info("🛑 No relevant chunks found after retry - stopping conversation")
        state["should_stop"] = True
        state["stop_reason"] = "no_relevant_chunks"
        # Set a final answer message with proper formatting
        answer_text = "I don't have enough information in the knowledge base to answer this question."
        state["answer"] = format_final_response(answer_text, [])
        # Add messages for consistency
        if not messages or not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=original_query))
        messages.append(AIMessage(content=state["answer"]))
        state["messages"] = messages
    else:
        state["needs_fallback"] = False
        state["should_stop"] = False
    
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
    
    # Check if we should stop before generating answer
    if state.get("should_stop", False):
        logger.info(f"🛑 Stopping conversation - reason: {state.get('stop_reason', 'unknown')}")
        # Answer should already be set in retrieve_node for stopping conditions
        if not state.get("answer"):
            state["answer"] = "Conversation ended."
        return state
    
    context = state.get("context", [])
    question = state.get("query", "")
    messages = state.get("messages", [])
    needs_fallback = state.get("needs_fallback", False)
    
    # Check if we need fallback (no relevant chunks found)
    if needs_fallback or not context:
        logger.warning("⚠️  Using fallback message - no relevant chunks found")
        answer_text = "I don't have enough information in the knowledge base to answer."
        fallback_answer = format_final_response(answer_text, [])
        
        # Add AI response to messages
        if not messages or not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=fallback_answer))
        
        state["answer"] = fallback_answer
        state["messages"] = messages
        # Don't stop here - let the user try again with a different question
        state["should_stop"] = False
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

