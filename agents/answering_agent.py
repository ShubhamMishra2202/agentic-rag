"""Answering agent for generating responses."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_retrieved_chunks(context_str: str) -> list:
    """Parse retrieved chunks to extract structured information.

    Args:
        context_str: Formatted string with documents from retrieval agent

    Returns:
        List of dicts with 'content' and 'source' keys
    """
    chunks = []
    if not context_str or context_str.strip() == "":
        return chunks

    # Split by document separator
    doc_sections = context_str.split("Document ")

    for section in doc_sections:
        if not section.strip():
            continue

        # Extract source and content
        # Format: "Document X (Source: source_name, Score: 0.1234):\ncontent"
        if "(Source:" in section:
            parts = section.split("(Source:", 1)
            if len(parts) == 2:
                source_part = parts[1].split("):", 1)
                if len(source_part) == 2:
                    # Extract source name (may include Score, so split on comma)
                    source_with_score = source_part[0].strip()
                    # If there's a comma, take only the part before it (the source name)
                    if "," in source_with_score:
                        source = source_with_score.split(",")[0].strip()
                    else:
                        source = source_with_score
                    content = source_part[1].strip()
                    chunks.append({"source": source, "content": content})
        else:
            # Fallback: treat entire section as content
            chunks.append({"source": "Unknown", "content": section.strip()})

    return chunks


def extract_sources_from_chunks(parsed_chunks: list) -> list:
    """Extract unique sources from parsed chunks.

    Args:
        parsed_chunks: List of chunk dicts with 'source' and 'content' keys

    Returns:
        List of unique source names
    """
    sources = []
    seen_sources = set()

    for chunk in parsed_chunks:
        source = chunk.get("source", "Unknown")
        if source and source not in seen_sources and source != "Unknown":
            sources.append(source)
            seen_sources.add(source)

    return sources


def format_final_response(answer: str, sources: list) -> str:
    """Format the final response with answer and sources.

    Args:
        answer: The generated answer text
        sources: List of source names

    Returns:
        Formatted response string with Answer and Sources sections
    """
    formatted = f"Answer:\n{answer}\n\n"

    if sources:
        formatted += "Sources:\n"
        for i, source in enumerate(sources, 1):
            formatted += f"{i}. {source}\n"
    else:
        formatted += "Sources:\nNo relevant sources found in the knowledge base.\n"

    return formatted


def create_answering_agent():
    """Create an answering agent with citation support and chat history.

    Returns:
        Chain configured for answering with citations and conversation context
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions based on retrieved context and previous conversation.

PRIORITY ORDER FOR ANSWERING:
1. Priority 1: Relevant conversation history - Use previous conversation context to understand follow-up questions and maintain coherence.
2. Priority 2: Vector DB chunks - Base your answer primarily on the retrieved document chunks from the knowledge base.

CRITICAL INSTRUCTIONS:
1. Generate ONLY the answer explanation - do NOT include a "Sources:" section (that will be added automatically).
2. Always cite your sources using the format [Source: <source_name>] when referencing information from the context.
3. Extract the source name from the context format "Document X (Source: <source_name>)".
4. Use the conversation history to understand context and provide coherent, follow-up answers.
5. If the user asks a follow-up question, reference previous answers when relevant.
6. CONTRADICTION DETECTION AND HANDLING:
   - Carefully analyze all retrieved chunks for contradictory information about the same topic or fact.
   - A contradiction occurs when different sources make opposing factual claims about the same subject that cannot both be true.
   - If you detect contradictions:
     * Explicitly state: "Note: The retrieved documents contain contradictory information."
     * Identify the specific conflicting claims clearly
     * Present both perspectives side-by-side
     * Indicate which sources [Source: X] support which claims
     * If possible, explain why the contradiction might exist (different contexts, dates, perspectives, or evolving information)
     * Help the user understand the uncertainty rather than presenting one view as definitive
   - Be especially careful when chunks make opposing factual claims, numerical discrepancies, or conflicting definitions about the same subject.
7. If no context is provided or context is empty, answer gracefully:
   - State: "I don't have enough information from the retrieved documents to answer this question."
   - Suggest: "Please try rephrasing your query or ensure documents have been ingested into the system."
8. Base your answer ONLY on the provided context. Do not use external knowledge beyond what's in the context.
9. If the context doesn't contain enough information to fully answer the question, say so clearly.
10. Format your answer clearly with proper citations for each claim made.

Example citation format:
- "According to the documents [Source: paper.pdf], self-attention is..."
- "The retrieved information [Source: article.pdf] indicates that..."

Always be transparent about limitations and contradictions in the source material.""",
            ),
            (
                "human",
                """Previous Conversation:
{conversation_history}

Current Context:
{context}

Question: {question}""",
            ),
        ]
    )

    chain = prompt | llm
    return chain


def generate_answer(
    question: str, retrieved_chunks: list, chat_history: list = None
) -> str:
    """Generate answer from retrieved chunks with citations and chat history.

    Args:
        question: User's question
        retrieved_chunks: List of retrieved chunk strings from retrieval agent
        chat_history: Optional list of previous messages for context

    Returns:
        Final answer with citations
    """
    # Combine all chunks into context string
    context_str = "\n".join(retrieved_chunks) if retrieved_chunks else ""

    # Check for empty context
    if not context_str or context_str.strip() == "":
        logger.warning(
            f"No context provided for question '{question}' - answering gracefully"
        )
        answer_text = "I don't have enough information from the retrieved documents to answer this question. Please try rephrasing your query or ensure documents have been ingested into the system."
        return format_final_response(answer_text, [])

    # Parse chunks to extract sources
    parsed_chunks = parse_retrieved_chunks(context_str)
    sources = extract_sources_from_chunks(parsed_chunks)

    logger.info(
        f"Generating answer: {len(retrieved_chunks)} chunks, {len(parsed_chunks)} parsed docs, {len(sources)} sources"
    )

    # Build conversation context from chat history
    conversation_context = ""
    if chat_history:
        # Extract last N turns for context (avoid token limits)
        # Keep last 6 messages (approximately 3 Q&A pairs)
        recent_history = chat_history[-6:]
        for msg in recent_history:
            if hasattr(msg, "type"):
                if msg.type == "human":
                    conversation_context += f"Previous question: {msg.content}\n"
                elif msg.type == "ai":
                    conversation_context += f"Previous answer: {msg.content}\n"

    # Create answering agent with chat history support
    answering_agent = create_answering_agent()

    result = answering_agent.invoke(
        {
            "context": context_str,
            "question": question,
            "conversation_history": (
                conversation_context
                if conversation_context
                else "No previous conversation."
            ),
        }
    )

    answer_text = result.content if hasattr(result, "content") else str(result)

    # Format final response with answer and sources
    final_response = format_final_response(answer_text, sources)

    logger.info(f"Answer generated: {len(answer_text)} chars, {len(sources)} sources")

    return final_response
