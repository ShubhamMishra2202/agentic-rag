"""Intent classifier for routing queries to retrieval or direct answer."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
import config
import logging

logger = logging.getLogger(__name__)


def classify_intent(query: str) -> Literal["retrieval_required", "direct_answer"]:
    """Classify query intent to determine if retrieval is needed.

    Args:
        query: User's query string

    Returns:
        "retrieval_required" if query needs document retrieval,
        "direct_answer" if query can be answered without retrieval
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=0.1,  # Lower temperature for more consistent classification
        api_key=config.OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intent classifier for a RAG (Retrieval-Augmented Generation) system.

Your task is to classify user queries into one of two categories:

1. **retrieval_required**: The query requires searching through a knowledge base/document collection to answer.
   - Questions about specific documents, papers, or ingested content
   - Factual questions that need information from the knowledge base
   - Questions about topics covered in the documents
   - Follow-up questions that reference previous document-based answers
   - Technical questions that need specific information from sources

2. **direct_answer**: The query can be answered without retrieving documents.
   - Greetings (hello, hi, how are you)
   - General conversational queries
   - Questions about the system itself (what can you do, how do you work)
   - Simple clarifications that don't need document context
   - Meta-questions about the conversation

CRITICAL: When in doubt, classify as "retrieval_required" to ensure we don't miss document-based queries.

Respond with ONLY one word: either "retrieval_required" or "direct_answer".""",
            ),
            ("human", "Query: {query}\n\nIntent:"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"query": query})

    intent = result.content.strip().lower()

    # Validate and normalize the response
    if "retrieval" in intent or "required" in intent:
        intent = "retrieval_required"
    elif "direct" in intent or "answer" in intent:
        intent = "direct_answer"
    else:
        # Default to retrieval_required if unclear
        logger.warning(
            f"Unclear intent classification: '{intent}', defaulting to 'retrieval_required'"
        )
        intent = "retrieval_required"

    logger.debug(f"Intent classified: '{query[:50]}...' -> {intent}")

    return intent
