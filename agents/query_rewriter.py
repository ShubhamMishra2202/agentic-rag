"""Query rewriter for improving search queries."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import config


def rewrite_query(user_query: str, past_conversations: List[Dict]) -> str:
    """Rewrite a query to improve retrieval.
    
    Args:
        user_query: Original user query
        past_conversations: Past conversations
    Returns:
        Rewritten query string
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rewrite the query to improve document retrieval while preserving the original intent.
If the query is a follow-up question (e.g., "How does it work?", "What are its advantages?"), 
expand it to include context from previous conversations to make it more complete and searchable.
For example, if the previous conversation was about "self-attention" and the current query is "How does it work?",
rewrite it to "How does self-attention work?" to improve retrieval."""),
        ("human", """Past conversations:
{past_conversations}

Current query: {query}

Rewritten query:""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": user_query, "past_conversations": past_conversations})
    
    return result.content


def refine_query_for_retry(original_query: str, retrieved_chunks: List[str]) -> str:
    """Refine query when retrieved chunks are not relevant enough.
    
    This function is called when the initial search returns chunks with low similarity scores.
    It rephrases or expands the query to improve retrieval relevance.
    
    Args:
        original_query: Original user query
        retrieved_chunks: List of retrieved chunks (for context on what didn't work)
        
    Returns:
        Refined query string
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    # Extract a sample of retrieved content to understand what was found
    sample_content = "\n".join([chunk[:200] for chunk in retrieved_chunks[:2]]) if retrieved_chunks else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query refinement assistant. When a search query doesn't retrieve relevant documents 
(based on low similarity scores), you need to rephrase or expand the query to improve retrieval.

Strategies:
1. Make the query more specific by adding context or related terms
2. Break complex queries into key concepts
3. Use synonyms or alternative phrasings
4. Add domain-specific terminology if applicable
5. Expand abbreviations or acronyms

Return ONLY the refined query, no explanations or additional text."""),
        ("human", """Original query: {query}

Sample of what was retrieved (not relevant enough):
{sample_content}

Refine the query to be more specific and improve retrieval. Return only the refined query.""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "query": original_query,
        "sample_content": sample_content[:500] if sample_content else "No content retrieved"
    })
    
    return result.content.strip()



