"""Query rewriter for improving search queries."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import config


def rewrite_query(user_query: str,past_conversations: List[Dict]) -> str:
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
        ("system", "Rewrite the query to improve document retrieval while preserving the original intent."),
        ("human", "Query: {query}"),
        ("human", "Past conversations: {past_conversations}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": user_query, "past_conversations": past_conversations})
    
    return result.content



