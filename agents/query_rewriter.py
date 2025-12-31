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



