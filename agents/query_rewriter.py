"""Query rewriter for improving search queries."""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import config


def rewrite_query(query: str) -> str:
    """Rewrite a query to improve retrieval.
    
    Args:
        query: Original user query
        
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
        ("human", "Query: {query}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query})
    
    return result.content

