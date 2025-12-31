"""Answering agent for generating responses."""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import config


def create_answering_agent():
    """Create an answering agent.
    
    Returns:
        ChatOpenAI instance configured for answering
    """
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on retrieved context."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    return chain

