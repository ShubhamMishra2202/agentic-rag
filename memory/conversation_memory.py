"""Conversation memory management."""
from langchain.memory import ConversationBufferMemory
from typing import List, Dict


class ConversationMemory:
    """Manages conversation history."""
    
    def __init__(self):
        """Initialize conversation memory."""
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation.
        
        Args:
            role: Message role ("human" or "ai")
            content: Message content
        """
        if role == "human":
            self.memory.chat_memory.add_user_message(content)
        elif role == "ai":
            self.memory.chat_memory.add_ai_message(content)
    
    def get_messages(self) -> List[Dict]:
        """Get all conversation messages.
        
        Returns:
            List of message dictionaries
        """
        return self.memory.chat_memory.messages
    
    def clear(self):
        """Clear conversation history."""
        self.memory.clear()

