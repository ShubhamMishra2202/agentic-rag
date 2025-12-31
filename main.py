"""Main entry point for the agentic RAG application."""
from graph.rag_graph import create_rag_graph


def main():
    """Run the agentic RAG system with chat history support."""
    graph = create_rag_graph()
    
    # Initialize with empty messages for chat history
    current_state = {
        "query": "",
        "messages": [],
        "context": [],
        "answer": ""
    }
    
    # Example: Multiple queries in a conversation to demonstrate chat history
    queries = [
        # "How does gut microbiome composition influence cardiotoxicity risk?",
        # "How do anthracyclines affect the heart?"
        "What types of cardiovascular toxicities are associated with cancer treatment?"
    ]
    
    print("\n" + "="*80)
    print("AGENTIC RAG SYSTEM - Chat History Demo")
    print("="*80 + "\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'─'*80}")
        print(f"User: {query}\n")
        
        # Update state with new query
        current_state["query"] = query
        
        # Invoke graph (messages will persist from previous iterations)
        result = graph.invoke(current_state)
        
        # Update current_state with result for next iteration
        current_state = result
        
        print(f"Assistant: {result['answer']}\n")
    
    # Display final conversation history
    print("\n" + "="*80)
    print("CONVERSATION HISTORY SUMMARY")
    print("="*80)
    print(f"Total messages in history: {len(current_state['messages'])}")
    print(f"Message breakdown:")
    for i, msg in enumerate(current_state['messages'], 1):
        msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {i}. [{msg_type}]: {content_preview}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

