"""Main entry point for the agentic RAG application."""
from graph.rag_graph import create_rag_graph


def main():
    """Run the agentic RAG system."""
    graph = create_rag_graph()
    
    # Example usage
    initial_state = {
        "query": "What is the main topic?",
        "messages": []
    }
    
    result = graph.invoke(initial_state)
    print(result)


if __name__ == "__main__":
    main()

