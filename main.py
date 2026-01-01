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
        "answer": "",
        "needs_fallback": False,
        "should_stop": False,
        "stop_reason": "",
        "intent": "",
    }

    print("\n" + "=" * 80)
    print("AGENTIC RAG SYSTEM - Interactive Mode")
    print("=" * 80 + "\n")
    print("Type your questions (or 'quit'/'exit' to end):\n")

    query_count = 0
    while True:
        # Get user input
        query = input("You: ").strip()

        if not query:
            continue

        query_count += 1
        print(f"\n{'─'*80}")
        print(f"Query {query_count}")
        print(f"{'─'*80}\n")

        # Update state with new query
        current_state["query"] = query
        current_state["should_stop"] = False
        current_state["stop_reason"] = ""

        # Invoke graph (messages will persist from previous iterations)
        result = graph.invoke(current_state)

        # Update current_state with result for next iteration
        current_state = result

        print(f"Assistant: {result['answer']}\n")

        # Check if conversation should stop
        if result.get("should_stop", False):
            stop_reason = result.get("stop_reason", "unknown")
            print(f"\n{'─'*80}")
            print(f"Conversation ended. Reason: {stop_reason}")
            print(f"{'─'*80}\n")
            break


if __name__ == "__main__":
    main()
