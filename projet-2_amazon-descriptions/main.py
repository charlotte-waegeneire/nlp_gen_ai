from ProductRagSystem import ProductRAGSystem


def main() -> None:
    """
    Main function to demonstrate the RAG system's capabilities.
    Tests the system with various questions and displays results.
    """
    rag_system = ProductRAGSystem()
    rag_system.setup_llm()

    test_questions = [
        "Tell me about OnePlus 7T cases.",
        "What's the best Apple Watch band?",
        "How can I cook my potatoes?"  # Non-relevant question
    ]

    # Test each question
    for question in test_questions:
        result = rag_system.query(question)
        print("\n" + "=" * 50)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")

        if not result["answer"].startswith("No relevant documents"):
            print("\nReferenced Documents:")
            unique_docs = list(set(result["source_documents"]))
            for doc in unique_docs:
                print(f"- {doc}")

        print(f"Tokens: {result['metrics']['total_tokens']}")
        print(f"Cost: ${result['metrics']['total_cost']:.4f}")


if __name__ == "__main__":
    main()
