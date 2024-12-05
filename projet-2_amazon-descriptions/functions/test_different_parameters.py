from langchain.chains import RetrievalQA
from typing import List, Dict, Any


def test_different_parameters(rag_system) -> List[Dict[str, Any]]:
    """
    Test different parameter configurations for the RAG system.

    WARNING - This function will execute multiple queries and consume OpenAI credits.
    It is not used in the final script, but has been provided for demonstration purposes.

    Returns:
        List of dictionaries containing test results for each configuration
    """
    test_questions = [
        "Tell me about OnePlus 7T cases",
        "What are the best waterproof phone cases available?",
        "I need a phone case with card holder, what do you recommend?"
    ]

    configs = [
        {"temperature": 0.0, "top_p": 1.0},  # Most deterministic
        {"temperature": 0.3, "top_p": 0.9},  # Balanced
        {"temperature": 0.7, "top_p": 0.8}  # More creative
    ]

    results = []
    for config in configs:
        print(f"\nTesting with temperature={config['temperature']}, top_p={config['top_p']}")
        print("-" * 80)

        rag_system.setup_llm(
            temperature=config["temperature"],
            top_p=config["top_p"]
        )

        for question in test_questions:
            result = rag_system.query(question)
            results.append({
                "config": config,
                "result": result
            })
            print("-" * 80)

    return results
