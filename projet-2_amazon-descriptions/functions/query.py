from langchain_community.callbacks.manager import get_openai_callback
from typing import Dict, Any


def query(
        self: "ProductRAGSystem",
        question: str,
        verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute a query and return the answer with metrics.

    Args:
        self: The RAG system instance
        question: User's question
        verbose: Whether to print detailed metrics, default to False

    Returns:
        Dict containing the answer, source documents, and usage metrics
    """
    with get_openai_callback() as cb:
        # Retrieve relevant documents
        relevant_docs = self.db.similarity_search(question, k=4)

        # Prepare context
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            context += f"\nDocument {i}:\n"
            context += f"Title: {doc.metadata['title']}\n"
            context += f"Content: {doc.page_content}\n"
            context += "-" * 50 + "\n"

        # Execute query
        result = self.qa_chain.invoke({"query": question})

        if verbose:
            print(f"\nOpenAI Metrics:")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Estimated cost: ${cb.total_cost:.4f}")

        return {
            "question": question,
            "answer": result["result"],
            "source_documents": [doc.metadata["title"] for doc in relevant_docs],
            "metrics": {
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }
        }
