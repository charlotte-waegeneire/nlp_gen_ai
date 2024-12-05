from .load_and_process_data import load_and_process_data
from .query import query
from .setup_llm import setup_llm
from .test_different_parameters import test_different_parameters

RAG_PROMPT_TEMPLATE = """You're an assistant who answers questions about electronic products.

STRICT RULES:
1. Before answering, check that the documents provided contain relevant information.
2. If relevant documents are found, give a concise, structured answer.
3. If no relevant documents are found, answer ONLY: "No relevant documents found to answer this question."
4. Use ONLY the information in the documents provided.
5. Answer format:
   - For a question about a specific product: give a paragraph description
   - For a comparison: use bullets (-)
   - Never start your answer with a hyphen

Documents available:
{context}

Question: {question}"""

__all__ = [
    "load_and_process_data",
    "query",
    "setup_llm",
    "test_different_parameters",
    "RAG_PROMPT_TEMPLATE"
]
