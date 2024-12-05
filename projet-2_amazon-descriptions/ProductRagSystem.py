from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from typing import Dict, Any
from dotenv import load_dotenv
import os

from functions import (
    load_and_process_data,
    query,
    setup_llm,
    RAG_PROMPT_TEMPLATE
)

# Load environment variables from .env file
load_dotenv()

# Check if API key is properly loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")


class ProductRAGSystem:
    """
    Retrieval-Augmented Generation system for product-related queries.
    """

    def __init__(self, persist_directory: str = "chroma_db") -> None:
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory for storing the vector database
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create prompt template
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Load or create the database
        if not os.path.exists(persist_directory):
            print("Creating new vector database...")
            documents = load_and_process_data()
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            print("Loading existing vector database...")
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

    def setup_llm(
            self,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.0,  # Controls randomness (0=deterministic, 1=creative)
            top_p: float = 1.0  # Controls diversity (0=most diverse, 1=most deterministic)
    ) -> None:
        setup_llm(self, model_name, temperature, top_p)

    def query(
            self,
            question: str,
            verbose: bool = False
    ) -> Dict[str, Any]:
        return query(self, question, verbose)
