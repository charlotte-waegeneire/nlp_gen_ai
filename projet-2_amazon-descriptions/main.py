from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Check if API key is properly loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Enhanced prompt template for product-related queries
RAG_PROMPT_TEMPLATE = """You're an assistant who answers questions about electronic products.

STRICT RULES:
1. Before answering, check that the documents provided contain relevant information.
2. If relevant documents are found, give a concise, structured answer.
3. If no relevant documents are found, answer ONLY: "No relevant documents found to answer this question.
4. Use ONLY the information in the documents provided.
5. Answer format:
   - For a question about a specific product: give a paragraph description
   - For a comparison: use bullets (-)
   - Never start your answer with a hyphen

Documents available:
{context}

Question: {question}"""


def load_and_process_data() -> List[Document]:
    """
    Load and process product data from JSONL file into document chunks.

    Returns:
        List[Document]: List of processed document chunks with metadata.
    """
    documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Size of each text chunk
        chunk_overlap=128  # Overlap between chunks to maintain context
    )

    def clean_metadata_value(value: Any) -> str:
        """
        Clean metadata values to ensure ChromaDB compatibility.

        Args:
            value: Any metadata value to clean

        Returns:
            str: Cleaned metadata value
        """
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return str(value)

    count = 0
    with open("data/meta.jsonl", "r") as f:
        for line in f:
            count += 1
            data = json.loads(line)

            # Combine all relevant product information
            text = ""
            if data.get("title"):
                text += f"Title: {data['title']}\n"
            if data.get("description"):
                text += f"Description: {' '.join(data['description'])}\n"
            if data.get("features"):
                text += f"Features: {' '.join(data['features'])}\n"
            if data.get("details"):
                text += f"Details: {', '.join([f'{k}: {str(v)}' for k, v in data['details'].items()])}\n"

            # Clean and prepare metadata
            metadata = {
                "title": clean_metadata_value(data.get("title")),
                "main_category": clean_metadata_value(data.get("main_category")),
                "store": clean_metadata_value(data.get("store")),
                "average_rating": clean_metadata_value(data.get("average_rating")),
                "rating_number": clean_metadata_value(data.get("rating_number"))
            }

            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)

    print(f"Loaded {count} products, created {len(documents)} document chunks")
    # Display examples for verification
    print("\nExample loaded documents:")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


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

    def setup_llm(self,
                  model_name: str = "gpt-3.5-turbo",
                  temperature: float = 0.0,  # Controls randomness (0=deterministic, 1=creative)
                  top_p: float = 1.0  # Controls diversity of token selection
                  ) -> None:
        """
        Configure the LLM with specified parameters.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Controls randomness in the output
            top_p: Controls diversity in token selection
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p
        )

        # Create RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Number of relevant documents to retrieve
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute a query and return the answer with metrics.

        Args:
            question: User's question
            verbose: Whether to print detailed metrics

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
                "source_documents": [doc.metadata['title'] for doc in relevant_docs],
                "metrics": {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }


def test_different_parameters() -> List[Dict[str, Any]]:
    """
    Test different parameter configurations for the RAG system.

    WARNING - This function will execute multiple queries and consume OpenAI credits.
    It is not used in the final script, but has been provided for demonstration purposes.

    Returns:
        List of dictionaries containing test results for each configuration
    """
    rag_system = ProductRAGSystem()  # Created once

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


def main() -> None:
    """
    Main function to demonstrate the RAG system's capabilities.
    Tests the system with various questions and displays results.
    """
    # Create and configure RAG system
    rag_system = ProductRAGSystem()
    rag_system.setup_llm()

    # Test questions including a non-relevant query to verify document usage
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

        # Only display referenced documents if the answer isn't "No relevant documents..."
        if not result['answer'].startswith("No relevant documents"):
            print("\nReferenced Documents:")
            # Remove duplicates and display unique titles
            unique_docs = list(set(result['source_documents']))
            for doc in unique_docs:
                print(f"- {doc}")

        print(f"\nMetrics:")
        print(f"Tokens: {result['metrics']['total_tokens']}")
        print(f"Cost: ${result['metrics']['total_cost']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
