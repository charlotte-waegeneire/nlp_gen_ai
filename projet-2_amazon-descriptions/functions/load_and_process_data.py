from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Any
import json


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
    with open("../data/meta.jsonl", "r") as f:
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

    print("\nExample loaded documents:")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

    return documents
