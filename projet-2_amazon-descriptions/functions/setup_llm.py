from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def setup_llm(
        self: "ProductRagSystem",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,  # Controls randomness (0=deterministic, 1=creative)
        top_p: float = 1.0  # Controls diversity (0=most diverse, 1=most deterministic)
) -> None:
    """
    Configure the LLM with specified parameters.

    Args:
        self: ProductRagSystem instance
        model_name: Name of the OpenAI model to use, default to "gpt-3.5-turbo"
        temperature: Controls randomness in the output, default to 0.0
        top_p: Controls diversity in token selection, default to 1.0
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
