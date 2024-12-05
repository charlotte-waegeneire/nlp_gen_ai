import streamlit as st

from ProductRagSystem import ProductRAGSystem


def app():
    st.title("Amazon Product Descriptions")
    st.markdown(
        "Welcome to the Amazon Product Descriptions app! This app demonstrates how to use a RAG system to answer questions about Amazon product descriptions.")
    st.write(
        "To get started, please write down a question in the text box below and click the ASK button. The RAG system will then provide an answer based on the available Amazon product descriptions.")

    question = st.text_input("Ask a question:")

    rag_system = ProductRAGSystem()
    rag_system.setup_llm()

    st.warning("Please note that this app uses a live OpenAI API and may consume credits.")

    if st.button("ASK"):
        result = rag_system.query(question)

        st.write(f"{result['answer']}\n")

        if not result['answer'].startswith("No relevant documents"):
            st.write("Referenced Documents:")
            for doc in result['source_documents']:
                st.write(f"- {doc}")

        st.write(f"Tokens: {result['metrics']['total_tokens']}")
        st.write(f"Cost: ${result['metrics']['total_cost']:.4f}")


if __name__ == "__main__":
    app()
