from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict
from dotenv import load_dotenv
import os
import json

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Vérifier que la clé API est bien chargée
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La clé API OpenAI n'a pas été trouvée dans les variables d'environnement")

# Template du prompt amélioré
RAG_PROMPT_TEMPLATE = """Tu es un assistant qui répond aux questions sur les produits électroniques.

RÈGLES STRICTES:
1. Avant de répondre, vérifie si les documents fournis contiennent des informations pertinentes
2. Si des documents pertinents sont trouvés, donne une réponse concise et structurée
3. Si aucun document pertinent n'est trouvé, réponds UNIQUEMENT: "Aucun document pertinent trouvé pour répondre à cette question.
4. Utilise UNIQUEMENT les informations des documents fournis
5. Format de réponse:
   - Pour une question sur un produit spécifique: donner une description en paragraphe
   - Pour une comparaison: utiliser des puces (-)
   - Ne jamais commencer la réponse par un tiret

Documents disponibles:
{context}

Question: {question}"""


def load_and_process_data():
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )

    def clean_metadata_value(value):
        """Nettoie les valeurs de métadonnées pour les rendre compatibles avec ChromaDB."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    count = 0
    with open("data/meta.jsonl", "r") as f:
        for line in f:
            count += 1
            data = json.loads(line)

            # Combiner toutes les informations pertinentes
            text = ""
            if data.get("title"):
                text += f"Title: {data['title']}\n"
            if data.get("description"):
                text += f"Description: {' '.join(data['description'])}\n"
            if data.get("features"):
                text += f"Features: {' '.join(data['features'])}\n"
            if data.get("details"):
                text += f"Details: {', '.join([f'{k}: {str(v)}' for k, v in data['details'].items()])}\n"

            # Nettoyer les métadonnées
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
    # Afficher quelques exemples pour vérification
    print("\nExemples de documents chargés:")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


class ProductRAGSystem:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Créer le prompt
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Charger ou créer la base de données
        if not os.path.exists(persist_directory):
            print("Création d'une nouvelle base de données...")
            documents = load_and_process_data()
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            print("Chargement de la base de données existante...")
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

    def setup_llm(self, model_name: str = "gpt-3.5-turbo",
                  temperature: float = 0.0,
                  top_p: float = 1.0):
        """Configure le modèle LLM avec les paramètres spécifiés."""
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p
        )

        # Créer la chaîne RAG
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )

    def query(self, question: str, verbose: bool = False) -> Dict:
        """Exécute une requête et retourne la réponse avec les métriques."""
        with get_openai_callback() as cb:
            # Récupérer les documents pertinents
            relevant_docs = self.db.similarity_search(question, k=4)

            # Préparer le contexte
            context = ""
            for i, doc in enumerate(relevant_docs, 1):
                context += f"\nDocument {i}:\n"
                context += f"Titre: {doc.metadata['title']}\n"
                context += f"Contenu: {doc.page_content}\n"
                context += "-" * 50 + "\n"

            # Exécuter la requête
            result = self.qa_chain.invoke({"query": question})

            if verbose:
                print(f"\nMétriques OpenAI:")
                print(f"Tokens totaux: {cb.total_tokens}")
                print(f"Coût estimé: ${cb.total_cost:.4f}")

            return {
                "question": question,
                "answer": result["result"],
                "source_documents": [doc.metadata['title'] for doc in relevant_docs],
                "metrics": {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }


def test_different_parameters():
    """Teste différentes configurations de paramètres."""
    rag_system = ProductRAGSystem()  # Créé une seule fois

    test_questions = [
        "Tell me about OnePlus 7T cases",
        "What are the best waterproof phone cases available?",
        "I need a phone case with card holder, what do you recommend?"
    ]

    configs = [
        {"temperature": 0.0, "top_p": 1.0},
        {"temperature": 0.3, "top_p": 0.9},
        {"temperature": 0.7, "top_p": 0.8}
    ]

    results = []
    for config in configs:
        print(f"\nTest avec temperature={config['temperature']}, top_p={config['top_p']}")
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


def main():
    # Créer et configurer le système RAG
    rag_system = ProductRAGSystem()
    rag_system.setup_llm()

    # Questions de test
    test_questions = [
        "Tell me about OnePlus 7T cases.",
        "What's the best Apple Watch band?",
        "How can I cook my potatoes ?" # Question non pertinente pour vérifier que le modèle utilise bien les documents
    ]

    # Tester chaque question
    for question in test_questions:
        result = rag_system.query(question)
        print("\n" + "=" * 50)
        print(f"Question: {question}")
        print(f"Réponse: {result['answer']}")

        # N'afficher les documents référés que si la réponse n'est pas "Aucun document pertinent..."
        if not result['answer'].startswith("Aucun document pertinent"):
            print("\nDocuments référés:")
            # Éliminer les doublons et afficher uniquement les titres uniques
            unique_docs = list(set(result['source_documents']))
            for doc in unique_docs:
                print(f"- {doc}")

        print(f"\nMétriques:")
        print(f"Tokens: {result['metrics']['total_tokens']}")
        print(f"Coût: ${result['metrics']['total_cost']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
