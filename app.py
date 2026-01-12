from flask import Flask, request, jsonify
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
import os

# ---------------- CONFIG ----------------
QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://71a9a1e0-19f8-4e29-bc0e-2c45000de462.us-west-1-0.aws.cloud.qdrant.io:6333"
)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "YOUR_QDRANT_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "currently_knowledgebase")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- LANGCHAIN INIT ----------------
def initialize_components():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    return vectorstore


def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )

    prompt_template = """
You are a world-class, extremely polite, and helpful AI customer support assistant.
Answer ONLY using the provided context.
If the answer is not present, politely say you don't know.

Context:
{context}

User Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )


vectorstore = initialize_components()
qa_chain = create_qa_chain(vectorstore)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Customer Support RAG API running"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"answer": result["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render compatible
    app.run(host="0.0.0.0", port=port)
