from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
import os

# -----------------------------
# 🔐 Load Environment Variables
# -----------------------------
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "currently_knowledgebase")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# -----------------------------
# 🌐 Flask App Setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# 🧠 LangChain QA Setup
# -----------------------------
def initialize_components():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    doc_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    return doc_store

def create_qa_chain(doc_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    prompt = PromptTemplate(
        template="""Answer the following question based only on the given context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# -----------------------------
# ⚙️ Init LangChain System
# -----------------------------
doc_store = initialize_components()
qa_chain = create_qa_chain(doc_store)

# -----------------------------
# 📬 Webhook Endpoint for WhatsApp
# -----------------------------
@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    user_message = request.form.get("Body")
    user_number = request.form.get("From")

    print(f"📨 Message from {user_number}: {user_message}")

    try:
        result = qa_chain.invoke({"query": user_message})
        answer = result["result"]
    except Exception as e:
        answer = "Sorry, something went wrong processing your request."

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)

    return str(resp)

# -----------------------------
# 🏠 Optional Homepage
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "✅ WhatsApp bot is live! Use /webhook to receive messages."

# -----------------------------
# 🚀 Render Will Use Gunicorn (via Procfile)
# -----------------------------
if __name__ == "__main__":
    app.run(port=5000)
