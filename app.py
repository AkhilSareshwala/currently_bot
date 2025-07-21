from flask import Flask, request, jsonify, render_template_string
from twilio.twiml.messaging_response import MessagingResponse
import psycopg2
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
import os

# --------------- Configuration (Read from Environment for Render deployment) -----------------
QDRANT_URL = os.environ.get("QDRANT_URL", "https://71a9a1e0-19f8-4e29-bc0e-2c45000de462.us-west-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1Ni...")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDSu...")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "currently_knowledgebase")

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "Bot")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "Irshad@7078")

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')

# --------------- Flask App -----------------
app = Flask(__name__)

# --------------- Database Helpers -----------------
def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def get_answer_from_db(question):
    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT answer FROM chatbot_qa WHERE question = %s", (question,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def insert_qa_to_db(question, answer):
    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chatbot_qa (question, answer) VALUES (%s, %s) ON CONFLICT (question) DO NOTHING",
            (question, answer)
        )
        conn.commit()
    conn.close()

# --------------- Gemini & LangChain QA Setup -----------------
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
        temperature=0.2
    )
    prompt_template = """
You are a world-class, extremely polite, and helpful AI customer support assistant for our company.
Please provide genuinely useful, clear, and accurate answers based only on the company knowledge base (context below).
You always greet the customer, thank them for their inquiry, and offer further assistance politely.
If information is not found, apologize and suggest human support if possible.

Company Knowledge Base:
{context}

Customer's Question:
{question}

Your answer (be concise, empathetic, and supportive):
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# --------------- LangChain System Init ---------------
doc_store = initialize_components()
qa_chain = create_qa_chain(doc_store)

# --------------- WhatsApp Webhook (Twilio) ---------------
@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    user_message = request.form.get("Body")
    user_number = request.form.get("From")

    print(f"📨 Message from {user_number}: {user_message}")

    answer = get_answer_from_db(user_message)
    if answer:
        response_text = answer
    else:
        try:
            result = qa_chain.invoke({"query": user_message})
            response_text = result["result"]
            insert_qa_to_db(user_message, response_text)
        except Exception as e:
            print("LLM/Webhook error:", e)
            response_text = ("Sorry, something went wrong while processing your request. "
                             "Please try again, or contact customer support.")

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response_text)
    return str(resp)

# --------------- Simple Web Frontend (for testing in browser/Render health check) ---------------
HTML = """
<!doctype html>
<title>Customer Support QA Bot</title>
<h2>Ask our AI customer care assistant</h2>
<form id="qa-form">
  <input type="text" id="question" name="question" style="width:400px;" required>
  <button type="submit">Ask</button>
</form>
<div id="answer" style="margin-top: 20px; font-weight: bold;"></div>
<script>
  document.getElementById('qa-form').onsubmit = async function(e) {
    e.preventDefault();
    document.getElementById('answer').innerText = 'Loading...';
    let q = document.getElementById('question').value;
    let resp = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    let data = await resp.json();
    if(data.answer) {
      document.getElementById('answer').innerText = data.answer + (data.source==="database" ? " (from database)" : " (from LLM)");
    } else {
      document.getElementById('answer').innerText = "Error: " + (data.error || "Unknown error");
    }
  }
</script>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    in_json = request.get_json()
    user_question = in_json.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Missing question"}), 400
    db_answer = get_answer_from_db(user_question)
    if db_answer:
        return jsonify({'answer': db_answer, "source": "database"})
    try:
        result = qa_chain.invoke({"query": user_question})
        answer = result["result"]
        insert_qa_to_db(user_question, answer)
        return jsonify({'answer': answer, "source": "llm"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------- Run App (for Render) ---------------
if __name__ == "__main__":
    # On Render, use PORT env-var (Render auto-assigns), else default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
