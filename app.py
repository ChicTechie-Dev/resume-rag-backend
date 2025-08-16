import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --------------------------
# Load environment
# --------------------------
load_dotenv()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama3-8b-8192")  # Groq LLM
INDEX_PATH = "faiss_index"  # Prebuilt FAISS index (built with build_index.py)
TOP_K = 12
MAX_NEW_TOKENS = 180

# --------------------------
# Load FAISS index
# --------------------------
print("Loading FAISS index...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded")

# --------------------------
# Initialize Groq Client
# --------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --------------------------
# Helpers
# --------------------------
def retrieve_context(query: str) -> List[str]:
    """Retrieve top-K relevant chunks."""
    results = vectordb.similarity_search(query, k=TOP_K)
    return [doc.page_content for doc in results]

def make_prompt(context_blocks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return (
        "You are a highly accurate Resume Q&A assistant.\n"
        "Answer ONLY from the resume context below.\n"
        "If the answer is not present, say:\n"
        "\"Sorry, I don’t have that information in my resume.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def generate_with_groq(prompt: str) -> str:
    """Send prompt to Groq API and return response."""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a resume assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.4
    )
    return completion.choices[0].message.content.strip()

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="Resume Q&A Bot (Groq)")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        return {"answer": "Please ask a question about my resume."}

    # Step 1: Retrieve context
    context = retrieve_context(q)

    # Step 2: Build prompt
    prompt = make_prompt(context, q)

    # Step 3: Generate with Groq
    answer = generate_with_groq(prompt)

    return {"answer": answer}
