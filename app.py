import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
# --------------------------
# Settings
# --------------------------
MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama3-8b-8192")  # Groq LLM
PDF_PATH = "resume.pdf"  # PDF should be in the same folder
TOP_K = 12
MAX_NEW_TOKENS = 180

# --------------------------
# Load & index resume.pdf
# --------------------------
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

print("Chunking...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(chunks)} chunks from {PDF_PATH}")

print("Building embeddings (MiniLM-L6-v2)...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Creating FAISS index...")
vectordb = FAISS.from_documents(chunks, emb)

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
        "Your task is to answer the user's question using ONLY the provided resume context.\n"
        "Do not use any outside knowledge, do not guess, and do not make assumptions.\n"
        "If the exact information is not found in the context, respond with:\n"
        "\"Sorry, I don’t have that information in my resume.\"\n\n"
        "Guidelines:\n"
        "- Use only facts from the provided context.\n"
        "- Give concise, professional answers.\n"
        "- If dates, roles, or company names are mentioned, reproduce them exactly as in the resume.\n"
        "- Never invent details that are not present in the context.\n"
        "- Maintain proper formatting and clarity.\n\n"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "chunks": len(chunks)}

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
