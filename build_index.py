import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "resume.pdf"
INDEX_PATH = "faiss_index"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

print("Chunking...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(chunks)} chunks from {PDF_PATH}")

print("Building embeddings...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Creating FAISS index...")
vectordb = FAISS.from_documents(chunks, emb)

print("Saving index...")
vectordb.save_local(INDEX_PATH)
print("✅ Index saved to", INDEX_PATH)
