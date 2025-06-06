"""
rag_vectorstore.py

This module provides a utility to load the persistent FAISS vectorstore for use in your RAG pipeline.
"""
import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore.faiss"
METADATA_PATH = Path(__file__).parent / "vectorstore.pkl"

# Use Gemini-compatible embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
)

def load_vectorstore():
    if VECTORSTORE_PATH.exists() and METADATA_PATH.exists():
        return FAISS.load_local(str(VECTORSTORE_PATH), embeddings, str(METADATA_PATH))
    else:
        raise FileNotFoundError("No persistent vectorstore found. Please ingest a PDF first.")
