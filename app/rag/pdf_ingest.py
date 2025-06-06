"""
pdf_ingest.py

This script provides utilities to:
- Extract text from PDF files
- Split text into chunks
- Add chunks to a persistent FAISS vectorstore
- Ensure new PDFs are added to the same vectorstore

Usage:
    python pdf_ingest.py <path_to_pdf>

Requirements:
    pip install langchain-community langchain-text-splitters faiss-cpu pypdf
"""
import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore.faiss"
METADATA_PATH = Path(__file__).parent / "vectorstore.pkl"

# Embeddings model (replace with your preferred one)
embeddings = OpenAIEmbeddings()


def load_or_create_vectorstore():
    if VECTORSTORE_PATH.exists() and METADATA_PATH.exists():
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), embeddings, str(METADATA_PATH))
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS(embeddings.embed_query, embeddings)
    return vectorstore


def ingest_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {pdf_path}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def add_chunks_to_vectorstore(chunks, vectorstore):
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to vectorstore.")


def persist_vectorstore(vectorstore):
    vectorstore.save_local(str(VECTORSTORE_PATH), str(METADATA_PATH))
    print(f"Vectorstore persisted to {VECTORSTORE_PATH} and {METADATA_PATH}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingest.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    vectorstore = load_or_create_vectorstore()
    chunks = ingest_pdf(pdf_path)
    add_chunks_to_vectorstore(chunks, vectorstore)
    persist_vectorstore(vectorstore)
    print("Done.")

if __name__ == "__main__":
    main()
