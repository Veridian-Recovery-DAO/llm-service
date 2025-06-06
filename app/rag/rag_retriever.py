"""
rag_retriever.py

This module provides a function to get a retriever from the persistent vectorstore.
"""
from .rag_vectorstore import load_vectorstore

def get_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever()
