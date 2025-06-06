from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

from .llm_logic import (
    get_llm_response,
    get_rag_response,
    get_chat_history_qa,
    get_tool_calling_response,
    get_streaming_response,
    get_multimodal_response,
)

load_dotenv()

app = FastAPI(
    title="Veridian Recovery LLM Service",
    description="Provides access to an LLM trained in recovery literature.",
    version="0.1.0",
)


class LLMQuery(BaseModel):
    query: str
    userId: str | None = None  # Optional: for context or logging
    history: list | None = None  # Add chat history for conversational endpoints
    messages: list | None = None  # For agentic/conversational RAG (optional, for future use)
    image_url: str | None = None  # For multimodal endpoint


class LLMResponse(BaseModel):
    answer: str
    query_received: str


@app.post("/ask-llm", response_model=LLMResponse)
async def ask_llm(request_body: LLMQuery):
    """
    Accepts a user query and returns a response from the LLM.
    """
    try:
        response_text = get_llm_response(request_body.query, request_body.userId)
        return LLMResponse(answer=response_text, query_received=request_body.query)
    except Exception as e:
        print(f"Error in /ask-llm endpoint: {e}")  # Log the actual error server-side
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request with the LLM.",
        )


@app.post("/rag")
def rag_endpoint(request: LLMQuery):
    return get_rag_response(request.query)


@app.post("/chat-history-qa")
def chat_history_qa_endpoint(request: LLMQuery):
    return get_chat_history_qa(request.query, request.history)


@app.post("/tool-calling")
def tool_calling_endpoint(request: LLMQuery):
    return get_tool_calling_response(request.query)


@app.post("/streaming")
def streaming_endpoint(request: LLMQuery):
    return get_streaming_response(request.query)


@app.post("/multimodal")
def multimodal_endpoint(request: LLMQuery):
    return get_multimodal_response(request.query, request.image_url)


@app.get("/")
async def root():
    return {"message": "Veridian Recovery LLM Service is running."}


# To run: uvicorn app.main:app --reload --port 8000
# (from within the llm-service directory)
