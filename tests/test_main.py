from fastapi.testclient import TestClient
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Veridian Recovery LLM Service is running."}


def test_rag_endpoint():
    payload = {
        "query": "What are urge management strategies?"
    }
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), str) or "answer" in response.json() or response.json()  # Accepts string or dict

def test_rag_valid_query():
    payload = {"query": "What are urge management strategies?"}
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_rag_missing_query():
    payload = {}
    response = client.post("/rag", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_rag_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/rag", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_chat_history_qa_endpoint():
    payload = {
        "query": "What is recovery?",
        "history": [
            {"user": "What is addiction?", "assistant": "Addiction is a chronic condition..."},
            {"user": "How does recovery work?", "assistant": "Recovery is a process..."}
        ]
    }
    response = client.post("/chat-history-qa", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), str) or "answer" in response.json() or response.json()  # Accepts string or dict

def test_chat_history_qa_valid_query():
    payload = {
        "query": "What is recovery?",
        "history": [
            {"user": "What is addiction?", "assistant": "Addiction is a chronic condition..."},
            {"user": "How does recovery work?", "assistant": "Recovery is a process..."}
        ]
    }
    response = client.post("/chat-history-qa", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_chat_history_qa_missing_query():
    payload = {"history": []}
    response = client.post("/chat-history-qa", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_chat_history_qa_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/chat-history-qa", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_tool_calling_endpoint():
    payload = {
        "query": "Use the AINetwork tool to fetch info."
    }
    response = client.post("/tool-calling", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a result

def test_tool_calling_valid_query():
    payload = {"query": "Use the AINetwork tool to fetch info."}
    response = client.post("/tool-calling", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_tool_calling_missing_query():
    payload = {}
    response = client.post("/tool-calling", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_tool_calling_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/tool-calling", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_agentic_rag_endpoint():
    payload = {
        "messages": [
            {"role": "user", "content": "What is Task Decomposition?"}
        ]
    }
    response = client.post("/agentic-rag", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), str) or "answer" in response.json() or response.json()  # Accepts string or dict

def test_agentic_rag_valid_query():
    payload = {
        "messages": [
            {"role": "user", "content": "What is Task Decomposition?"}
        ]
    }
    response = client.post("/agentic-rag", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_agentic_rag_missing_messages():
    payload = {}
    response = client.post("/agentic-rag", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_agentic_rag_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/agentic-rag", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_streaming_endpoint():
    payload = {"query": "Stream a response about recovery."}
    response = client.post("/streaming", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), str) or "answer" in response.json() or response.json()  # Accepts string or dict

def test_streaming_valid_query():
    payload = {"query": "Stream a response about recovery."}
    response = client.post("/streaming", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_streaming_missing_query():
    payload = {}
    response = client.post("/streaming", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_streaming_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/streaming", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_multimodal_endpoint():
    payload = {"query": "Describe this image.", "image_url": "http://example.com/image.png"}
    response = client.post("/multimodal", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), str) or "answer" in response.json() or response.json()  # Accepts string or dict

def test_multimodal_valid_query():
    payload = {"query": "Describe this image.", "image_url": "http://example.com/image.png"}
    response = client.post("/multimodal", json=payload)
    assert response.status_code == 200
    assert response.json()  # Should return a string or dict

def test_multimodal_missing_query():
    payload = {"image_url": "http://example.com/image.png"}
    response = client.post("/multimodal", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_multimodal_invalid_payload():
    payload = {"invalid_field": "foo"}
    response = client.post("/multimodal", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()