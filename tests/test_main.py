from fastapi.testclient import TestClient
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Veridian Recovery LLM Service is running."}

def test_ask_llm_valid_query():
    payload = {
        "query": "How can I manage urges during recovery?",
        "userId": "12345"
    }
    response = client.post("/ask-llm", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["query_received"] == payload["query"]

def test_ask_llm_missing_query():
    payload = {
        "userId": "12345"
    }
    response = client.post("/ask-llm", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_ask_llm_invalid_payload():
    payload = {
        "invalid_field": "This is not a valid field"
    }
    response = client.post("/ask-llm", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_ask_llm_server_error():
    payload = {
        "query": "trigger_error",  # Assuming "trigger_error" causes an exception in get_llm_response
        "userId": "12345"
    }
    response = client.post("/ask-llm", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "An error occurred while processing your request with the LLM."