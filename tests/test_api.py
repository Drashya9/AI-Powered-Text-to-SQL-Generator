import pytest
from fastapi.testclient import TestClient
import sqlglot
from main import app


# This "fixture" tells PyTest to start the app (and load the LangChain chain)
# ONCE for the whole test file, rather than for every single test.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_response_structure(client):
    """Does the API actually return the right JSON format?"""
    response = client.post(
        "/generate",
        json={"natural_language_query": "Show me all the dance programs."}
    )

    assert response.status_code == 200, f"App crashed with error: {response.text}"

    data = response.json()
    assert "sql_query" in data
    assert "retrieved_schema" in data


def test_generated_sql_syntax(client):
    """Is the generated SQL actually valid code?"""
    test_questions = [
        "What are the names of all the dance programs?",
        "How many funding sources do we have?"
    ]

    for question in test_questions:
        response = client.post(
            "/generate",
            json={"natural_language_query": question}
        )
        assert response.status_code == 200, f"App crashed with error: {response.text}"

        generated_sql = response.json()["sql_query"]

        try:
            sqlglot.parse_one(generated_sql)
            is_valid_syntax = True
        except sqlglot.errors.ParseError as e:
            print(f"\n[FAILED] Bad SQL for question: '{question}'")
            print(f"Generated SQL: {generated_sql}")
            print(f"Error: {e}")
            is_valid_syntax = False

        assert is_valid_syntax, "The LLM generated invalid SQL!"


def test_empty_query_does_not_crash_server(client):
    """Edge case: an empty natural language query should not 500."""
    response = client.post("/generate", json={"natural_language_query": ""})
    assert response.status_code == 200
    assert "sql_query" in response.json()


def test_missing_field_returns_422(client):
    """Edge case: a malformed request body should fail FastAPI validation, not crash the app."""
    response = client.post("/generate", json={})
    assert response.status_code == 422
