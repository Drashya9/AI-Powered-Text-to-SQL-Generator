import pytest
from fastapi.testclient import TestClient
import sqlglot
from main import app

# This "fixture" tells PyTest to start the app (and load models) 
# ONCE for the whole test file, rather than for every single test.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_api_response_structure(client):
    """Test 1: Does the API actually return the right JSON format?"""
    response = client.post(
        "/generate", 
        json={"natural_language_query": "Show me all the dance programs."}
    )
    
    # If it fails now, we print the actual error message to help us debug!
    assert response.status_code == 200, f"App crashed with error: {response.text}"
    
    data = response.json()
    assert "sql_query" in data
    assert "retrieved_schema" in data

def test_generated_sql_syntax(client):
    """Test 2: Is the generated SQL actually valid code?"""
    # Kept to 2 questions to make the test run a bit faster
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
            
        assert is_valid_syntax == True, "The LLM generated invalid SQL!"