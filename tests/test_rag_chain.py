import pytest
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from main import extract_sql, format_schema, EMBEDDING_MODEL, INDEX_PATH


# --- Unit tests: the pure helper functions used inside the LCEL chain ---

def test_extract_sql_strips_prompt_echo():
    full_text = "You are a SQL expert...\nSQL Query: SELECT * FROM users;"
    assert extract_sql(full_text) == "SELECT * FROM users;"


def test_extract_sql_handles_repeated_marker():
    """The LLM sometimes echoes 'SQL Query:' from the prompt itself before its answer."""
    full_text = "...\nQuestion: what?\nSQL Query:\nSQL Query: SELECT 1;"
    assert extract_sql(full_text) == "SELECT 1;"


def test_format_schema_returns_top_document_content():
    docs = [
        Document(page_content="CREATE TABLE foo (id INT);"),
        Document(page_content="CREATE TABLE bar (id INT);"),
    ]
    assert format_schema(docs) == "CREATE TABLE foo (id INT);"


# --- Integration test: the actual FAISS store built by rag_builder.py ---

@pytest.fixture(scope="module")
def vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def test_vector_store_loads_and_retrieves_a_schema(vectorstore):
    """The vector store rag_builder.py produces should load and return a real schema."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("How many students are enrolled in the dance program?")

    assert len(results) == 1
    assert "CREATE TABLE" in results[0].page_content.upper()


def test_retrieval_is_relevant_to_query_topic(vectorstore):
    """A query about funding should retrieve a schema that plausibly relates to funding."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("What is the total funding amount by source?")

    schema_text = results[0].page_content.lower()
    assert "fund" in schema_text or "grant" in schema_text or "source" in schema_text
