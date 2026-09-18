import pandas as pd
import os
import sqlglot
from sqlglot import exp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- CONFIG ---
DATA_PATH = "processed_data/train_data.jsonl"
INDEX_PATH = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and effective for semantic search


def clean_schema(schema_text: str) -> str:
    """Keeps CREATE TABLE statements plus exactly ONE sample row per INSERT
    (not zero, not all). Measured empirically: stripping ALL sample data
    actually hurt retrieval accuracy (25% -> 19%) -- all-MiniLM-L6-v2 is a
    general sentence-embedding model that needs some real, human-readable
    words ('Tofu Stir Fry', 'Vegan') to semantically anchor a bare, formal
    CREATE TABLE statement. What's redundant is repeating that flavor across
    3-4 near-duplicate rows, not having a data example at all."""
    try:
        statements = sqlglot.parse(schema_text)
    except Exception:
        return schema_text  # fall back to raw text if it doesn't parse

    kept = []
    for stmt in statements:
        if isinstance(stmt, exp.Create):
            kept.append(stmt)
        elif isinstance(stmt, exp.Insert):
            values = stmt.find(exp.Values)
            if values is not None and values.expressions:
                values.set("expressions", [values.expressions[0]])
            kept.append(stmt)

    if not kept:
        return schema_text  # fall back if nothing came back as CREATE/INSERT

    return "; ".join(stmt.sql() for stmt in kept) + ";"


def build_vector_store(model_name: str = None, index_path: str = None):
    """model_name/index_path let this build against a different embedding
    model (e.g. a fine-tuned one, given as a local directory path) without
    touching the default vector_store/ -- each variant gets its own folder,
    nothing gets overwritten."""
    model_name = model_name or f"sentence-transformers/{MODEL_NAME}"
    index_path = index_path or INDEX_PATH

    print(f"--- 1. Loading Processed Data from {DATA_PATH} ---")

    try:
        df = pd.read_json(DATA_PATH, lines=True, orient='records')
    except ValueError:
        print("Error: Could not read JSONL file. Ensure 'data_loader.py' ran successfully.")
        return

    print("Extracting and cleaning unique schemas...")

    # Clean first, then deduplicate -- schemas that only differed by their
    # INSERT sample data now correctly collapse into a single entry too.
    cleaned_schemas = (clean_schema(s) for s in df['schema'].dropna())
    unique_schemas = list(set(cleaned_schemas))
    print(f"Found {len(unique_schemas)} unique table schemas.")

    # --- 2. Initialize Embedding Model (LangChain wrapper around SentenceTransformers) ---
    print(f"--- 2. Loading Embedding Model ({model_name}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # --- 3. Build the LangChain FAISS vector store ---
    # LangChain's FAISS.from_texts() embeds the schemas AND stores the
    # original text alongside the vectors in one object (a docstore),
    # so we no longer need a separate schemas.pkl file.
    print("--- 3. Building FAISS Vector Store (embedding + indexing) ---")
    docs = [Document(page_content=schema) for schema in unique_schemas]
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- 4. Save ---
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

    print(f"\nSuccess! Vector store saved to: {os.path.abspath(index_path)}")
    print("  - index.faiss / index.pkl (LangChain FAISS store: vectors + schema text)")

if __name__ == "__main__":
    build_vector_store()
