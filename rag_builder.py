import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- CONFIG ---
DATA_PATH = "processed_data/train_data.jsonl"
INDEX_PATH = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and effective for semantic search

def build_vector_store():
    print(f"--- 1. Loading Processed Data from {DATA_PATH} ---")

    try:
        df = pd.read_json(DATA_PATH, lines=True, orient='records')
    except ValueError:
        print("Error: Could not read JSONL file. Ensure 'data_loader.py' ran successfully.")
        return

    print("Extracting unique schemas...")

    # Deduplicate! We only want unique tables in our "Library"
    unique_schemas = list(set(df['schema'].dropna().tolist()))
    print(f"Found {len(unique_schemas)} unique table schemas.")

    # --- 2. Initialize Embedding Model (LangChain wrapper around SentenceTransformers) ---
    print(f"--- 2. Loading Embedding Model ({MODEL_NAME}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{MODEL_NAME}")

    # --- 3. Build the LangChain FAISS vector store ---
    # LangChain's FAISS.from_texts() embeds the schemas AND stores the
    # original text alongside the vectors in one object (a docstore),
    # so we no longer need a separate schemas.pkl file.
    print("--- 3. Building FAISS Vector Store (embedding + indexing) ---")
    docs = [Document(page_content=schema) for schema in unique_schemas]
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- 4. Save ---
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)

    print(f"\nSuccess! Vector store saved to: {os.path.abspath(INDEX_PATH)}")
    print("  - index.faiss / index.pkl (LangChain FAISS store: vectors + schema text)")

if __name__ == "__main__":
    build_vector_store()
