import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

# --- CONFIG ---
DATA_PATH = "processed_data/train_data.jsonl"
INDEX_PATH = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2" # Fast and effective for semantic search

def build_vector_store():
    print(f"--- 1. Loading Processed Data from {DATA_PATH} ---")
    
    try:
        df = pd.read_json(DATA_PATH, lines=True, orient='records')
    except ValueError:
        print("Error: Could not read JSONL file. Ensure 'data_loader.py' ran successfully.")
        return

    print("Extracting unique schemas from prompts...")
    
    # We need to pull the Schema back out of the formatted prompt.
    # The format is rigid, so we can split by our known headers.
    schemas = []
    
    for prompt in df['formatted_prompt']:
        try:
            # Extract text between "### Database Schema:" and "### Response:"
            part1 = prompt.split("### Database Schema:\n")[1]
            schema_text = part1.split("\n\n### Response:")[0].strip()
            
            if schema_text:
                schemas.append(schema_text)
        except IndexError:
            # Skip rows if formatting is unexpected (shouldn't happen with clean data)
            continue
            
    # Deduplicate! We only want unique tables in our "Library"
    unique_schemas = list(set(schemas))
    print(f"Found {len(unique_schemas)} unique table schemas.")
    
    # --- 2. Initialize Embedding Model ---
    print(f"--- 2. Loading Embedding Model ({MODEL_NAME}) ---")
    # This downloads a small model that converts text to numbers
    encoder = SentenceTransformer(MODEL_NAME)
    
    # --- 3. Create Embeddings ---
    print("Creating vectors (This turns text into numbers)...")
    vectors = encoder.encode(unique_schemas, show_progress_bar=True)
    
    # FAISS expects float32 format
    vectors = np.array(vectors).astype('float32')
    
    # --- 4. Build FAISS Index ---
    print("--- 3. Building FAISS Index ---")
    # Dimension of the vectors (384 for MiniLM)
    d = vectors.shape[1] 
    
    # Create a flat (exact) index. Good for datasets < 1M items.
    index = faiss.IndexFlatL2(d) 
    index.add(vectors)
    
    # --- 5. Save Everything ---
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    # Save the FAISS index (the math part)
    faiss.write_index(index, os.path.join(INDEX_PATH, "schema.index"))
    
    # Save the actual text schemas (the readable part)
    # We need this so when FAISS finds "Vector #42", we know what text that corresponds to.
    with open(os.path.join(INDEX_PATH, "schemas.pkl"), "wb") as f:
        pickle.dump(unique_schemas, f)
        
    print(f"\nSuccess! Vector store saved to: {os.path.abspath(INDEX_PATH)}")
    print("  - schema.index (The search engine)")
    print("  - schemas.pkl  (The database definitions)")

if __name__ == "__main__":
    build_vector_store()