from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# --- Configuration ---
INDEX_PATH = "vector_store/schema.index"
SCHEMAS_PATH = "vector_store/schemas.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "NumbersStation/nsql-350M"  # Small, fast model designed specifically for SQL

# --- Initialize Application ---
app = FastAPI(
    title="NLP to SQL Microservice",
    description="Production-ready microservice using RAG and LLMs to generate SQL.",
    version="1.0.0"
)

# --- Global Variables for Models ---
retriever = None
faiss_index = None
schema_library = []
tokenizer = None
model = None

# --- Data Models ---
class QueryRequest(BaseModel):
    natural_language_query: str

class QueryResponse(BaseModel):
    sql_query: str
    retrieved_schema: str

@app.on_event("startup")
async def load_models():
    """
    This runs when the server starts. It loads our vector database and AI models into memory.
    """
    global retriever, faiss_index, schema_library, tokenizer, model
    
    print("Loading RAG Components...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(SCHEMAS_PATH):
        raise RuntimeError("Vector store not found. Did you run rag_builder.py?")
        
    # 1. Load FAISS and Schemas
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(SCHEMAS_PATH, "rb") as f:
        schema_library = pickle.load(f)
    
    # 2. Load Embedding Model
    retriever = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Load LLM (Text-to-SQL Model)
    print("Loading LLM for SQL Generation (this may take a moment on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    print("All models loaded successfully. Server is ready!")

def retrieve_schema(query: str) -> str:
    """Uses FAISS to find the most relevant table schema for the user's query."""
    # Convert query to vector
    query_vector = retriever.encode([query]).astype('float32')
    
    # Search FAISS (k=1 means return the top 1 closest match)
    distances, indices = faiss_index.search(query_vector, k=1)
    
    best_match_index = indices[0][0]
    return schema_library[best_match_index]

@app.post("/generate", response_model=QueryResponse)
async def generate_sql(request: QueryRequest):
    """
    The main API endpoint. Takes natural language and returns SQL.
    """
    try:
        user_query = request.natural_language_query
        
        # 1. RAG Step: Retrieve the relevant schema
        best_schema = retrieve_schema(user_query)
        
        # 2. Construct the Prompt (Instructing the LLM)
        prompt = f"""You are a SQL expert. Write a SQL query to answer the following question based on the provided schema.
        
Schema:
{best_schema}

Question: {user_query}
SQL Query:"""

        # 3. Generate SQL using the LLM
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # We tell the model to generate max 100 new tokens (enough for a SQL query)
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the output back to text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the SQL part (removing the prompt)
        sql_output = generated_text.split("SQL Query:")[1].strip()
        
        return QueryResponse(
            sql_query=sql_output,
            retrieved_schema=best_schema
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))