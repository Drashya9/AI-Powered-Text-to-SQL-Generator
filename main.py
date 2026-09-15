from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# --- Configuration ---
INDEX_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "NumbersStation/nsql-350M"  # Small, fast model designed specifically for SQL
LORA_ADAPTER_PATH = "lora_adapter"  # produced by finetune_lora.py, optional

# --- Initialize Application ---
app = FastAPI(
    title="NLP to SQL Microservice",
    description="Production-ready microservice using a LangChain RAG chain to generate SQL.",
    version="2.0.0"
)

# --- Global Variables ---
chain = None
retriever = None

# --- Data Models ---
class QueryRequest(BaseModel):
    natural_language_query: str

class QueryResponse(BaseModel):
    sql_query: str
    retrieved_schema: str

PROMPT_TEMPLATE = """You are a SQL expert. Write a SQL query to answer the following question based on the provided schema.

Schema:
{schema}

Question: {question}
SQL Query:"""


def format_schema(docs) -> str:
    """The retriever returns a list of Documents; we only asked for k=1."""
    return docs[0].page_content


def extract_sql(full_text: str) -> str:
    """The LLM echoes the prompt back, so pull out just the generated SQL."""
    return full_text.split("SQL Query:")[-1].strip()


@app.on_event("startup")
async def load_models():
    """
    This runs when the server starts. It builds the LangChain RAG chain:
    retriever -> prompt -> LLM -> output parser.
    """
    global chain, retriever

    print("Loading RAG Components...")
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        raise RuntimeError("Vector store not found. Did you run rag_builder.py?")

    # 1. Embeddings + FAISS vector store (built by rag_builder.py)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 2. LLM wrapped as a LangChain HuggingFacePipeline
    print("Loading LLM for SQL Generation (this may take a moment on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    if os.path.exists(LORA_ADAPTER_PATH):
        print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()  # fold adapter into base weights for fast inference
    else:
        print("No LoRA adapter found, using base model as-is.")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 3. Prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # 4. The chain (LCEL): retrieve schema -> fill prompt -> generate -> parse SQL
    chain = (
        {"schema": retriever | RunnableLambda(format_schema), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(extract_sql)
    )

    print("All models loaded successfully. Server is ready!")


@app.post("/generate", response_model=QueryResponse)
async def generate_sql(request: QueryRequest):
    """
    The main API endpoint. Takes natural language and returns SQL.
    """
    try:
        user_query = request.natural_language_query

        # Retrieve schema separately so we can also return it in the response
        retrieved_docs = retriever.invoke(user_query)
        best_schema = retrieved_docs[0].page_content

        sql_output = chain.invoke(user_query)

        return QueryResponse(
            sql_query=sql_output,
            retrieved_schema=best_schema
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
