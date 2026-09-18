import re
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

from data_loader import PROMPT_TEMPLATE

# --- Configuration ---
# Retrieval was measured at 27% top-1 accuracy with base all-MiniLM-L6-v2.
# Fine-tuning all-mpnet-base-v2 on our own (question, schema) pairs raised
# that to 42% (see finetune_embeddings.py / vector_store_finetuned_mpnet/).
# A cross-encoder reranker was also tested on top and added 0pp here, so it's
# deliberately not wired in -- it would only add latency for no measured gain.
INDEX_PATH = "vector_store_finetuned_mpnet"
EMBEDDING_MODEL = "finetuned_embedding_model_mpnet"
# Generation model: swapped from NumbersStation/nsql-350M after evaluate_generation.py
# showed Qwen2.5-Coder-1.5B-Instruct zero-shot beating nsql-350M+LoRA on every
# metric on the same held-out set (syntax valid 79%->100%, schema-grounded
# 23%->95%, exact match 0%->23%). Slower per-query (larger model, CPU-only),
# but the accuracy gap was too large to leave nsql in production.
LLM_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_ADAPTER_PATH = None  # nsql's LoRA adapter doesn't transfer to Qwen's architecture -- see old_models/lora_adapter/

CHAT_SYSTEM_PROMPT = (
    "You are a SQL expert. Write a SQL query to answer the question based on the "
    "provided schema. Use only columns and tables explicitly mentioned in the "
    "provided schema. Respond with ONLY the SQL query, no explanation, no markdown "
    "code fences."
)
CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

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

def format_schema(docs) -> str:
    """The retriever returns a list of Documents; we only asked for k=1."""
    return docs[0].page_content


def extract_sql(full_text: str) -> str:
    """Pulls just the generated SQL out of a raw model completion. Handles
    both formats seen in this project: chat models (Qwen) that may wrap SQL
    in markdown fences, and completion-style models (nsql) that echo the
    full prompt ending in "SQL Query:"."""
    fence_match = CODE_FENCE_RE.search(full_text)
    if fence_match:
        return fence_match.group(1).strip()
    if "SQL Query:" in full_text:
        return full_text.split("SQL Query:")[-1].strip()
    return full_text.strip()


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

    if LORA_ADAPTER_PATH and os.path.exists(LORA_ADAPTER_PATH):
        print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()  # fold adapter into base weights for fast inference
    else:
        print("No LoRA adapter configured, using base model as-is.")

    # Chat-template models (e.g. Qwen) need chat-formatted input to perform as
    # tested; completion-style models (e.g. nsql) use the shared PROMPT_TEMPLATE
    # instead, matching how they were trained. See evaluate_generation.py, which
    # this mirrors exactly so production behavior matches what was measured.
    has_chat_template = tokenizer.chat_template is not None
    print(f"Chat template detected: {has_chat_template} (using {'chat' if has_chat_template else 'completion'} prompting)")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120 if has_chat_template else 100,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=not has_chat_template,
        **({} if has_chat_template else {"repetition_penalty": 1.3, "no_repeat_ngram_size": 3}),
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 3. Prompt
    if has_chat_template:
        def build_chat_prompt(inputs: dict) -> str:
            messages = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{inputs['schema']}\n\nQuestion: {inputs['question']}"},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = RunnableLambda(build_chat_prompt)
    else:
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
