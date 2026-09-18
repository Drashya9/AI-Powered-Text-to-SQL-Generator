import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from evaluate_lora import DATA_PATH, build_held_out_test_set
from evaluate_execution import run_query, results_match
from rag_builder import clean_schema
from main import INDEX_PATH, EMBEDDING_MODEL

load_dotenv()

# --- CONFIG ---
TOP_K = 10
GEMINI_MODEL = "gemini-3.6-flash"
SECONDS_BETWEEN_CALLS = 8  # conservative, stays well within the free tier's rate limit

PROMPT_TEMPLATE = """You are a SQL expert. You are given a natural language question and a list of {k} candidate database schemas. At most ONE of these schemas is actually relevant to answering the question -- the rest are distractors from an imperfect retrieval step.

Question: {question}

Candidate schemas:
{candidates}

Instructions:
- Identify which schema number is relevant to this question.
- If none of the schemas are relevant, respond with exactly: SCHEMA: NONE
- Otherwise, respond in exactly this format (no extra commentary):
SCHEMA: <number>
SQL: <the SQL query, nothing else on that line>
"""


def build_prompt(question: str, candidates: list) -> str:
    candidate_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    return PROMPT_TEMPLATE.format(k=len(candidates), question=question, candidates=candidate_text)


def parse_response(text: str):
    """Returns (schema_number_or_None, sql_or_None)."""
    schema_match = re.search(r"SCHEMA:\s*(\d+|NONE)", text, re.IGNORECASE)
    if not schema_match or schema_match.group(1).upper() == "NONE":
        return None, None
    schema_num = int(schema_match.group(1))
    sql_match = re.search(r"SQL:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    sql = sql_match.group(1).strip().split("\n")[0].strip() if sql_match else None
    return schema_num, sql


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found -- check .env")
    client = genai.Client(api_key=api_key)

    print("--- Loading dataset and held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    n = len(test_rows)
    print(f"Test set size: {n} rows, top-k={TOP_K}")

    print("--- Loading fine-tuned mpnet retriever ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    correct = 0
    picked_right_schema = 0
    said_none_correctly = 0
    errors = 0

    for i, row in enumerate(test_rows.itertuples(), 1):
        true_schema_cleaned = clean_schema(row.schema).strip()
        gt_rows, gt_err = run_query(row.schema, row.sql)

        docs = vectorstore.similarity_search(row.instruction, k=TOP_K)
        candidates = [d.page_content for d in docs]
        correct_idx = next(
            (idx for idx, c in enumerate(candidates) if c.strip() == true_schema_cleaned), None
        )

        prompt = build_prompt(row.instruction, candidates)
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            schema_num, sql = parse_response(response.text or "")
        except Exception as e:
            print(f"  [{i}/{n}] API error: {type(e).__name__}: {e}")
            errors += 1
            time.sleep(SECONDS_BETWEEN_CALLS)
            continue

        if schema_num is None:
            if correct_idx is None:
                said_none_correctly += 1
                correct += 1  # correctly identified no relevant schema exists
        else:
            chosen_idx = schema_num - 1
            if correct_idx is not None and chosen_idx == correct_idx:
                picked_right_schema += 1
                if sql and gt_err is None:
                    gen_rows, gen_err = run_query(row.schema, sql)
                    if gen_err is None and results_match(gen_rows, gt_rows):
                        correct += 1

        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} processed... (correct so far: {correct})")

        time.sleep(SECONDS_BETWEEN_CALLS)

    print("\n" + "=" * 60)
    print(f"GEMINI ({GEMINI_MODEL}) TOP-{TOP_K} EVALUATION (n={n})")
    print("=" * 60)
    print(f"Picked the correct schema:        {picked_right_schema}/{n} = {100*picked_right_schema/n:.1f}%")
    print(f"Correctly said NONE relevant:      {said_none_correctly}")
    print(f"Correct (execution matches truth): {correct}/{n} = {100*correct/n:.1f}%")
    print(f"API errors encountered:            {errors}")


if __name__ == "__main__":
    main()
