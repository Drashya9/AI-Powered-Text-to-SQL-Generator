import os
import shutil
from datasets import load_dataset
import pandas as pd

# --- CONFIGURATION ---
GRETEL_DATASET = "gretelai/synthetic_text_to_sql"
OUTPUT_DIR = "processed_data"

# Single source of truth for the prompt shape -- used by BOTH finetune_lora.py
# (training) and main.py (serving), so there's no train/serve mismatch. Ends
# in "SQL Query:" so extract_sql() in main.py has a reliable marker to split
# on, and so a SQL-only completion (no explanation) is the natural output.
PROMPT_TEMPLATE = """You are a SQL expert. Write a SQL query to answer the following question based on the provided schema. Use only columns and tables explicitly mentioned in the provided schema.

Schema:
{schema}

Question: {question}
SQL Query:"""


def format_gretel_example(example):
    # Store the raw fields only. sql/explanation are stored once here, not
    # also baked into a second "formatted_prompt" string, so nothing is
    # duplicated on disk. (explanation is kept for reference/debugging even
    # though training no longer targets it -- see finetune_lora.py.)
    return {
        "instruction": example['sql_prompt'],
        "schema": example['sql_context'],
        "sql": example['sql'],
        "explanation": example['sql_explanation'],
    }


def build_prompt(question: str, schema: str) -> str:
    """Fills the shared PROMPT_TEMPLATE. `schema` should already be cleaned
    (see rag_builder.clean_schema) before being passed in here."""
    return PROMPT_TEMPLATE.format(schema=schema, question=question)

def process_data():
    # Clean start: remove the folder if it exists to prevent appending errors.
    # This only runs when process_data() is actually called (i.e. `python
    # data_loader.py`), not as a side effect of importing build_prompt().
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"--- 1. Downloading Data from {GRETEL_DATASET} ---")
    dataset = load_dataset(GRETEL_DATASET, split="train")
    
    # Debug: Print raw entry to ensure we aren't crazy
    print("\n--- DEBUG: RAW DATA ENTRY (Row 0) ---")
    print(f"Raw Prompt: {dataset[0]['sql_prompt'][:50]}...")
    print(f"Raw SQL:    {dataset[0]['sql'][:50]}...")
    print("-" * 30)

    print("--- 2. Formatting Data ---")
    # Select only columns we need to avoid clutter
    dataset = dataset.select_columns(['sql_prompt', 'sql_context', 'sql', 'sql_explanation'])
    
    processed_data = []
    for item in dataset:
        processed_data.append(format_gretel_example(item))
        
    df = pd.DataFrame(processed_data)
    
    output_path = os.path.join(OUTPUT_DIR, "train_data.jsonl")
    df.to_json(output_path, orient='records', lines=True)
    
    print(f"Successfully saved {len(df)} rows to: {output_path}")

if __name__ == "__main__":
    process_data()