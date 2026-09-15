import os
import shutil
from datasets import load_dataset
import pandas as pd

# --- CONFIGURATION ---
GRETEL_DATASET = "gretelai/synthetic_text_to_sql"
OUTPUT_DIR = "processed_data"

SYSTEM_PROMPT = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output a brief explanation of your logic, followed by the valid SQL query."""

def format_gretel_example(example):
    # Store the raw fields only. SYSTEM_PROMPT is a constant, so it's applied
    # by build_prompt() at use-time instead of being repeated in every row
    # (that alone would add ~26MB of pure duplication across 100k rows).
    # sql/explanation are stored once here, not also baked into a second
    # "formatted_prompt" string, so nothing is duplicated on disk.
    return {
        "instruction": example['sql_prompt'],
        "schema": example['sql_context'],
        "sql": example['sql'],
        "explanation": example['sql_explanation'],
    }


def build_prompt(instruction: str, schema: str) -> str:
    """Reconstructs the full training-style prompt from the raw fields on demand."""
    return f"""{SYSTEM_PROMPT}

### Instruction:
{instruction}

### Database Schema:
{schema}

### Response:
"""

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