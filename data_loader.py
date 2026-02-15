import os
import shutil
from datasets import load_dataset
import pandas as pd

# --- CONFIGURATION ---
GRETEL_DATASET = "gretelai/synthetic_text_to_sql"
OUTPUT_DIR = "processed_data"

# Clean start: Remove the folder if it exists to prevent appending errors
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

SYSTEM_PROMPT = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output a brief explanation of your logic, followed by the valid SQL query."""

def format_gretel_example(example):
    instruction = example['sql_prompt']
    schema = example['sql_context']
    output_sql = example['sql']
    explanation = example['sql_explanation']
    
    # Create the response format: Explanation + SQL
    formatted_response = f"-- {explanation}\n{output_sql}"

    # Create the final prompt
    formatted_text = f"""{SYSTEM_PROMPT}

### Instruction:
{instruction}

### Database Schema:
{schema}

### Response:
{formatted_response}
"""
    return {
        "formatted_prompt": formatted_text,
        "ground_truth_sql": output_sql,
        "explanation": explanation
    }

def process_data():
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