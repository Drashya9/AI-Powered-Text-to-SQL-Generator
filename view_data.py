import pandas as pd

# The file we created in the last step
FILE_PATH = "processed_data/train_data.jsonl"

try:
    print(f"--- Loading {FILE_PATH} ---")
    
    # lines=True is CRITICAL for .jsonl files
    df = pd.read_json(FILE_PATH, lines=True, orient='records')
    
    print(f"Successfully loaded {len(df)} examples.\n")
    
    # 1. Show a quick summary table
    print("--- First 5 Rows ---")
    print(df.head())
    print(len(df))
    
    # 2. Show one full example clearly
    print("\n--- Full Sample Entry (Row 0) ---")
    sample = df.iloc[1]
    for column, value in sample.items():
        print(f"[{column.upper()}]:\n{value}\n{'-'*40}")

except ValueError:
    print("Error: Could not read file. Did you generate 'train_data.jsonl' correctly?")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Check your folder structure.")