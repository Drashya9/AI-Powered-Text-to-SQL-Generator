import requests
from datasets import load_dataset
import sqlglot
from tqdm import tqdm
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/generate"
DATASET_NAME = "gretelai/synthetic_text_to_sql"
# Testing on 100 unseen samples so it doesn't take hours. 
NUM_TEST_SAMPLES = 100 

def evaluate_api():
    print(f"--- 1. Downloading Test Dataset: {DATASET_NAME} ---")
    # Load specifically the unseen "test" split
    test_dataset = load_dataset(DATASET_NAME, split="test")
    
    # Shuffle and pick a subset to test
    test_dataset = test_dataset.shuffle(seed=42).select(range(NUM_TEST_SAMPLES))
    
    print(f"Loaded {len(test_dataset)} test samples.\n")
    
    # --- METRICS TRACKING ---
    successful_api_calls = 0
    valid_syntax_count = 0
    correct_table_retrieved = 0
    
    print("--- 2. Starting Evaluation (Sending to FastAPI) ---")
    
    # tqdm creates a progress bar in the terminal
    for row in tqdm(test_dataset, desc="Evaluating"):
        question = row['sql_prompt']
        ground_truth_schema = row['sql_context']
        
        try:
            # 1. Send the question to your local server
            start_time = time.time()
            response = requests.post(
                API_URL, 
                json={"natural_language_query": question},
                timeout=30 # Wait up to 30 seconds for the LLM
            )
            
            if response.status_code == 200:
                successful_api_calls += 1
                data = response.json()
                generated_sql = data["sql_query"]
                retrieved_schema = data["retrieved_schema"]
                
                # --- METRIC 1: Check Syntax Validity ---
                try:
                    sqlglot.parse_one(generated_sql)
                    valid_syntax_count += 1
                except sqlglot.errors.ParseError:
                    pass # Invalid syntax
                    
                # --- METRIC 2: Check RAG Retrieval Accuracy ---
                # We check if the table name from the ground truth is inside our retrieved schema
                # (Extracting just the table name from 'CREATE TABLE my_table (...)')
                try:
                    expected_table_name = ground_truth_schema.split("CREATE TABLE ")[1].split(" ")[0].split("(")[0]
                    if expected_table_name.lower() in retrieved_schema.lower():
                        correct_table_retrieved += 1
                except IndexError:
                    pass # Failsafe if the ground truth schema was formatted weirdly
                    
        except requests.exceptions.RequestException as e:
            print(f"\nAPI Error: Ensure your FastAPI server is running! ({e})")
            break
            
    # --- 3. PRINT FINAL REPORT ---
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    print(f"Total Samples Tested:     {NUM_TEST_SAMPLES}")
    
    if successful_api_calls > 0:
        syntax_accuracy = (valid_syntax_count / successful_api_calls) * 100
        rag_accuracy = (correct_table_retrieved / successful_api_calls) * 100
        
        print(f"Successful API Calls:     {successful_api_calls}/{NUM_TEST_SAMPLES}")
        print(f"SQL Syntax Accuracy:      {syntax_accuracy:.1f}%")
        print(f"RAG Retrieval Accuracy:   {rag_accuracy:.1f}%")
        
    else:
        print("All API calls failed. Check the server")
    print("="*40)

if __name__ == "__main__":
    evaluate_api()