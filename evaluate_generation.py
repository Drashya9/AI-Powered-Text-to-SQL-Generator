import re
import torch
import sqlglot
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from evaluate_lora import (
    DATA_PATH, build_held_out_test_set, referenced_tables, schema_tables, normalize_sql,
)
from evaluate_execution import run_query, results_match
from data_loader import PROMPT_TEMPLATE
from rag_builder import clean_schema

# --- CONFIG: change these to evaluate a different model. That's it -- one
# script for every generation model tested, instead of a new file each time. ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = None    # e.g. "old_models/lora_adapter" to test a LoRA adapter on top of MODEL_NAME; None to skip
LABEL = None           # None = auto-derived from MODEL_NAME/ADAPTER_PATH

MAX_NEW_TOKENS = 120

# Note: schema fed to the model here is the GROUND-TRUTH schema from the
# dataset, not anything retrieved from the vector store -- this isolates
# generation quality from retrieval quality (same methodology as
# evaluate_lora.py). Real end-to-end accuracy through the live /generate
# endpoint depends on retrieval too (42% top-1), so it will be lower than
# these numbers whenever retrieval picks the wrong schema.

# Used only for chat-template models (instruction-tuned). Completion-style
# base models (e.g. nsql-350M) use the shared PROMPT_TEMPLATE instead, matching
# how they were actually trained.
CHAT_SYSTEM_PROMPT = (
    "You are a SQL expert. Write a SQL query to answer the question based on the "
    "provided schema. Use only columns and tables explicitly mentioned in the "
    "provided schema. Respond with ONLY the SQL query, no explanation, no markdown "
    "code fences."
)
CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Reference numbers from prior runs (see metrics.docx Part A), kept here for
# context in the printed table only -- not recomputed. Add a line by hand once
# a new model's run is worth keeping around for future comparisons.
KNOWN_RESULTS = [
    {"label": "nsql-350M (base)", "n": 100, "syntax_valid_pct": 75.0, "schema_grounded_pct": 22.0,
     "exact_match_pct": 0.0, "schema_executable_pct": 75.0, "gen_executes_pct": 5.3,
     "execution_accuracy_pct": 2.7, "execution_accuracy_full_pct": 2.0},
    {"label": "nsql-350M+LoRA (prod)", "n": 100, "syntax_valid_pct": 79.0, "schema_grounded_pct": 23.0,
     "exact_match_pct": 0.0, "schema_executable_pct": 75.0, "gen_executes_pct": 4.0,
     "execution_accuracy_pct": 1.3, "execution_accuracy_full_pct": 1.0},
]


def extract_sql(text: str) -> str:
    fence_match = CODE_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    if "SQL Query:" in text:
        return text.split("SQL Query:")[-1].strip()
    return text.strip()


def generate_sql(model, tokenizer, question: str, schema: str, has_chat_template: bool) -> str:
    cleaned = clean_schema(schema)
    if has_chat_template:
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{cleaned}\n\nQuestion: {question}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        prompt = PROMPT_TEMPLATE.format(schema=cleaned, question=question)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3, no_repeat_ngram_size=3,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_sql(text)


def evaluate(model, tokenizer, test_rows, label: str, has_chat_template: bool) -> dict:
    n = len(test_rows)
    valid_syntax = 0
    grounded = 0
    exact_match = 0
    schema_ok = 0
    gen_executes = 0
    exec_match = 0

    print(f"\n--- Evaluating: {label} ({n} held-out examples) ---")
    for i, row in enumerate(test_rows.itertuples(), 1):
        generated = generate_sql(model, tokenizer, row.instruction, row.schema, has_chat_template)

        is_valid = False
        try:
            sqlglot.parse_one(generated)
            is_valid = True
            valid_syntax += 1
        except Exception:
            pass

        if is_valid:
            gen_tables = referenced_tables(generated)
            valid_tables = schema_tables(row.schema)
            if gen_tables and gen_tables.issubset(valid_tables):
                grounded += 1

        if normalize_sql(generated) == normalize_sql(row.sql):
            exact_match += 1

        # Execution-based metrics, reusing the same generation above rather
        # than generating a second time.
        gt_rows, gt_err = run_query(row.schema, row.sql)
        if gt_err is None:
            schema_ok += 1
            gen_rows, gen_err = run_query(row.schema, generated)
            if gen_err is None:
                gen_executes += 1
                if results_match(gen_rows, gt_rows):
                    exec_match += 1

        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} processed...")

    return {
        "label": label,
        "n": n,
        "syntax_valid_pct": 100 * valid_syntax / n,
        "schema_grounded_pct": 100 * grounded / n,
        "exact_match_pct": 100 * exact_match / n,
        "schema_executable_pct": 100 * schema_ok / n,
        "gen_executes_pct": 100 * gen_executes / schema_ok if schema_ok else 0.0,
        "execution_accuracy_pct": 100 * exec_match / schema_ok if schema_ok else 0.0,
        "execution_accuracy_full_pct": 100 * exec_match / n,
    }


def print_report(results: list):
    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST SET REPORT (n={results[0]['n']}, never seen in training)")
    print("=" * 70)
    header = f"{'Metric':<48}" + "".join(f"{r['label']:>18}" for r in results)
    print(header)
    print("-" * len(header))
    rows = [
        ("syntax_valid_pct", "SQL syntax valid %"),
        ("schema_grounded_pct", "Schema-grounded %"),
        ("exact_match_pct", "Exact match %"),
        ("schema_executable_pct", "Ground-truth schema executable %"),
        ("gen_executes_pct", "Generated SQL executes % (of executable)"),
        ("execution_accuracy_pct", "EXECUTION ACCURACY % (of executable)"),
        ("execution_accuracy_full_pct", "Execution accuracy % (of full set)"),
    ]
    for key, name in rows:
        line = f"{name:<48}" + "".join(f"{r[key]:>17.1f}%" for r in results)
        print(line)


def main():
    print("--- Loading dataset and building held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    print(f"Test set size: {len(test_rows)} rows")

    print(f"\n--- Loading {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if ADAPTER_PATH:
        print(f"--- Applying LoRA adapter: {ADAPTER_PATH} ---")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH).merge_and_unload()

    has_chat_template = tokenizer.chat_template is not None
    print(f"Chat template detected: {has_chat_template} (using {'chat' if has_chat_template else 'completion'} prompting)")

    label = LABEL or (MODEL_NAME.split("/")[-1] + ("+LoRA" if ADAPTER_PATH else " (zero-shot)"))
    results = evaluate(model, tokenizer, test_rows, label, has_chat_template)

    print_report(KNOWN_RESULTS + [results])


if __name__ == "__main__":
    main()
