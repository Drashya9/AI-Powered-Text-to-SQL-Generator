import re
import torch
import pandas as pd
import sqlglot
from sqlglot import exp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIG ---
DATA_PATH = "processed_data/train_data.jsonl"
BASE_MODEL = "NumbersStation/nsql-350M"
ADAPTER_PATH = "lora_adapter"

# Must match finetune_lora.py's TRAIN_SUBSET_SIZE / seed exactly, so we know
# precisely which 500 rows the adapter was trained on and can exclude them
# from the test set -- otherwise this would be testing on training data.
TRAIN_SUBSET_SIZE = 500
TRAIN_SEED = 42

TEST_FRACTION_OF_TRAIN = 0.20  # "almost 20%" of the training set size
TEST_SEED = 123
MAX_NEW_TOKENS = 120

# Same prompt template main.py actually uses in production -- this is the
# format that matters, since that's what the LoRA adapter has to work under
# regardless of what format it was trained on.
PROMPT_TEMPLATE = """You are a SQL expert. Write a SQL query to answer the following question based on the provided schema.

Schema:
{schema}

Question: {question}
SQL Query:"""


def extract_sql(full_text: str) -> str:
    return full_text.split("SQL Query:")[-1].strip()


def build_held_out_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs the exact 500-row training sample (same seed as
    finetune_lora.py) and excludes it, then samples a held-out test set from
    what's left. Guarantees zero overlap between train and test rows."""
    train_rows = df.sample(n=TRAIN_SUBSET_SIZE, random_state=TRAIN_SEED)
    remaining = df.drop(train_rows.index)
    n_test = int(TRAIN_SUBSET_SIZE * TEST_FRACTION_OF_TRAIN)
    return remaining.sample(n=n_test, random_state=TEST_SEED)


def referenced_tables(sql_text: str) -> set:
    try:
        parsed = sqlglot.parse_one(sql_text)
        return {t.name.lower() for t in parsed.find_all(exp.Table) if t.name}
    except Exception:
        return set()


def schema_tables(schema_text: str) -> set:
    return {m.lower() for m in re.findall(r"CREATE TABLE\s+(\w+)", schema_text, re.IGNORECASE)}


def normalize_sql(sql_text: str) -> str:
    return re.sub(r"\s+", " ", sql_text.strip().lower()).rstrip(";")


def generate_sql(model, tokenizer, question: str, schema: str) -> str:
    prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_sql(text)


def evaluate(model, tokenizer, test_rows, label: str) -> dict:
    n = len(test_rows)
    valid_syntax = 0
    grounded = 0
    exact_match = 0

    print(f"\n--- Evaluating: {label} ({n} held-out examples) ---")
    for i, row in enumerate(test_rows.itertuples(), 1):
        generated = generate_sql(model, tokenizer, row.instruction, row.schema)

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

        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} processed...")

    return {
        "label": label,
        "n": n,
        "syntax_valid_pct": 100 * valid_syntax / n,
        "schema_grounded_pct": 100 * grounded / n,
        "exact_match_pct": 100 * exact_match / n,
    }


def print_report(results: list):
    print("\n" + "=" * 60)
    print(f"HELD-OUT TEST SET REPORT (n={results[0]['n']}, never seen in training)")
    print("=" * 60)
    header = f"{'Metric':<28}" + "".join(f"{r['label']:>16}" for r in results)
    print(header)
    print("-" * len(header))
    for key, name in [
        ("syntax_valid_pct", "SQL syntax valid %"),
        ("schema_grounded_pct", "Schema-grounded %"),
        ("exact_match_pct", "Exact match %"),
    ]:
        row = f"{name:<28}" + "".join(f"{r[key]:>15.1f}%" for r in results)
        print(row)


def main():
    print("--- Loading dataset and building held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    print(f"Test set size: {len(test_rows)} rows (excluded from the {TRAIN_SUBSET_SIZE}-row training set)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("\n--- Loading BASE model ---")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_results = evaluate(base_model, tokenizer, test_rows, "Base (nsql-350M)")

    print("\n--- Loading FINE-TUNED model (base + LoRA adapter, merged) ---")
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).merge_and_unload()
    tuned_results = evaluate(tuned_model, tokenizer, test_rows, "Fine-tuned (LoRA)")

    print_report([base_results, tuned_results])


if __name__ == "__main__":
    main()
