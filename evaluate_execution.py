import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from evaluate_lora import (
    DATA_PATH,
    BASE_MODEL,
    ADAPTER_PATH,
    build_held_out_test_set,
    generate_sql,
)

# --- CONFIG ---
# Override this (e.g. via TEST_LIMIT = 10) for a quick smoke test.
TEST_LIMIT = None


def run_query(schema_script: str, sql: str):
    """Materializes `schema_script` (CREATE TABLE + INSERT statements already
    present in the dataset) in a fresh in-memory SQLite DB, then runs `sql`
    against it. Returns (rows, error) -- rows is None if either the schema
    failed to set up (dialect mismatch) or the query itself errored."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_script)
    except sqlite3.Error as e:
        conn.close()
        return None, f"schema setup failed: {e}"

    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows, None
    except sqlite3.Error as e:
        conn.close()
        return None, f"query failed: {e}"


def results_match(a, b) -> bool:
    """Row order is undefined in SQL without ORDER BY, so we ignore it;
    column order and values still have to match exactly."""
    return sorted(a) == sorted(b)


def evaluate_execution(model, tokenizer, test_rows, label: str) -> dict:
    n = len(test_rows)
    schema_ok = 0
    gen_executes = 0
    exec_match = 0

    print(f"\n--- Execution-based evaluation: {label} ({n} examples) ---")
    for i, row in enumerate(test_rows.itertuples(), 1):
        # Establish ground truth first (cheap, no model call). If the
        # dataset's own SQL doesn't even run against its own schema in
        # SQLite (dialect mismatch), we can't judge correctness -- skip,
        # and don't waste a model generation on it.
        gt_rows, gt_err = run_query(row.schema, row.sql)
        if gt_err is not None:
            if i % 10 == 0 or i == n:
                print(f"  {i}/{n} processed...")
            continue
        schema_ok += 1

        generated = generate_sql(model, tokenizer, row.instruction, row.schema)
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
        "schema_executable_pct": 100 * schema_ok / n,
        "gen_executes_pct": 100 * gen_executes / schema_ok if schema_ok else 0.0,
        "execution_accuracy_pct": 100 * exec_match / schema_ok if schema_ok else 0.0,
        "execution_accuracy_of_full_set_pct": 100 * exec_match / n,
    }


def print_report(results: list):
    print("\n" + "=" * 70)
    print(f"EXECUTION-BASED ACCURACY REPORT (n={results[0]['n']} held-out examples)")
    print("=" * 70)
    header = f"{'Metric':<50}" + "".join(f"{r['label']:>16}" for r in results)
    print(header)
    print("-" * len(header))
    rows = [
        ("schema_executable_pct", "Ground-truth schema executable in SQLite %"),
        ("gen_executes_pct", "Generated SQL executes % (of executable rows)"),
        ("execution_accuracy_pct", "EXECUTION ACCURACY % (of executable rows)"),
        ("execution_accuracy_of_full_set_pct", "Execution accuracy % (of full test set)"),
    ]
    for key, name in rows:
        line = f"{name:<50}" + "".join(f"{r[key]:>15.1f}%" for r in results)
        print(line)


def main():
    print("--- Loading dataset and building held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    if TEST_LIMIT:
        test_rows = test_rows.head(TEST_LIMIT)
    print(f"Test set size: {len(test_rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("\n--- Loading BASE model ---")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_results = evaluate_execution(base_model, tokenizer, test_rows, "Base (nsql-350M)")

    print("\n--- Loading FINE-TUNED model (base + LoRA adapter, merged) ---")
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).merge_and_unload()
    tuned_results = evaluate_execution(tuned_model, tokenizer, test_rows, "Fine-tuned (LoRA)")

    print_report([base_results, tuned_results])


if __name__ == "__main__":
    main()
