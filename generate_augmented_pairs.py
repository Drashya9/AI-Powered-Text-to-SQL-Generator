import json
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate_lora import DATA_PATH
from finetune_embeddings import build_finetune_pairs

# --- CONFIG ---
PARAPHRASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "augmented_pairs_cache.jsonl"

PARAPHRASE_SYSTEM_PROMPT = (
    "Rewrite the given question using different words and synonyms (e.g. \"funding\" -> \"grants\"), "
    "keeping the exact same meaning and intent. Respond with ONLY the rewritten question, nothing else."
)


def generate_paraphrase(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text if text else question


def main():
    print("--- Loading dataset and reconstructing the exact 3,000-row fine-tuning pool ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    pairs_df = build_finetune_pairs(df)  # same seed/exclusions as the production embedding fine-tune
    total = len(pairs_df)
    print(f"Pool size: {total} rows (same as finetune_embeddings.py)")

    # Resume support: this step takes hours, so skip rows already cached from
    # a prior (possibly interrupted) run instead of starting over.
    done_indices = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_indices.add(json.loads(line)["index"])
        print(f"Resuming: {len(done_indices)} rows already cached in {OUTPUT_PATH}")

    print(f"--- Loading paraphrase model: {PARAPHRASE_MODEL} ---")
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL, torch_dtype=torch.bfloat16)

    print("--- Generating paraphrases (writing incrementally, safe to resume if interrupted) ---")
    done_count = len(done_indices)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, (idx, row) in enumerate(pairs_df.iterrows(), 1):
            if idx in done_indices:
                continue
            paraphrase = generate_paraphrase(model, tokenizer, row["instruction"])
            f.write(json.dumps({
                "index": int(idx),
                "instruction": row["instruction"],
                "paraphrase": paraphrase,
                "schema": row["schema"],
            }) + "\n")
            f.flush()
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                print(f"  {done_count}/{total} done...")

    print(f"\nDone. {total} paraphrases cached in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
