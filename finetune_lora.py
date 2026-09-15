import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import build_prompt

# --- CONFIG ---
DATA_PATH = "processed_data/train_data.jsonl"
BASE_MODEL = "NumbersStation/nsql-350M"
ADAPTER_OUTPUT_DIR = "lora_adapter"
MAX_LENGTH = 384

# CPU-only box (no CUDA here), so we fine-tune on a subset rather than the
# full 100k rows -- this keeps a full run to well under half an hour while
# still being a real, representative fine-tune. Bump this up if you have a
# GPU or more time.
TRAIN_SUBSET_SIZE = 500
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4  # effective batch size = 8
EPOCHS = 1
LEARNING_RATE = 2e-4

# A handful of held-out questions to sanity-check base vs. fine-tuned output.
EVAL_QUESTIONS = [
    "How many funding sources do we have?",
    "What is the total volume of timber sold by each salesperson?",
    "List all customers who made a purchase in the last month.",
]


class SQLDataset(Dataset):
    """Wraps tokenized (prompt, completion) pairs. Loss is only computed on
    the completion tokens -- the prompt is masked out with label = -100 so
    the model isn't wasting capacity learning to reproduce the schema text
    back to us, only learning to write correct SQL for it."""

    def __init__(self, rows, tokenizer):
        self.examples = []
        for row in rows:
            prompt = build_prompt(row["instruction"], row["schema"])
            completion = f"-- {row['explanation']}\n{row['sql']}" + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH).input_ids
            full_ids = tokenizer(
                prompt + completion, truncation=True, max_length=MAX_LENGTH
            ).input_ids

            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100  # mask prompt tokens out of the loss

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"]) for ex in batch]
        labels = [torch.tensor(ex["labels"]) for ex in batch]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }

    return collate_fn


def generate_answers(model, tokenizer, questions, schema="(schema unknown for this quick check)"):
    """Quick before/after sanity check -- not a rigorous eval, just a gut check
    that the fine-tune moved the model in the right direction."""
    model.eval()
    outputs = []
    for question in questions:
        prompt = build_prompt(question, schema)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text.split("### Response:")[-1].strip())
    return outputs


def main():
    print(f"--- 1. Loading training data (subset of {TRAIN_SUBSET_SIZE}) ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    subset = df.sample(n=min(TRAIN_SUBSET_SIZE, len(df)), random_state=42).to_dict("records")

    print(f"--- 2. Loading base model: {BASE_MODEL} ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("--- 3. Baseline generations (before fine-tuning) ---")
    before = generate_answers(base_model, tokenizer, EVAL_QUESTIONS)

    print("--- 4. Wrapping model with LoRA adapters ---")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv_proj", "out_proj"],  # attention projections, confirmed via named_modules()
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("--- 5. Tokenizing dataset ---")
    dataset = SQLDataset(subset, tokenizer)

    print("--- 6. Training ---")
    training_args = TrainingArguments(
        output_dir="lora_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )
    trainer.train()

    print(f"--- 7. Saving LoRA adapter to {ADAPTER_OUTPUT_DIR} ---")
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

    print("--- 8. Fine-tuned generations (after LoRA) ---")
    after = generate_answers(model, tokenizer, EVAL_QUESTIONS)

    print("\n" + "=" * 60)
    print("BEFORE vs. AFTER (same questions, no retrieved schema -- generic check)")
    print("=" * 60)
    for q, b, a in zip(EVAL_QUESTIONS, before, after):
        print(f"\nQ: {q}")
        print(f"  BEFORE: {b}")
        print(f"  AFTER:  {a}")

    print(f"\nDone. Adapter saved to: {ADAPTER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
