import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import build_prompt
from rag_builder import clean_schema
from main import extract_sql, CHAT_SYSTEM_PROMPT

# --- CONFIG: change these two to fine-tune a different model. Chat-template
# detection below picks the right prompt format and LoRA target modules
# automatically -- no other changes needed. ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_OUTPUT_DIR = "lora_adapter_qwen"

DATA_PATH = "processed_data/train_data.jsonl"
MAX_LENGTH = 256  # covers ~90% of training examples without truncation (measured: median 172, p90 245 tokens); reduced from 384 to cut backward-pass activation memory

# CPU-only box (no CUDA here), so we fine-tune on a subset rather than the
# full 100k rows. This exact size/seed must match evaluate_lora.py's
# TRAIN_SUBSET_SIZE/TRAIN_SEED -- that's what guarantees zero overlap with
# the held-out test set, regardless of which model is being trained here.
TRAIN_SUBSET_SIZE = 500
TRAIN_SEED = 42
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


def build_training_texts(tokenizer, question: str, schema: str, sql: str, has_chat_template: bool):
    """Returns (prompt_text, full_text). Loss is only computed on the tokens
    that differ between the two -- the completion/assistant turn -- so the
    model isn't wasting capacity learning to reproduce the schema back to us."""
    if has_chat_template:
        user_content = f"Schema:\n{schema}\n\nQuestion: {question}"
        prompt_messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_messages = prompt_messages + [{"role": "assistant", "content": sql}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        return prompt_text, full_text
    else:
        prompt_text = build_prompt(question, schema)
        full_text = prompt_text + sql + tokenizer.eos_token
        return prompt_text, full_text


class SQLDataset(Dataset):
    """Wraps tokenized (prompt, completion) pairs. Loss is only computed on
    the completion tokens -- the prompt is masked out with label = -100."""

    def __init__(self, rows, tokenizer, has_chat_template: bool):
        self.examples = []
        for row in rows:
            # Clean the schema (drop INSERT sample data) so training input
            # matches what retrieval will actually hand the model in
            # production. Target is SQL only, no explanation -- only ~35% of
            # a completion's tokens were the SQL itself in earlier attempts,
            # the rest was prose nothing downstream ever uses.
            prompt, full = build_training_texts(
                tokenizer, row["instruction"], clean_schema(row["schema"]), row["sql"], has_chat_template
            )

            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH).input_ids
            full_ids = tokenizer(full, truncation=True, max_length=MAX_LENGTH).input_ids

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


def generate_answers(model, tokenizer, questions, has_chat_template: bool, schema="(schema unknown for this quick check)"):
    """Quick before/after sanity check -- not a rigorous eval, just a gut check
    that the fine-tune moved the model in the right direction."""
    model.eval()
    outputs = []
    for question in questions:
        if has_chat_template:
            messages = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            prompt = build_prompt(question, schema)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(extract_sql(text))
    return outputs


def main():
    print(f"--- 1. Loading training data (subset of {TRAIN_SUBSET_SIZE}) ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    subset = df.sample(n=min(TRAIN_SUBSET_SIZE, len(df)), random_state=TRAIN_SEED).to_dict("records")

    print(f"--- 2. Loading base model: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_chat_template = tokenizer.chat_template is not None
    print(f"Chat template detected: {has_chat_template} (using {'chat' if has_chat_template else 'completion'} training format)")

    # Larger chat models (Qwen: 1.5B params, ~6.2GB in fp32) overflow this
    # project's 8GB RAM once backward-pass activations are added on top --
    # confirmed empirically as disk-swap thrashing (0.03s of CPU time used
    # after 28 minutes of wall-clock time). bfloat16 halves the frozen base
    # model's footprint; nsql-350M is small enough to stay at fp32.
    model_dtype = torch.bfloat16 if has_chat_template else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=model_dtype)

    print("--- 3. Baseline generations (before fine-tuning) ---")
    before = generate_answers(base_model, tokenizer, EVAL_QUESTIONS, has_chat_template)

    print("--- 4. Wrapping model with LoRA adapters ---")
    # Attention-only, matching the original nsql fine-tune's spirit (small,
    # cheap adapter). Qwen2's architecture uses separate q/k/v/o projections
    # (confirmed via named_modules()), unlike nsql/CodeGen's fused qkv_proj.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] if has_chat_template else ["qkv_proj", "out_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    if has_chat_template:
        # Trades compute for memory on the backward pass (recomputes
        # activations instead of storing all of them) -- necessary at this
        # model size on 8GB RAM. enable_input_require_grads() is required
        # for checkpointing to work correctly with a frozen base model.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    print("--- 5. Tokenizing dataset ---")
    dataset = SQLDataset(subset, tokenizer, has_chat_template)

    print("--- 6. Training ---")
    # Smaller per-step batch for the larger model, same effective batch size
    # (8) via more accumulation steps -- reduces peak activation memory.
    batch_size = 1 if has_chat_template else BATCH_SIZE
    grad_accum = GRAD_ACCUM_STEPS * BATCH_SIZE if has_chat_template else GRAD_ACCUM_STEPS
    training_args = TrainingArguments(
        output_dir="lora_checkpoints",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
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
    after = generate_answers(model, tokenizer, EVAL_QUESTIONS, has_chat_template)

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
