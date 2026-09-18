import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from evaluate_lora import DATA_PATH, build_held_out_test_set, TRAIN_SUBSET_SIZE, TRAIN_SEED
from rag_builder import clean_schema, MODEL_NAME

# --- CONFIG ---
BASE_MODEL_NAME = f"sentence-transformers/{MODEL_NAME}"  # override for a different base model
FINETUNE_OUTPUT_DIR = "finetuned_embedding_model"
FINETUNE_SUBSET_SIZE = 3000  # embedding fine-tuning is much cheaper per-example than LLM
                              # fine-tuning, so we can afford more than LoRA's 500
FINETUNE_SEED = 777
BATCH_SIZE = 16
EPOCHS = 2


def build_finetune_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Excludes both the LoRA training rows AND the 100-row held-out test set
    (same reconstruction as build_held_out_test_set) so there is zero overlap
    between what this trains on and what evaluate_retrieval.py later measures."""
    lora_train_rows = df.sample(n=TRAIN_SUBSET_SIZE, random_state=TRAIN_SEED)
    remaining = df.drop(lora_train_rows.index)
    held_out = build_held_out_test_set(df)  # reconstructs the exact same 100 rows
    excluded = set(lora_train_rows.index) | set(held_out.index)
    pool = df.drop(index=list(excluded))
    return pool.sample(n=min(FINETUNE_SUBSET_SIZE, len(pool)), random_state=FINETUNE_SEED)


def main():
    print("--- Loading dataset ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    pairs_df = build_finetune_pairs(df)
    print(f"Fine-tuning on {len(pairs_df)} (question, schema) pairs "
          f"(excludes LoRA's 500 training rows and the 100-row held-out test set)")

    examples = []
    for row in pairs_df.itertuples():
        cleaned = clean_schema(row.schema)
        examples.append(InputExample(texts=[row.instruction, cleaned]))

    print(f"--- Loading base model: {BASE_MODEL_NAME} ---")
    model = SentenceTransformer(BASE_MODEL_NAME)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    # MultipleNegativesRankingLoss: for each (question, schema) pair, every
    # OTHER schema in the same batch acts as a negative automatically -- no
    # need to hand-mine hard negatives. Standard, well-established approach
    # for fine-tuning sentence-transformers models for retrieval.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"--- Training for {EPOCHS} epochs, batch size {BATCH_SIZE} ---")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(train_dataloader) * EPOCHS),
        show_progress_bar=True,
    )

    model.save(FINETUNE_OUTPUT_DIR)
    print(f"\nDone. Fine-tuned embedding model saved to: {FINETUNE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
