import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from finetune_embeddings import BATCH_SIZE, EPOCHS
from rag_builder import clean_schema

# --- CONFIG ---
# Matches production's actual base model (all-mpnet-base-v2) -- NOT
# finetune_embeddings.py's own BASE_MODEL_NAME default, which is still
# MiniLM; production was built by overriding it at runtime.
BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
AUGMENTED_CACHE_PATH = "augmented_pairs_cache.jsonl"
FINETUNE_OUTPUT_DIR = "finetuned_embedding_model_mpnet_augmented"


def main():
    print(f"--- Loading augmented pairs cache from {AUGMENTED_CACHE_PATH} ---")
    rows = []
    with open(AUGMENTED_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} cached rows (each contributes 2 training examples: original + paraphrase)")

    examples = []
    for row in rows:
        cleaned = clean_schema(row["schema"])
        examples.append(InputExample(texts=[row["instruction"], cleaned]))
        examples.append(InputExample(texts=[row["paraphrase"], cleaned]))
    print(f"Total training examples: {len(examples)} ({len(rows)} original + {len(rows)} paraphrased)")

    print(f"--- Loading base model: {BASE_MODEL_NAME} ---")
    model = SentenceTransformer(BASE_MODEL_NAME)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    # Same loss as the production fine-tune: in-batch negatives, no manual
    # hard-negative mining needed. Doubling the pair count (original +
    # paraphrase per schema) is the actual experiment here -- does phrasing
    # diversity baked into training help more than bolting query expansion
    # on at inference time did (which made things worse, see B12).
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
