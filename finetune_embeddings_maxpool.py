import pandas as pd
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader

from finetune_embeddings import BATCH_SIZE, EPOCHS, build_finetune_pairs
from evaluate_lora import DATA_PATH
from rag_builder import clean_schema

# --- CONFIG ---
# Original (non-augmented) 3,000-pair dataset, same as production -- this
# run isolates ONE variable: pooling strategy. Mean pooling averages every
# token's contribution, which is precisely why every inference-time query
# modification (keywords, paraphrase, hybrid search) diluted the signal and
# hurt accuracy. Max pooling keeps only the most salient dimension per
# feature instead of diluting across all tokens -- untested so far.
BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
FINETUNE_OUTPUT_DIR = "finetuned_embedding_model_mpnet_maxpool"


def main():
    print("--- Loading dataset and reconstructing the same 3,000-row fine-tuning pool as production ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    pairs_df = build_finetune_pairs(df)
    print(f"Pool size: {len(pairs_df)} rows")

    examples = []
    for row in pairs_df.itertuples():
        cleaned = clean_schema(row.schema)
        examples.append(InputExample(texts=[row.instruction, cleaned]))
    print(f"Total training examples: {len(examples)} (original questions only, no augmentation)")

    print(f"--- Building model with MAX pooling: {BASE_MODEL_NAME} ---")
    word_embedding_model = models.Transformer(BASE_MODEL_NAME)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="max")
    normalize_model = models.Normalize()
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model])

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"--- Training for {EPOCHS} epochs, batch size {BATCH_SIZE} ---")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(train_dataloader) * EPOCHS),
        show_progress_bar=True,
    )

    model.save(FINETUNE_OUTPUT_DIR)
    print(f"\nDone. Max-pooling fine-tuned embedding model saved to: {FINETUNE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
