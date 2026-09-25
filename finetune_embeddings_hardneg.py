import json
import os
import random
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from finetune_embeddings import BATCH_SIZE, EPOCHS, build_finetune_pairs
from evaluate_lora import DATA_PATH
from rag_builder import clean_schema

# --- CONFIG ---
# Same original (non-augmented) 3,000-pair dataset as production, so the ONLY
# change vs. the 42% baseline is the explicit hard negative per example.
BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Hard negatives are mined from the ORIGINAL production retriever (hardcoded
# rather than imported from main.py so this stays reproducible even if
# production later switches to a different embedding model).
MINING_INDEX_PATH = "vector_store_finetuned_mpnet"
MINING_EMBEDDING_MODEL = "finetuned_embedding_model_mpnet"
MINING_TOP_K = 10           # retrieve this many candidates per training question
NEGATIVE_POOL = 5           # sample the negative from the top-N WRONG candidates
MINING_SEED = 1234
TRIPLETS_CACHE_PATH = "hardneg_triplets_cache.jsonl"
FINETUNE_OUTPUT_DIR = "finetuned_embedding_model_mpnet_hardneg"


def mine_triplets(pairs_df):
    """For each training question, retrieve top-K from the production index,
    drop the true schema, and sample one hard negative from the closest
    NEGATIVE_POOL wrong candidates. Sampling (rather than always taking the
    single closest wrong schema) guards against false negatives -- this
    dataset has many near-duplicate schemas (e.g. ten different
    funding_sources tables) that would be wrongly treated as negatives if
    always picked at rank 1."""
    if os.path.exists(TRIPLETS_CACHE_PATH):
        triplets = []
        with open(TRIPLETS_CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triplets.append(json.loads(line))
        if len(triplets) == len(pairs_df):
            print(f"Loaded {len(triplets)} cached triplets from {TRIPLETS_CACHE_PATH}")
            return triplets
        print(f"Cache has {len(triplets)} rows but expected {len(pairs_df)}; re-mining")

    print(f"--- Loading production retriever for mining ({MINING_INDEX_PATH}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=MINING_EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(MINING_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    rng = random.Random(MINING_SEED)
    triplets = []
    skipped = 0
    for i, row in enumerate(pairs_df.itertuples(), 1):
        true_cleaned = clean_schema(row.schema).strip()
        docs = vectorstore.similarity_search(row.instruction, k=MINING_TOP_K)
        wrong = [d.page_content for d in docs if d.page_content.strip() != true_cleaned]
        if not wrong:
            skipped += 1
            continue
        negative = rng.choice(wrong[:NEGATIVE_POOL])
        triplets.append({
            "instruction": row.instruction,
            "positive": true_cleaned,
            "negative": negative,
        })
        if i % 250 == 0 or i == len(pairs_df):
            print(f"  mined {i}/{len(pairs_df)}...")

    print(f"Mined {len(triplets)} triplets ({skipped} skipped: no wrong candidate in top-{MINING_TOP_K})")
    with open(TRIPLETS_CACHE_PATH, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")
    return triplets


def main():
    print("--- Loading dataset and reconstructing the same 3,000-row fine-tuning pool as production ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    pairs_df = build_finetune_pairs(df)
    print(f"Pool size: {len(pairs_df)} rows")

    triplets = mine_triplets(pairs_df)

    examples = [InputExample(texts=[t["instruction"], t["positive"], t["negative"]]) for t in triplets]
    print(f"Total training examples: {len(examples)} (question, true schema, hard negative)")

    print(f"--- Loading base model: {BASE_MODEL_NAME} ---")
    model = SentenceTransformer(BASE_MODEL_NAME)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    # MultipleNegativesRankingLoss natively accepts (anchor, positive, negative)
    # triplets: the explicit hard negative is used IN ADDITION to the usual
    # in-batch negatives, so this is a strict superset of the baseline setup.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"--- Training for {EPOCHS} epochs, batch size {BATCH_SIZE} ---")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(train_dataloader) * EPOCHS),
        show_progress_bar=True,
    )

    model.save(FINETUNE_OUTPUT_DIR)
    print(f"\nDone. Hard-negative fine-tuned embedding model saved to: {FINETUNE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
