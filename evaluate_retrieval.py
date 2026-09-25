import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from evaluate_lora import DATA_PATH, build_held_out_test_set
from rag_builder import clean_schema

# --- CONFIG: change these two lines to evaluate a different retriever. One
# script for every embedding model / index tested, instead of a new file each
# time. ---
INDEX_PATH = "vector_store_finetuned_mpnet_maxpool"
EMBEDDING_MODEL = "finetuned_embedding_model_mpnet_maxpool"
LABEL = "max-pooling mpnet (original 3,000 pairs)"

CHECK_KS = [1, 10, 50]

# Reference results from earlier runs, for context in the printed output only.
KNOWN_RESULTS = {
    "production mpnet (mean pooling, 3,000 pairs)": (42.0, 80.0, 90.0),
    "augmented mpnet (mean pooling, 6,000 pairs)": (51.0, 81.0, 90.0),
}


def main():
    print("--- Loading dataset and held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    n = len(test_rows)
    print(f"Test set size: {n} rows")

    print(f"--- Loading retriever + vector store ({INDEX_PATH}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    correct_at_k = {k: 0 for k in CHECK_KS}

    for i, row in enumerate(test_rows.itertuples(), 1):
        true_schema = clean_schema(row.schema).strip()
        docs = vectorstore.similarity_search(row.instruction, k=max(CHECK_KS))
        candidates = [d.page_content for d in docs]

        for k in CHECK_KS:
            if any(c.strip() == true_schema for c in candidates[:k]):
                correct_at_k[k] += 1

        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} processed...")

    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION: {LABEL} (n={n})")
    print("=" * 60)
    for k in CHECK_KS:
        print(f"Correct schema in top-{k}: {100*correct_at_k[k]/n:.1f}%")
    print("\nReference results:")
    for name, (t1, t10, t50) in KNOWN_RESULTS.items():
        print(f"  {name}: top-1 {t1}%, top-10 {t10}%, top-50 {t50}%")


if __name__ == "__main__":
    main()
