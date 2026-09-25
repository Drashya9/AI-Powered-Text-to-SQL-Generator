import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from evaluate_lora import DATA_PATH, build_held_out_test_set
from rag_builder import clean_schema

# --- CONFIG: override to test query expansion on top of a different retriever ---
INDEX_PATH = "vector_store_finetuned_mpnet_augmented"
EMBEDDING_MODEL = "finetuned_embedding_model_mpnet_augmented"
PARAPHRASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # smaller than the 1.5B generation model -- paraphrasing needs less capacity, and this leaves more memory headroom
CHECK_KS = [1, 10, 50]
POOL_K = 50  # retrieved per query variant, before fusion

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
    return text if text else question  # fall back to the original if generation is empty


def fuse_by_best_distance(*result_lists):
    """Each result_list is [(Document, distance), ...] from one query variant.
    Keeps, per unique schema text, the smallest (best) distance seen across
    all variants -- i.e. a candidate only needs to be close for ONE phrasing
    of the question to rank well."""
    best = {}
    for results in result_lists:
        for doc, distance in results:
            text = doc.page_content
            if text not in best or distance < best[text]:
                best[text] = distance
    return sorted(best.items(), key=lambda kv: kv[1])  # ascending distance = most similar first


def main():
    print("--- Loading dataset and held-out test set ---")
    df = pd.read_json(DATA_PATH, lines=True, orient="records")
    test_rows = build_held_out_test_set(df)
    n = len(test_rows)
    print(f"Test set size: {n} rows")

    print(f"--- Loading fine-tuned mpnet retriever + vector store ({INDEX_PATH}) ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    print(f"--- Loading paraphrase model: {PARAPHRASE_MODEL} (inference only) ---")
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    # bfloat16 halves this model's memory footprint (~6.2GB -> ~3.1GB in fp32
    # vs bf16) -- needed given this project's 8GB RAM ceiling, confirmed via
    # an earlier thrashing incident (0.02s CPU time used after 33 minutes of
    # wall-clock time) even for inference-only loading alongside the FAISS
    # index and its own embedding model.
    model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL, torch_dtype=torch.bfloat16)

    correct_at_k = {k: 0 for k in CHECK_KS}

    for i, row in enumerate(test_rows.itertuples(), 1):
        true_schema = clean_schema(row.schema).strip()

        paraphrase = generate_paraphrase(model, tokenizer, row.instruction)
        docs_orig = vectorstore.similarity_search_with_score(row.instruction, k=POOL_K)
        docs_para = vectorstore.similarity_search_with_score(paraphrase, k=POOL_K)

        ranked = fuse_by_best_distance(docs_orig, docs_para)
        ranked_candidates = [text for text, _ in ranked[:max(CHECK_KS)]]

        for k in CHECK_KS:
            if any(c.strip() == true_schema for c in ranked_candidates[:k]):
                correct_at_k[k] += 1

        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} processed... (example paraphrase: {row.instruction!r} -> {paraphrase!r})")

    print("\n" + "=" * 60)
    print(f"QUERY-EXPANSION RETRIEVAL (original + 1 LLM paraphrase, fused by best distance) (n={n})")
    print("=" * 60)
    for k in CHECK_KS:
        print(f"Correct schema in top-{k}: {100*correct_at_k[k]/n:.1f}%")
    print("\n(compare: augmented mpnet alone -- top-1 51.0%, top-10 81.0%, top-50 90.0%;"
          " production mpnet alone -- top-1 42.0%, top-10 80.0%, top-50 90.0%)")


if __name__ == "__main__":
    main()
