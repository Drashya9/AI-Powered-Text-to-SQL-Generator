# Progress Log

Working notes on what's been done and what's next. Not user-facing docs (see `README.md` for that) — this is for picking the project back up quickly.

## Done

**RAG pipeline rewritten on LangChain** (`main.py`, `rag_builder.py`)
Replaced hand-rolled FAISS + SentenceTransformer glue code with an actual LCEL chain: `retriever | prompt | llm | StrOutputParser | extract_sql`. This is now a legitimate answer to "built a RAG pipeline using LangChain" — it wasn't before.

**Data cleanup** (`data_loader.py`)
`train_data.jsonl` was storing the same SQL/explanation twice (once inline in a giant formatted prompt, once as separate columns) plus a ~260-char constant boilerplate string repeated 100,000 times. Restructured to store only raw fields (`instruction`, `schema`, `sql`, `explanation`) and reconstruct the full prompt on demand via `build_prompt()`. **138MB → 74MB**, same data.

**Bug found and fixed**: `data_loader.py` had its "delete and recreate `processed_data/`" cleanup sitting at module level, so merely `import`-ing `build_prompt` from it silently wiped the training data. Moved inside `process_data()`.

**Test suite reorganized** into `tests/` (`test_api.py`, `test_rag_chain.py`) — 9 tests, `pytest.ini` added so tests can cleanly `from main import app`.

**LoRA fine-tuning implemented** (`finetune_lora.py`)
Targets `qkv_proj`/`out_proj` (confirmed via inspecting the model's actual module names — CodeGen architecture). Trains 983,040 / 357,695,488 params (0.27%). Loss masked to completion tokens only. Adapter merges into the base model at inference in `main.py` (`PeftModel.from_pretrained(...).merge_and_unload()`), falls back to base model if no adapter exists.

**Real quantitative evaluation built** (`evaluate_lora.py`)
Held-out test set: 100 examples, reconstructed from the exact same seed used for training so there's zero train/test overlap. Compares base vs. fine-tuned on SQL syntax validity, schema-grounding (does generated SQL only reference tables that exist in the given schema), and exact match.

## Current results (500 training examples, 1 epoch, CPU-only)

| Metric | Base | Fine-tuned (LoRA) |
|---|---|---|
| SQL syntax valid % | 75.0% | 79.0% |
| Schema-grounded % | 22.0% | 23.0% |
| Exact match % | 0.0% | 0.0% |

**Takeaway**: LoRA gave a real but modest syntax improvement. Schema-grounding barely moved — in ~3 of 4 generations, the model references a table that doesn't exist in the schema it was given. **This is the actual bottleneck**, not generation quality, and more LoRA training alone probably won't fix it — it's a retrieval/representation problem.

## Known issues / tech debt (not yet fixed)

- `DockerFile` is tracked in git with wrong case (works on Windows' case-insensitive filesystem; will likely break `docker build` on Linux CI/GitHub Actions runners, which expect exact `Dockerfile`).
- `@app.on_event("startup")` is deprecated in current FastAPI — should migrate to `lifespan`.
- No CI pipeline exists (`.github/workflows`) despite resume claiming one.
- No SQL execution against a real database, no read-only DB permission enforcement — despite an earlier interview answer claiming both. Generation only ever produces SQL text.
- Retrieval is `k=1`, no confidence threshold, no re-ranking.
- Training prompt format (`finetune_lora.py` / `data_loader.build_prompt`) doesn't match the serving prompt format (`main.py`'s `PROMPT_TEMPLATE`) — empirically not catastrophic (pytest still passes with valid SQL), but not rigorously validated at scale.

## Next: RAG retrieval improvement roadmap (prioritized)

Given schema-grounding is the real bottleneck:

1. **Build a direct retrieval-only benchmark** — measure whether the retrieved schema is actually correct, isolated from generation quality. Needed before any of the below can be evaluated properly.
2. **Clean schema text before embedding** — currently embeds `INSERT` sample-data noise (`'John Doe'`, `'North'`, etc.) along with the actual structure. Strip to just `CREATE TABLE` definitions.
3. **Fix the similarity metric** — `rag_builder.py` uses `IndexFlatL2` (Euclidean); `all-MiniLM-L6-v2` is benchmarked on cosine similarity. Switch to `IndexFlatIP` with normalized vectors.
4. **Two-stage retrieval**: FAISS top-k=10 (cheap/coarse) → cross-encoder re-rank → top-1. Likely the single biggest lever.
5. **Fine-tune the embedding model** on `(instruction, schema)` pairs already in the dataset (contrastive loss) — the retrieval-side mirror of the LoRA work already done on generation.
6. **Hybrid search** — combine dense (FAISS) with sparse keyword (BM25) to catch literal table/column name matches embeddings can miss.
7. **Query rewriting / HyDE** — embed a hypothetical schema description generated from the question, instead of the raw question, to close the vocabulary gap.
8. **Confidence threshold + fallback** — surface the actual similarity score; reject/flag low-confidence retrievals instead of always returning *something*.

## Interview-claim honesty tracker

| Claim | Status |
|---|---|
| RAG pipeline built with LangChain | ✅ True now |
| FAISS + schema-aware retrieval | ✅ True |
| 89.6% execution accuracy | ⚠️ Metric is actually syntax-validity + retrieval-accuracy, not execution accuracy (nothing executes SQL) |
| Executes SQL against PostgreSQL | ❌ Not implemented — never connects to a real DB |
| Read-only DB permission enforcement | ❌ Not implemented |
| 120+ PyTest cases | ❌ 9 tests currently |
| CI/CD pipeline | ❌ Not implemented |
