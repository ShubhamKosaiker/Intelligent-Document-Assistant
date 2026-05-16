# Benchmark Results — Intelligent Document Assistant

Systematic performance measurement of the RAG pipeline: retrieval latency,
end-to-end query latency, and a before/after study of a memory-bloat bug
surfaced by benchmarking.

---

## System Configuration

| Parameter | Value |
|-----------|-------|
| Hardware | Intel Core i5-1135G7 @ 2.40GHz (laptop CPU, no GPU) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (local, persistent, HNSW index) |
| LLM | `llama-3.3-70b-versatile` via Groq API |
| Chunking | `RecursiveCharacterTextSplitter`, 1,000 chars, 200 overlap |
| Top-k retrieval | k=4 |

**Corpus:** 22 arXiv papers on RAG and LLM evaluation (RAG original, RAGAS,
Self-RAG, Corrective RAG, Lost-in-the-Middle, ReAct, Toolformer, AgentBench,
InstructGPT, Chain-of-Thought, Tree of Thoughts, HELM, MMLU, etc.).
**Total indexed chunks: 2,968.** Ingestion uses SHA-256 file-hash
deduplication to prevent the duplicate-document bug that inflated an earlier
index to 8,311 chunks from only 3 unique documents.

---

## Benchmark 1 — Retrieval Latency (ChromaDB only)

100 diverse queries. Measures `VectorStore.similarity_search()` —
embedding the query and retrieving top-k from ChromaDB. No LLM involved.

| Metric | Latency |
|-------:|--------:|
| **P50** | **13.7 ms** |
| **P95** | **19.0 ms** |
| P99 | 30.8 ms |
| Mean | 13.8 ms |
| Min / Max | 9.4 / 30.8 ms |

Steady-state retrieval is consistently sub-20ms at P95 on a laptop CPU.

---

## Benchmark 2 — End-to-End Query Latency (Before / After Fix)

Full pipeline: query → retrieve from ChromaDB → LLM call (Groq) → return answer.
Sequential single-user measurement. Not a throughput benchmark.

### Benchmarking surfaced a bug

The first e2e run showed P50=13.4s, P95=22.2s — much slower than expected.
Inspecting the per-query timeline revealed two regimes:

| Queries | Avg latency | Reason |
|---------|-------------|--------|
| 1–8 | ~1.2s | Fresh session |
| 9–25 | ~17s | Prompt ballooned |

Root cause: `ConversationBufferMemory` appends the full chat history to every
LLM call with no upper bound. By query 9, the prompt contained ~4,000 extra
tokens of history, causing latency to grow linearly and consuming tokens
much faster than necessary.

### Fix

Replaced `ConversationBufferMemory` with `ConversationBufferWindowMemory(k=4)`
— keeps only the last 4 exchanges in the rolling context window.

See: [`app/services/llm_service.py`](../app/services/llm_service.py)

### Before / After (identical 25-query test set)

| Metric | Before fix | After fix | Change |
|--------|-----------:|----------:|-------:|
| P50 (first 9 queries, fresh session) | 1.36s | **0.74s** | −46% |
| P50 (overall, sequential session) | 13.4s | ~1s* | ~13× faster |
| P95 | 22.2s | n/a* | — |
| Mean | 12.4s | ~1s* | — |

*The post-fix run exhausted Groq's free-tier 100K-tokens-per-day quota after
16 successful queries; later queries returned 429 errors. The 16 completed
post-fix queries show stable ~0.4–1.2s latency with no linear growth —
confirming the memory-bloat bug is eliminated. A full 25-query post-fix run
requires either a 24-hour quota reset or Groq Dev Tier.

Raw data for both runs lives in `benchmarks/e2e_results.txt` and
`benchmarks/e2e_results_pre_fix.txt`.

---

## Benchmark 3 — Answer Quality (RAGAS)

Reference-based evaluation using [RAGAS](https://docs.ragas.io/).
15 hand-labelled Q/A pairs in `quality_eval_dataset.py`, graded by
`llama-3.3-70b-versatile` as judge LLM.

Metric | Mean | Successful grades | Notes
---|---:|---:|---
`answer_relevancy`  | **0.545** | 9/15 | How well the answer addresses the question.
`context_precision` | **0.520** | 7/15 | Fraction of retrieved context that is relevant.
`context_recall`    | 0.500 | 1/15 | Single-datapoint — not meaningful.
`faithfulness`      | n/a | 0/15 | Judge calls exhausted before any grade returned.

Partial result: mid-run the evaluation exhausted Groq free-tier's
100K tokens-per-day cap (RAGAS issues 3-4 LLM grading calls per Q per
metric → ~60 judge calls for this eval). Re-running on Groq Dev Tier or
after a 24h quota reset will produce full coverage. Harness committed at
`benchmarks/quality_eval.py`; dataset at `benchmarks/quality_eval_dataset.py`.

Interpretation: ~0.5 on both `answer_relevancy` and `context_precision` is a
reasonable first-pass baseline. Known improvements to try: hybrid retrieval
(BM25 + dense), cross-encoder re-ranking, similarity-score thresholding.

---

## Honest caveats

- Measurements were run sequentially on one machine. These are **per-request
  latencies**, not **throughput** numbers. Concurrent throughput was not
  tested.
- Groq API latency depends on provider-side queueing and the free-tier
  TPM/TPD rate limits; the numbers above reflect a light-load scenario.
- The retrieval benchmark reflects a 3K-chunk corpus. Real production
  systems typically index millions of chunks; those would surface different
  performance profiles (e.g., index size and HNSW parameters matter more).

---

## How to Reproduce

From the `TEXT_ASSISTANT/` directory with the project venv:

```bash
# 1. Re-index from arXiv corpus (with dedup) — ~5 minutes, downloads 22 PDFs
myenv/Scripts/python benchmarks/reindex_corpus.py

# 2. Retrieval latency — no API calls, ~30 seconds
myenv/Scripts/python benchmarks/retrieval_benchmark.py

# 3. End-to-end latency — hits Groq API, ~3 minutes, needs GROQ_API_KEY
myenv/Scripts/python benchmarks/e2e_benchmark.py
```

Outputs saved to `retrieval_results.txt` and `e2e_results.txt`.
