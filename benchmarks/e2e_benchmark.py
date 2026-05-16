"""
End-to-end query latency benchmark for the Intelligent Document Assistant.
Measures wall-clock time for the full pipeline: retrieve → LLM call (Groq) → return answer.
Sequential single-user measurement — this is per-request latency, not throughput.

Run from TEXT_ASSISTANT/ root:
    myenv/Scripts/python benchmarks/e2e_benchmark.py

Note: each query hits the Groq API. 25 queries ~ 25 API calls.
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join('app', '.env'))

from models.vector_store import VectorStore
from services.llm_service import LLMService
from config import Config

# 25 diverse queries — enough for honest P50/P95 without burning API quota
QUERIES = [
    "What are the symptoms of diabetes and how is it diagnosed?",
    "Explain the difference between Type 1 and Type 2 diabetes.",
    "What projects has Shubham Kosaiker worked on?",
    "What are the risk factors for cardiovascular disease?",
    "How does retrieval-augmented generation reduce hallucinations?",
    "What is the treatment for hypertension?",
    "What LLM frameworks and tools has Shubham used in his projects?",
    "How does the immune system respond to bacterial infections?",
    "What is the role of embeddings in semantic search?",
    "Describe the symptoms and treatment of pneumonia.",
    "What cloud services has Shubham Kosaiker deployed to?",
    "How does chemotherapy work to treat cancer?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the causes and symptoms of anemia?",
    "How does ChromaDB store and retrieve vector embeddings?",
    "What is the significance of the deep research benchmark dataset?",
    "How is Alzheimer's disease diagnosed and managed?",
    "What Python libraries are commonly used for NLP tasks?",
    "What are autoimmune diseases and how are they treated?",
    "Explain the architecture of a transformer model.",
    "What is the prognosis for patients with COPD?",
    "How does LangChain's ConversationalRetrievalChain work?",
    "What are the stages of clinical drug trials?",
    "What is the function of the liver and what happens in liver failure?",
    "How does vector similarity search differ from keyword search?",
]

assert len(QUERIES) == 25


def main():
    print("Loading vector store and LLM service...")
    vs = VectorStore(Config.VECTOR_DB_PATH)
    llm = LLMService(vs)

    chunk_count = vs.vector_store._collection.count()
    print(f"Indexed chunks : {chunk_count:,}")
    print(f"LLM            : llama-3.3-70b-versatile (Groq)")
    print(f"Running {len(QUERIES)} end-to-end queries...\n")
    print("(Each query = retrieve from ChromaDB + Groq API call)")
    print()

    latencies_s = []
    failed = 0

    for i, query in enumerate(QUERIES, 1):
        print(f"  [{i:02d}/{len(QUERIES)}] {query[:60]}...")
        t0 = time.perf_counter()
        try:
            answer = llm.get_response(query)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            # Detect rate-limit / internal errors swallowed by llm_service
            if "encountered an error" in answer.lower() or len(answer) < 30:
                failed += 1
                print(f"         FAILED ({elapsed:.2f}s): rate-limit or empty response")
            else:
                latencies_s.append(elapsed)
                print(f"         {elapsed:.2f}s  | answer: {answer[:80].strip()}...")
        except Exception as e:
            t1 = time.perf_counter()
            failed += 1
            print(f"         FAILED ({t1-t0:.2f}s): {e}")
        # Small delay between requests to stay under Groq TPM limit
        time.sleep(2)
        print()

    if not latencies_s:
        print("All queries failed. Check your GROQ_API_KEY.")
        return

    latencies_s.sort()
    p50 = statistics.median(latencies_s)
    p95 = latencies_s[max(0, int(len(latencies_s) * 0.95) - 1)]
    mean = statistics.mean(latencies_s)
    mn   = min(latencies_s)
    mx   = max(latencies_s)

    print("="*55)
    print("END-TO-END QUERY LATENCY BENCHMARK RESULTS")
    print("="*55)
    print(f"Hardware    : Intel Core i5-1135G7 @ 2.40GHz (laptop CPU)")
    print(f"Embedding   : sentence-transformers/all-MiniLM-L6-v2")
    print(f"LLM         : llama-3.3-70b-versatile via Groq API")
    print(f"Chunks      : {chunk_count:,}")
    print(f"Queries run : {len(latencies_s)} succeeded, {failed} failed")
    print(f"Measurement : sequential per-request latency (not throughput)")
    print()
    print(f"  P50  : {p50:.2f}s")
    print(f"  P95  : {p95:.2f}s")
    print(f"  Mean : {mean:.2f}s")
    print(f"  Min  : {mn:.2f}s")
    print(f"  Max  : {mx:.2f}s")
    print("="*55)

    with open(os.path.join(os.path.dirname(__file__), 'e2e_results.txt'), 'w') as f:
        f.write(f"Indexed chunks: {chunk_count}\n")
        f.write(f"LLM: llama-3.3-70b-versatile (Groq)\n")
        f.write(f"Queries succeeded: {len(latencies_s)}\n")
        f.write(f"P50:  {p50:.2f}s\n")
        f.write(f"P95:  {p95:.2f}s\n")
        f.write(f"Mean: {mean:.2f}s\n")
        f.write(f"Min:  {mn:.2f}s\n")
        f.write(f"Max:  {mx:.2f}s\n")
        f.write("\nRaw latencies (s):\n")
        for lat in sorted(latencies_s):
            f.write(f"  {lat:.3f}\n")

    print(f"\nRaw results saved to benchmarks/e2e_results.txt")


if __name__ == '__main__':
    main()
