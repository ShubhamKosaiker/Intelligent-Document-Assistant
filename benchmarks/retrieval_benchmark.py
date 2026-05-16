"""
Retrieval latency benchmark for the Intelligent Document Assistant.
Measures wall-clock time for ChromaDB similarity_search() — retrieval only, no LLM.

Run from TEXT_ASSISTANT/ root:
    myenv/Scripts/python benchmarks/retrieval_benchmark.py
"""

import sys
import os
import time
import statistics

# Make app/ importable and load .env from app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join('app', '.env'))

from models.vector_store import VectorStore
from config import Config

QUERIES = [
    # factual / medical
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What causes anemia?",
    "Describe the treatment for asthma.",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "What are common side effects of antibiotics?",
    "How does the immune system fight infections?",
    "What is the role of insulin in the body?",
    "What are the risk factors for heart disease?",
    "How is pneumonia treated?",
    # research / technical
    "What is a benchmark dataset for deep research agents?",
    "How are evaluation metrics defined in NLP benchmarks?",
    "What is retrieval-augmented generation?",
    "How do transformer models work?",
    "What is fine-tuning in machine learning?",
    "Explain attention mechanisms in neural networks.",
    "What is the difference between precision and recall?",
    "How does vector similarity search work?",
    "What are embeddings in NLP?",
    "What is RLHF?",
    # resume / personal doc
    "What projects has Shubham Kosaiker worked on?",
    "What are Shubham's technical skills?",
    "What is Shubham's educational background?",
    "Where did Shubham do his research internship?",
    "What LLM frameworks has Shubham used?",
    "What cloud platforms has Shubham worked with?",
    "What is Shubham's experience with Python?",
    "Has Shubham worked with Docker?",
    "What are Shubham's certifications?",
    "What is Shubham's GitHub username?",
    # mixed domain
    "What is the treatment protocol for stroke?",
    "How do RAG systems reduce hallucinations?",
    "What is the significance of P95 latency?",
    "How does ChromaDB store embeddings?",
    "What is LangChain used for?",
    "What are the symptoms of COVID-19?",
    "How does vector search differ from keyword search?",
    "What is Groq and why is it fast?",
    "What are the stages of clinical trials?",
    "What is transfer learning?",
    # broader
    "What causes migraines?",
    "How is depression treated?",
    "What is the difference between MRI and CT scan?",
    "How does chemotherapy work?",
    "What is COPD?",
    "How is Alzheimer's disease diagnosed?",
    "What is the role of the liver in metabolism?",
    "What are autoimmune diseases?",
    "How does vaccination work?",
    "What is a neural network?",
    # variations / longer queries
    "Can you explain the mechanism by which statins reduce cholesterol levels?",
    "What distinguishes supervised learning from unsupervised learning in ML?",
    "What are the key components of a large language model architecture?",
    "How does sentence-transformers encode semantic meaning into vectors?",
    "What is the difference between HNSW and flat indexing in vector databases?",
    "Describe the end-to-end pipeline of a document question-answering system.",
    "What safety measures are required when deploying medical AI systems?",
    "How does AWS S3 integrate with a document storage pipeline?",
    "What are the benefits of using a conversational memory buffer in chatbots?",
    "How do you evaluate the quality of answers in a RAG system?",
    # short queries
    "fever treatment",
    "LangChain retriever",
    "cancer symptoms",
    "embedding model",
    "Flask REST API",
    "blood pressure",
    "neural network layers",
    "resume skills",
    "kidney disease",
    "machine learning",
    # more medical
    "What is sepsis?",
    "How is epilepsy managed?",
    "What is the prognosis for stage 3 cancer?",
    "What medications treat high cholesterol?",
    "What is the function of white blood cells?",
    "How does dialysis work?",
    "What is multiple sclerosis?",
    "What are the symptoms of liver failure?",
    "How is anxiety disorder treated?",
    "What causes kidney stones?",
    # more technical
    "What is cosine similarity?",
    "How does BERT differ from GPT?",
    "What is semantic search?",
    "How does in-context learning work?",
    "What is a language model prompt?",
    "How do you chunk documents for RAG?",
    "What is the chunk overlap strategy?",
    "How does recursive text splitter work?",
    "What is a vector store?",
    "What is Chroma?",
    # final 10
    "What is temperature in LLM inference?",
    "How does greedy decoding differ from sampling?",
    "What is beam search?",
    "What are tokens in NLP?",
    "How does tokenization affect model performance?",
    "What is the context window of an LLM?",
    "What is instruction tuning?",
    "What is zero-shot prompting?",
    "What is chain-of-thought prompting?",
    "How does few-shot prompting work?",
]

assert len(QUERIES) == 100, f"Expected 100 queries, got {len(QUERIES)}"


def main():
    print("Loading vector store...")
    vs = VectorStore(Config.VECTOR_DB_PATH)

    chunk_count = vs.vector_store._collection.count()
    print(f"Indexed chunks : {chunk_count:,}")
    print(f"Running {len(QUERIES)} retrieval queries (k=4)...\n")

    latencies_ms = []

    for i, query in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        results = vs.similarity_search(query, k=4)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        latencies_ms.append(ms)
        if i % 20 == 0:
            print(f"  {i}/100 done  (last: {ms:.1f}ms)")

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    mean = statistics.mean(latencies_ms)
    mn   = min(latencies_ms)
    mx   = max(latencies_ms)

    print("\n" + "="*50)
    print("RETRIEVAL LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"Hardware    : Intel Core i5-1135G7 @ 2.40GHz (laptop CPU)")
    print(f"Embedding   : sentence-transformers/all-MiniLM-L6-v2")
    print(f"Vector DB   : ChromaDB (local, persistent)")
    print(f"Chunks      : {chunk_count:,}")
    print(f"Queries run : {len(QUERIES)}")
    print(f"k (top-k)   : 4")
    print()
    print(f"  P50  : {p50:.1f} ms")
    print(f"  P95  : {p95:.1f} ms")
    print(f"  P99  : {p99:.1f} ms")
    print(f"  Mean : {mean:.1f} ms")
    print(f"  Min  : {mn:.1f} ms")
    print(f"  Max  : {mx:.1f} ms")
    print("="*50)

    # Write raw results for the README
    with open(os.path.join(os.path.dirname(__file__), 'retrieval_results.txt'), 'w') as f:
        f.write(f"Indexed chunks: {chunk_count}\n")
        f.write(f"Queries: {len(QUERIES)}\n")
        f.write(f"P50:  {p50:.1f} ms\n")
        f.write(f"P95:  {p95:.1f} ms\n")
        f.write(f"P99:  {p99:.1f} ms\n")
        f.write(f"Mean: {mean:.1f} ms\n")
        f.write(f"Min:  {mn:.1f} ms\n")
        f.write(f"Max:  {mx:.1f} ms\n")
        f.write("\nRaw latencies (ms):\n")
        for lat in sorted(latencies_ms):
            f.write(f"  {lat:.2f}\n")

    print(f"\nRaw results saved to benchmarks/retrieval_results.txt")


if __name__ == '__main__':
    main()
