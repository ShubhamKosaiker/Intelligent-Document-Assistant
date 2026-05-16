"""
RAGAS quality evaluation for the Document Assistant.

Measures four reference-based metrics across 15 hand-labelled Q/A pairs:
  - faithfulness        : is the answer grounded in retrieved context?
  - answer_relevancy    : does the answer address the question?
  - context_precision   : was the retrieved context on-topic?
  - context_recall      : did retrieval surface everything needed?

Uses our existing Groq LLM as the judge and our HuggingFace embeddings —
no external API keys beyond what the app already needs.

Run:
    myenv/Scripts/python benchmarks/quality_eval.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join('app', '.env'))

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from models.vector_store import VectorStore
from services.llm_service import LLMService
from config import Config
from quality_eval_dataset import EVAL_SET


def main():
    print("=" * 60)
    print("RAGAS QUALITY EVALUATION")
    print("=" * 60)

    # 1. Load the live RAG system
    print("\n[1/4] Loading vector store and LLM service...")
    vs = VectorStore(Config.VECTOR_DB_PATH)
    llm_service = LLMService(vs)
    print(f"    Indexed chunks: {vs.vector_store._collection.count()}")

    # 2. Run every eval question through the real pipeline
    print(f"\n[2/4] Generating answers for {len(EVAL_SET)} questions...")
    questions, answers, contexts, references = [], [], [], []

    for i, item in enumerate(EVAL_SET, 1):
        q = item["question"]
        print(f"    [{i:02d}/{len(EVAL_SET)}] {q[:60]}")
        try:
            # Get top-k chunks (same retriever the app uses)
            docs = vs.vector_store.similarity_search(q, k=4)
            ctx = [d.page_content for d in docs]
            # Get the live generated answer
            ans = llm_service.get_response(q)
            if "encountered an error" in ans.lower():
                print(f"         ! rate-limited, skipping this Q")
                continue
            questions.append(q)
            answers.append(ans)
            contexts.append(ctx)
            references.append(item["reference"])
            time.sleep(2)  # pace to avoid Groq TPM cap
        except Exception as e:
            print(f"         ! error: {e}")

    if not questions:
        print("\nNo successful generations — Groq quota exhausted. "
              "Re-run after 24h quota reset.")
        return

    print(f"    Got {len(questions)} usable (Q, answer, context, ref) rows")

    # 3. Configure RAGAS to use Groq as judge + local HF embeddings
    print("\n[3/4] Configuring RAGAS (Groq as judge, HF embeddings)...")
    groq_client = OpenAI(
        api_key=Config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    judge_llm = llm_factory("llama-3.1-8b-instant", client=groq_client)
    judge_embed = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Run RAGAS
    print("\n[4/4] Running RAGAS metrics (LLM-as-judge per question)...")
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness(), answer_relevancy(),
                 context_precision(), context_recall()],
        llm=judge_llm,
        embeddings=judge_embed,
    )

    # 5. Report
    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    scores = result._repr_dict if hasattr(result, '_repr_dict') else dict(result)
    for k, v in scores.items():
        try:
            print(f"  {k:22s}: {float(v):.3f}")
        except (TypeError, ValueError):
            print(f"  {k:22s}: {v}")
    print("=" * 60)

    # Save raw results
    df = result.to_pandas()
    out = os.path.join("benchmarks", "quality_eval_results.csv")
    df.to_csv(out, index=False)
    print(f"\nPer-question scores saved to {out}")

    # Compact summary for README
    summary_out = os.path.join("benchmarks", "quality_eval_summary.txt")
    with open(summary_out, 'w') as f:
        f.write("RAGAS RESULTS\n")
        f.write(f"Questions evaluated: {len(questions)}\n")
        f.write(f"Judge LLM: llama-3.1-8b-instant (Groq) — generator: llama-3.3-70b-versatile\n")
        f.write(f"Embeddings: sentence-transformers/all-MiniLM-L6-v2\n\n")
        for k, v in scores.items():
            try:
                f.write(f"{k:22s}: {float(v):.3f}\n")
            except (TypeError, ValueError):
                f.write(f"{k:22s}: {v}\n")
    print(f"Summary saved to {summary_out}")


if __name__ == "__main__":
    main()
