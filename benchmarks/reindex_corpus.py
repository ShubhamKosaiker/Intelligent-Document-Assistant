"""
Re-index ChromaDB with a curated arXiv corpus on RAG / LLM evaluation.

What this does:
  1. Downloads 22 arXiv PDFs (RAG, LLM eval, agent benchmarks) into data/arxiv/
  2. Clears the existing ChromaDB
  3. Ingests each PDF with file-hash deduplication (skips if hash already seen)
  4. Prints final chunk count

Run from TEXT_ASSISTANT/ root:
    myenv/Scripts/python benchmarks/reindex_corpus.py
"""

import os
import sys
import hashlib
import shutil
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join('app', '.env'))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.vector_store import VectorStore
from config import Config

# 22 arXiv papers on RAG, LLM evaluation, and agent benchmarks.
# Each entry: (arxiv_id, short_name_for_file)
ARXIV_PAPERS = [
    ("2005.11401", "rag_original_lewis"),                 # RAG (Lewis et al. 2020)
    ("2312.10997", "rag_survey_gao"),                     # RAG survey
    ("2309.01431", "ragas_es"),                           # RAGAS eval framework
    ("2307.03172", "lost_in_middle"),                     # Context window analysis
    ("2104.07567", "dpr_karpukhin"),                      # Dense Passage Retrieval
    ("2310.11511", "self_rag"),                           # Self-RAG
    ("2401.15884"  , "corrective_rag"),                   # Corrective RAG (CRAG)
    ("2404.10981", "survey_rag_eval"),                    # RAG eval survey
    ("2302.04761", "toolformer"),                         # Toolformer
    ("2210.03629", "react_yao"),                          # ReAct
    ("2308.03188", "agentbench"),                         # AgentBench
    ("2305.14325", "multi_agent_debate"),                 # Multi-agent debate
    ("2303.17580", "hugginggpt"),                         # HuggingGPT
    ("2203.02155", "instructgpt"),                        # InstructGPT / RLHF
    ("2201.11903", "chain_of_thought"),                   # Chain-of-thought
    ("2205.11916", "zero_shot_cot"),                      # Zero-shot CoT
    ("2305.10601", "tree_of_thoughts"),                   # Tree of Thoughts
    ("2110.08207", "t0_multitask"),                       # T0 multitask
    ("2009.03300", "mmlu"),                               # MMLU benchmark
    ("2211.09527", "helm_holistic_eval"),                 # HELM
    ("2307.13702", "reasoning_eval"),                     # Reasoning evaluation
    ("2311.12983", "agent_benchmark_survey"),             # Agent benchmark survey
]

DATA_DIR = os.path.join('data', 'arxiv')
os.makedirs(DATA_DIR, exist_ok=True)


def download_paper(arxiv_id, name):
    """Download a paper PDF from arxiv.org. Skip if already on disk."""
    path = os.path.join(DATA_DIR, f"{name}__{arxiv_id}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(path, 'wb') as f:
            f.write(r.read())
        return path
    except Exception as e:
        print(f"    ! failed to download {arxiv_id}: {e}")
        if os.path.exists(path):
            os.remove(path)
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("RE-INDEX CORPUS — arXiv RAG / LLM evaluation papers")
    print("=" * 60)

    # 1) Download
    print("\n[1/3] Downloading arXiv papers...")
    paths = []
    for arxiv_id, name in ARXIV_PAPERS:
        print(f"    - {arxiv_id} ({name})")
        p = download_paper(arxiv_id, name)
        if p:
            paths.append(p)
        time.sleep(1)  # polite delay
    print(f"    Downloaded {len(paths)}/{len(ARXIV_PAPERS)} papers")

    # 2) Clear old ChromaDB
    print("\n[2/3] Clearing existing vector_db/ ...")
    if os.path.exists(Config.VECTOR_DB_PATH):
        shutil.rmtree(Config.VECTOR_DB_PATH)
    os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)

    # 3) Ingest with dedup
    print("\n[3/3] Ingesting documents (file-hash dedup)...")
    vs = VectorStore(Config.VECTOR_DB_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    seen_hashes = set()
    total_chunks = 0
    ingested = 0
    skipped_dupe = 0

    for path in paths:
        digest = file_sha256(path)
        if digest in seen_hashes:
            skipped_dupe += 1
            print(f"    SKIP (duplicate hash): {os.path.basename(path)}")
            continue
        seen_hashes.add(digest)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            # Tag chunks with source basename for traceability
            for c in chunks:
                c.metadata['source'] = os.path.basename(path)
                c.metadata['sha256'] = digest
            vs.add_documents(chunks)
            total_chunks += len(chunks)
            ingested += 1
            print(f"    OK: {os.path.basename(path):55s} {len(chunks):4d} chunks")
        except Exception as e:
            print(f"    ! {os.path.basename(path)}: {e}")

    final_count = vs.vector_store._collection.count()
    print("\n" + "=" * 60)
    print("RE-INDEX COMPLETE")
    print("=" * 60)
    print(f"Papers ingested      : {ingested}")
    print(f"Duplicates skipped   : {skipped_dupe}")
    print(f"Total chunks indexed : {final_count}")
    print(f"Unique sources       : {len(seen_hashes)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
