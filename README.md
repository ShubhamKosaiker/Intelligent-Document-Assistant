# 🧠 Intelligent Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) application that lets users upload documents and ask questions about them in natural language.

Built with **Flask**, **ChromaDB**, **HuggingFace embeddings**, **Groq LLaMA 3.3 70B**, and **AWS S3**. The project also includes benchmark scripts for retrieval latency, end-to-end latency, and RAGAS-based answer quality evaluation.

## 🚀 Demo

Upload a PDF or TXT file → Ask questions → Get context-aware answers powered by LLaMA 3.3 70B.

## 🏗️ Architecture

```text
User → Flask API → PDF/TXT Processing → HuggingFace Embeddings
                                      → ChromaDB Vector Store
                                      → Groq LLaMA 3.3 70B
                                      → AWS S3 File Storage
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq LLaMA 3.3 70B Versatile |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Backend | Flask |
| Document Processing | LangChain |
| File Storage | AWS S3 |
| Evaluation | RAGAS |
| Benchmarking | Custom Python scripts |

## ✨ Features

- Upload PDF and TXT documents
- Ask natural-language questions over uploaded files
- Conversational Q&A with memory across turns
- Local embeddings to avoid OpenAI embedding costs
- ChromaDB-based semantic retrieval
- AWS S3 integration for document storage
- Fast inference using Groq API
- Retrieval latency benchmarking
- End-to-end latency benchmarking
- RAGAS-based answer quality evaluation

## 📊 Benchmarks

This project includes reproducible benchmark scripts for retrieval latency, end-to-end query latency, and RAGAS-based answer quality evaluation.

### Retrieval Performance

| Area | Result |
|------|--------|
| Retrieval latency | P50: 13.7 ms, P95: 19.0 ms |
| Indexed corpus | 22 arXiv papers |
| Total chunks | 2,968 |

### RAGAS Answer Quality Evaluation

15 hand-labeled Q/A pairs were evaluated using the live RAG pipeline.

| Metric | Score | Meaning |
|--------|------:|---------|
| Faithfulness | 0.625 | Measures whether answers stay grounded in retrieved context |
| Answer Relevancy | 0.477 | Measures whether answers directly address the question |
| Context Precision | 0.861 | Measures whether retrieved chunks are relevant |
| Context Recall | 0.750 | Measures whether retrieval surfaces enough information |

### Key Finding

The retrieval layer performed strongly, with high context precision and solid context recall.

The weaker area was answer generation. Answer relevancy was below 0.5, showing that the next improvement should focus on prompt design and response constraints rather than retrieval alone.

See [`benchmarks/`](benchmarks/) for scripts, raw results, and methodology.

## 🧪 Evaluation Setup

| Component | Value |
|----------|-------|
| Generator LLM | llama-3.3-70b-versatile via Groq |
| Judge LLM | llama-3.1-8b-instant via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Retriever | ChromaDB, top-k=4 |
| Evaluation Framework | RAGAS |
| Evaluation Dataset | 15 hand-labeled Q/A pairs |

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/ShubhamKosaiker/Intelligent-Document-Assistant.git
cd Intelligent-Document-Assistant
```

### 2. Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` file

Create an `.env` file inside the `app/` directory:

```env
GROQ_API_KEY=your-groq-api-key
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
AWS_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=your-bucket-region
```

### 5. Run the application

```bash
python app/main.py
```

Open:

```text
http://127.0.0.1:8080
```

## 📁 Project Structure

```text
app/
├── main.py                    # Flask routes
├── config.py                  # Configuration
├── models/
│   └── vector_store.py        # ChromaDB vector store
├── services/
│   ├── llm_service.py         # Groq LLM + RAG chain
│   └── storage_service.py     # AWS S3 integration
├── static/
│   └── style.css
└── templates/
    └── index.html

benchmarks/
├── README.md
├── retrieval_benchmark.py
├── retrieval_results.txt
├── e2e_benchmark.py
├── e2e_results.txt
├── e2e_results_pre_fix.txt
├── e2e_results_post_fix.txt
├── quality_eval.py
├── quality_eval_dataset.py
├── quality_eval_results.csv
└── quality_eval_summary.txt
```

## 🔍 Benchmark Scripts

Run retrieval latency benchmark:

```bash
python benchmarks/retrieval_benchmark.py
```

Run end-to-end latency benchmark:

```bash
python benchmarks/e2e_benchmark.py
```

Run RAGAS quality evaluation:

```bash
python benchmarks/quality_eval.py
```

Note: RAGAS evaluation uses LLM-as-judge calls and may consume Groq API tokens.

## 🔐 API Keys

You will need:

- Groq API key: https://console.groq.com
- AWS credentials for S3 storage: https://aws.amazon.com/free

Keep secrets in `app/.env`. Do not commit API keys.

## 📌 Why This Project Matters

Most RAG demos stop at “it answers questions.”

This project goes further by measuring:

- How fast retrieval is
- How latency changes across conversations
- Whether retrieved chunks are relevant
- Whether generated answers are grounded
- Where the system actually needs improvement

The benchmark results showed that retrieval was relatively strong, but answer generation needed tighter prompting and better response constraints.

## 📄 License

MIT
