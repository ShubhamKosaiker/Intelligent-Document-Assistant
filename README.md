# 🧠 Intelligent Document Assistant

A production-ready RAG (Retrieval-Augmented Generation) application that lets you upload documents and ask questions about them in natural language. Built with Groq LLaMA, ChromaDB, HuggingFace embeddings, and AWS S3.

## 🚀 Demo
Upload any PDF or TXT file → Ask questions → Get context-aware answers powered by LLaMA 3.3 70B

## 🏗️ Architecture
```
User → Flask API → PDF/TXT Processing → HuggingFace Embeddings
                                      → ChromaDB Vector Store
                                      → Groq LLaMA 3.3 70B
                                      → AWS S3 (file storage)
```

## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| LLM | Groq LLaMA 3.3 70B Versatile |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | ChromaDB |
| File Storage | AWS S3 |
| Backend | Flask |
| Document Processing | LangChain |

## ✨ Features
- Upload PDF and TXT documents
- Conversational Q&A with memory across turns
- Local embeddings — no OpenAI cost for embeddings
- AWS S3 integration for document storage
- Fast inference via Groq API

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/ShubhamKosaiker/Intelligent-Document-Assistant.git
cd Intelligent-Document-Assistant
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```
GROQ_API_KEY=your-groq-api-key
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
AWS_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=your-bucket-region
```

### 5. Run
```bash
python app/main.py
```

Open `http://127.0.0.1:8080`

## 📁 Project Structure
```
app/
├── main.py              # Flask routes
├── config.py            # Configuration
├── models/
│   └── vector_store.py  # ChromaDB vector store
├── services/
│   ├── llm_service.py   # Groq LLM + RAG chain
│   └── storage_service.py # AWS S3
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## 🔑 Get Free API Keys
- **Groq**: https://console.groq.com (free)
- **AWS**: https://aws.amazon.com/free

## 📄 License
MIT