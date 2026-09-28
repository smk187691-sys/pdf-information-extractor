# 📑 Universal Document RAG Assistant

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-red.svg)](https://flask.palletsprojects.com/)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://python.langchain.com/)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![LLM Acceleration](https://img.shields.io/badge/Inference-Groq%20Cloud-orange.svg)](https://groq.com/)

A generic, high-performance **Document Retrieval-Augmented Generation (RAG)** application. The system enables users to upload **PDF** and **DOCX** documents across any domain—such as research papers, agriculture reports, technical manuals, machine learning notes, business plans, or legal contracts—automatically extracts, chunks, and embeds their content, and allows users to ask questions grounded strictly in the uploaded materials with exact document and page citations.

---

## 📌 Architecture & Pipeline

```text
                  ┌────────────────────────┐
                  │ Upload PDF / DOCX File │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Text Extraction Engine │
                  │ (PyPDFLoader/Docx2txt) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Document Chunking      │
                  │ (RecursiveSplitter)    │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Semantic Embeddings    │
                  │ (HuggingFace all-Mini) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Persistent FAISS Store │
                  │ (vector_db/db_faiss)   │
                  └───────────┬────────────┘
                              │
                              │ [User asks question]
                              ▼
                  ┌────────────────────────┐
                  │ Semantic Similarity    │
                  │ Search (Top-K Chunks)  │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Context & Grounding    │
                  │ System Prompt          │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Groq Cloud LLM         │
                  │ (Fast Inference)       │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Final Answer + Citations│
                  │ (Web UI & SQLite DB)   │
                  └────────────────────────┘
```

---

## ✨ Key Features

- **Generic Multi-Format Document Ingestion**: Upload any **PDF** or **DOCX** document. Seamlessly works for agriculture, science, engineering, business, law, notes, and research papers.
- **Robust Text Extraction & Fallbacks**: Utilizes `PyPDFLoader` for PDFs with 1-based page metadata, and `Docx2txtLoader` with automated `python-docx` fallback for complex DOCX tables and formatting.
- **Configurable Text Chunking**: Chunks text via `RecursiveCharacterTextSplitter` with customizable `CHUNK_SIZE` and `CHUNK_OVERLAP`.
- **HuggingFace Embeddings**: Uses dense 384-dimensional embeddings (`sentence-transformers/all-MiniLM-L6-v2`) cached in memory to eliminate redundant loads.
- **Persistent Multi-Document FAISS Index**: Incremental indexing preserves previously uploaded documents so embeddings persist across restarts.
- **Document Metadata Tracking**: Tracks `source`, `page`, `file_type`, and unique `document_id` for every chunk.
- **Strict Grounded Answering**: Employs prompt rules ensuring the LLM relies primarily on the context, prevents hallucinations, and states when information is not found.
- **Document & Page Citations**: Displays clickable/pill badges citing the exact document and page number for each retrieved answer.
- **Persistent Chat History**: SQLite database stores chat sessions with timestamps, question, answer, and sources.
- **Modern Responsive Web Interface**: Clean UI with real-time processing indicators, document management panel, and AJAX chat interaction.

---

## 🛠️ Tech Stack

- **Backend**: Flask 3.x, Python 3.10+
- **RAG & Orchestration**: LangChain, LangChain-Community, LangChain-Core
- **Embeddings**: LangChain-HuggingFace, `sentence-transformers`
- **Vector Database**: FAISS (`faiss-cpu`)
- **LLM Inference**: `langchain-groq` (Groq Cloud API)
- **Document Loaders**: `pypdf`, `python-docx`, `docx2txt`
- **Database**: SQLite3
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript (Fetch API)

---

## 🚀 Getting Started

### 1. Clone & Set Up Virtual Environment

```bash
# Create and activate virtual environment
python -m venv abc
# Windows:
abc\Scripts\activate
# Linux/macOS:
source abc/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r req.txt
pip install -e .
```

### 3. Environment Variables Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Groq API Configuration (or LLM_API_KEY)
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1000

# Embeddings & Vector Store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=vector_db/db_faiss
UPLOAD_FOLDER=uploads

# Chunking & Retrieval Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4

# Server Configuration
PORT=5000
HOST=0.0.0.0
```

### 4. Run the Application

```bash
python run.py
```

The application will start on: `http://localhost:5000`

---

## 🌐 API Endpoints

### 1. Document Upload
- **URL**: `POST /upload`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (PDF or DOCX file)
- **Response**:
```json
{
  "success": true,
  "filename": "agriculture.pdf",
  "file_type": "pdf",
  "pages": 45,
  "chunks": 132,
  "document_id": "agriculture_pdf_a1b2c3d4",
  "message": "Document processed successfully"
}
```

### 2. Chat / Question Answering
- **URL**: `POST /chat`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "question": "What is the main objective of this document?"
}
```
- **Response**:
```json
{
  "success": true,
  "question": "What is the main objective of this document?",
  "answer": "The main objective is...",
  "sources": [
    {
      "document": "agriculture.pdf",
      "page": 5,
      "label": "agriculture.pdf (Page 5)"
    }
  ]
}
```

### 3. List Indexed Documents
- **URL**: `GET /documents`
- **Response**:
```json
{
  "success": true,
  "documents": [
    {
      "filename": "agriculture.pdf",
      "file_type": "pdf",
      "pages": 45,
      "chunks": 132,
      "status": "Ready"
    }
  ]
}
```

### 4. Clear Documents / Reset Vector Store
- **URL**: `POST /reset-documents`

---

## 🧪 Testing the Application

### Sample Documents & Questions:
1. **Agriculture PDF (`agriculture_report.pdf`)**:
   - *Question*: "By how much does drip irrigation reduce evaporation?"
   - *Expected Answer*: "Drip irrigation reduces evaporation by 45 percent." (Source: `agriculture_report.pdf (Page 1)`)
2. **Machine Learning DOCX (`machine_learning_notes.docx`)**:
   - *Question*: "What causes overfitting and how can it be prevented?"
   - *Expected Answer*: "Overfitting occurs when a model learns training data noise... Prevented with regularization, dropout, early stopping." (Source: `machine_learning_notes.docx`)
3. **Out-of-Scope Query ("Information Not Found")**:
   - *Question*: "What is the capital of Mars and its temperature in 2050?"
   - *Expected Answer*: Clearly states the information was not found in the uploaded documents.
