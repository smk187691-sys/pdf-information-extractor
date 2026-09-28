import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1000"))

# Embeddings & Vector Store Configuration
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = os.environ.get("VECTOR_DB_PATH", os.environ.get("DB_FAISS_PATH", "vector_db/db_faiss"))
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
DATA_PATH = os.environ.get("DATA_PATH", "data/")

# Chunking & Retrieval Parameters
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("TOP_K", "4"))

# Upload Constraints
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit