import os
from typing import Optional
from langchain_community.vectorstores import FAISS

from app.componenets.document_loader import load_single_document, create_text_chunks
from app.componenets.vector_store import (
    load_vector_store,
    add_documents_to_vector_store,
    register_document_info,
    DB_FAISS_PATH
)
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DATA_PATH, UPLOAD_FOLDER

logger = get_logger(__name__)


def inject_data() -> Optional[FAISS]:
    """Scan data directory and upload directory for existing files to build/load vector store."""
    try:
        logger.info("Checking for existing vector store...")
        existing_db = load_vector_store()
        if existing_db is not None:
            logger.info("Existing vector store found.")
            return existing_db

        logger.info("No vector store found. Scanning document directories to initialize store...")
        all_chunks = []

        for folder in [DATA_PATH, UPLOAD_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".pdf", ".docx"]:
                        file_path = os.path.join(folder, filename)
                        try:
                            docs = load_single_document(file_path, original_filename=filename)
                            chunks = create_text_chunks(docs)
                            all_chunks.extend(chunks)
                            pages_count = len(docs) if ext == ".pdf" else "N/A"
                            register_document_info(filename, ext.lstrip("."), pages_count, len(chunks))
                        except Exception as e:
                            logger.warning(f"Could not load initial document '{filename}': {e}")

        if all_chunks:
            logger.info(f"Indexing {len(all_chunks)} chunks into new vector store...")
            return add_documents_to_vector_store(all_chunks)
        else:
            logger.info("No initial documents found to index.")
            return None

    except Exception as e:
        error_msg = CustomException("Failed to initialize database.", e)
        logger.error(str(error_msg))
        return None


if __name__ == "__main__":
    inject_data()