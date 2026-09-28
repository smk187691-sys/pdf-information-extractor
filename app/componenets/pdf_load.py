import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DATA_PATH
from app.componenets.document_loader import create_text_chunks, load_single_document

logger = get_logger(__name__)


def load_pdf_files() -> List[Document]:
    """Load existing PDF files from DATA_PATH directory (backwards compatibility)."""
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path '{DATA_PATH}' does not exist.")
        logger.info(f"Loading PDF files from {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logger.warning(f"No PDF files found in {DATA_PATH}")
        else:
            logger.info(f"Successfully fetched {len(documents)} document pages.")
        return documents
    except Exception as e:
        error_msg = CustomException("Failed to load PDF files from directory", e)
        logger.error(str(error_msg))
        return []


__all__ = ["load_pdf_files", "create_text_chunks", "load_single_document"]