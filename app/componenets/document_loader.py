import os
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip null/control characters from extracted text."""
    if not text:
        return ""
    # Remove null characters
    text = text.replace("\x00", "")
    # Replace multiple empty lines with maximum 2 newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Replace weird space sequences with standard spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


import hashlib
import uuid

def generate_document_id(filename: str = "") -> str:
    """Generate a unique document identifier like 'doc_8f31a92'.

    Random suffix ensures two different documents with the same filename
    never collide (filename alone is not reliable).
    """
    return "doc_" + uuid.uuid4().hex[:7]


def load_single_document(file_path: str, original_filename: Optional[str] = None, document_id: Optional[str] = None) -> List[Document]:
    """
    Load a PDF or DOCX file, extract text, clean it, and attach consistent metadata including document_id.
    """
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise CustomException("Uploaded file is empty (0 bytes).")

        display_name = original_filename if original_filename else os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        doc_id = document_id or generate_document_id(display_name)

        logger.info(f"Extracting content from '{display_name}' (type: {ext}, size: {file_size} bytes, id: {doc_id})")

        raw_docs: List[Document] = []

        if ext == ".pdf":
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    page_num = page.metadata.get("page", idx)
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": display_name,
                                "page": int(page_num) + 1,  # 1-indexed for display
                                "file_type": "pdf",
                                "document_id": doc_id
                            }
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse PDF document '{display_name}'. The file may be corrupted or password protected.", e)

        elif ext == ".docx":
            docx_text = ""
            # Try primary Docx2txtLoader
            try:
                loader = Docx2txtLoader(file_path)
                docx_docs = loader.load()
                for doc in docx_docs:
                    c = clean_text(doc.page_content)
                    if c:
                        docx_text += c + "\n\n"
            except Exception as e_docx2txt:
                logger.warning(f"Docx2txtLoader failed on '{display_name}', attempting python-docx fallback: {e_docx2txt}")

            # Fallback to python-docx if empty or failed
            if not docx_text.strip():
                try:
                    from docx import Document as DocxReader
                    doc_reader = DocxReader(file_path)
                    paragraphs = [p.text for p in doc_reader.paragraphs if p.text and p.text.strip()]
                    for table in doc_reader.tables:
                        for row in table.rows:
                            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                            if row_text:
                                paragraphs.append(row_text)
                    docx_text = "\n\n".join(paragraphs)
                except Exception as e_pydocx:
                    raise CustomException(f"Failed to parse DOCX document '{display_name}'. The file may be invalid or corrupted.", e_pydocx)

            cleaned = clean_text(docx_text)
            if cleaned:
                raw_docs.append(Document(
                    page_content=cleaned,
                    metadata={
                        "source": display_name,
                        "page": "N/A",  # DOCX does not have fixed pages
                        "file_type": "docx",
                        "document_id": doc_id
                    }
                ))

        else:
            raise CustomException(f"Unsupported file format '{ext}'. Only .pdf and .docx are supported.")

        if not raw_docs:
            raise CustomException(f"Document '{display_name}' contains no readable text content.")

        logger.info(f"Extracted {len(raw_docs)} document segment(s) from '{display_name}'.")
        return raw_docs

    except CustomException:
        raise
    except Exception as e:
        error_msg = CustomException(f"Error while loading document '{file_path}'", e)
        logger.error(str(error_msg))
        raise error_msg


def create_text_chunks(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split extracted documents into clean chunks while preserving metadata.
    """
    try:
        if not documents:
            raise CustomException("No documents provided to chunk.")

        c_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        c_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

        logger.info(f"Chunking {len(documents)} document pages/sections (chunk_size={c_size}, chunk_overlap={c_overlap})")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        # Filter out empty or whitespace-only chunks
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        # Assign a sequential chunk_id within each document (keeps FAISS/metadata aligned)
        counters: dict = {}
        for chunk in valid_chunks:
            d_id = str(chunk.metadata.get("document_id", "unknown"))
            counters[d_id] = counters.get(d_id, 0) + 1
            chunk.metadata["chunk_id"] = counters[d_id]

        logger.info(f"Created {len(valid_chunks)} text chunks successfully.")

        if not valid_chunks:
            raise CustomException("Chunking resulted in 0 valid chunks.")

        return valid_chunks

    except CustomException:
        raise
    except Exception as e:
        error_msg = CustomException("Failed to create text chunks from document", e)
        logger.error(str(error_msg))
        raise error_msg
