import os
import json
import shutil
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.componenets.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DB_FAISS_PATH, UPLOAD_FOLDER

logger = get_logger(__name__)

# The Flask dev server handles requests in threads, so a user clicking "Delete"
# repeatedly used to start several concurrent FAISS mutations at once (index
# corruption + every CPU core busy). All document mutations are serialised here.
_DOC_MUTATION_LOCK = threading.Lock()

DOCS_METADATA_FILE = os.path.join(os.path.dirname(DB_FAISS_PATH), "uploaded_docs.json")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_docs_registry() -> List[Dict[str, Any]]:
    """Load registry of uploaded documents."""
    try:
        if os.path.exists(DOCS_METADATA_FILE):
            with open(DOCS_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read docs registry: {e}")
    return []


def _save_docs_registry(registry: List[Dict[str, Any]]):
    """Save registry of uploaded documents."""
    try:
        os.makedirs(os.path.dirname(DOCS_METADATA_FILE), exist_ok=True)
        with open(DOCS_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write docs registry: {e}")


def get_uploaded_documents_info() -> List[Dict[str, Any]]:
    """Return summary list of all processed documents."""
    return _load_docs_registry()


def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    """Look up a registry entry by its document_id."""
    for doc in _load_docs_registry():
        if doc.get("document_id") == document_id:
            return doc
    return None


def find_duplicate_by_hash(content_hash: str) -> Optional[Dict[str, Any]]:
    """Return an existing registry entry with the same content hash, if any."""
    if not content_hash:
        return None
    for doc in _load_docs_registry():
        if doc.get("content_hash") and doc.get("content_hash") == content_hash:
            return doc
    return None


def register_document_info(
    filename: str,
    file_type: str,
    pages: Any,
    chunks_count: int,
    document_id: Optional[str] = None,
    file_path: Optional[str] = None,
    content_hash: Optional[str] = None,
    upload_time: Optional[str] = None,
):
    """Record document processing metadata including document_id, storage path and hash."""
    registry = _load_docs_registry()
    # Check if document already exists in registry, update it; otherwise append
    found = False
    for doc in registry:
        if document_id and doc.get("document_id") == document_id:
            doc["filename"] = filename
            doc["pages"] = pages
            doc["chunks"] = chunks_count
            doc["page_count"] = pages
            doc["chunk_count"] = chunks_count
            doc["file_type"] = file_type
            doc["status"] = "Ready"
            doc["file_path"] = file_path or doc.get("file_path")
            doc["content_hash"] = content_hash or doc.get("content_hash")
            doc["upload_time"] = upload_time or doc.get("upload_time") or _now()
            found = True
            break
    if not found:
        registry.append({
            "filename": filename,
            "file_type": file_type,
            "pages": pages,
            "chunks": chunks_count,
            "page_count": pages,
            "chunk_count": chunks_count,
            "document_id": document_id or filename,
            "file_path": file_path or "",
            "content_hash": content_hash or "",
            "upload_time": upload_time or _now(),
            "status": "Ready"
        })
    _save_docs_registry(registry)


def remove_document_from_registry(document_id: str) -> bool:
    """Remove one registry entry by document_id. Returns True if removed."""
    registry = _load_docs_registry()
    new_registry = [d for d in registry if d.get("document_id") != document_id]
    if len(new_registry) == len(registry):
        return False
    _save_docs_registry(new_registry)
    return True


def load_vector_store() -> Optional[FAISS]:
    """Load existing FAISS vector store from disk."""
    try:
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise CustomException("Failed to initialize embedding model.")

        index_file = os.path.join(DB_FAISS_PATH, "index.faiss")
        if os.path.exists(DB_FAISS_PATH) and os.path.exists(index_file):
            logger.info(f"Loading existing vector database from '{DB_FAISS_PATH}'...")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning(f"No vector database found at '{DB_FAISS_PATH}'.")
            return None

    except Exception as e:
        error_msg = CustomException("Failed to load vector store from disk.", e)
        logger.error(str(error_msg))
        return None


def save_vector_store(text_chunks: List[Document]) -> Optional[FAISS]:
    """Create a new FAISS vector store from text chunks and save to disk."""
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save.")

        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise CustomException("Failed to initialize embedding model.")

        logger.info(f"Creating new vector store with {len(text_chunks)} chunks...")
        db = FAISS.from_documents(text_chunks, embedding_model)
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        db.save_local(DB_FAISS_PATH)
        logger.info(f"Vector store saved successfully at '{DB_FAISS_PATH}'.")
        return db

    except Exception as e:
        error_msg = CustomException("Failed to create and save vector store.", e)
        logger.error(str(error_msg))
        raise error_msg


def add_documents_to_vector_store(text_chunks: List[Document]) -> FAISS:
    """
    Add new document chunks to the existing vector store, or create one if none exists.
    """
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to add.")

        db = load_vector_store()
        if db is not None:
            logger.info(f"Adding {len(text_chunks)} chunks to existing vector store...")
            db.add_documents(text_chunks)
            os.makedirs(DB_FAISS_PATH, exist_ok=True)
            db.save_local(DB_FAISS_PATH)
            logger.info("Updated vector store saved successfully to disk.")
            return db
        else:
            logger.info("No existing vector store found; creating initial store...")
            return save_vector_store(text_chunks)

    except Exception as e:
        error_msg = CustomException("Failed to add documents to vector store.", e)
        logger.error(str(error_msg))
        raise error_msg


def reset_vector_store():
    """Remove vector store and document registry."""
    try:
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH, ignore_errors=True)
        if os.path.exists(DOCS_METADATA_FILE):
            os.remove(DOCS_METADATA_FILE)
        logger.info("Vector database and document registry reset.")
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")


# ============================================================
# Document deletion (id-based vector removal, no re-embedding)
# ============================================================

def _load_all_indexed_documents(db: FAISS) -> List[Document]:
    """Return every Document currently stored inside the FAISS docstore."""
    docs: List[Document] = []
    try:
        # Preferred: direct read of the in-memory docstore
        docs = list(db.docstore._dict.values())
    except Exception:
        try:
            # Fallback: walk the vector-id -> docstore-id mapping
            for docstore_id in db.index_to_docstore_id.values():
                try:
                    doc = db.docstore.search(docstore_id)
                    if doc is not None:
                        docs.append(doc)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Could not enumerate FAISS docstore: {e}")
    return docs


def _chunk_belongs_to(doc: Document, document_id: str, filename: str = "") -> bool:
    """Match a chunk to a document: strict by document_id when present,
    filename fallback only for legacy chunks that have no document_id."""
    md = doc.metadata or {}
    md_id = md.get("document_id")
    if md_id:
        return md_id == document_id
    return bool(filename) and md.get("source") == filename


def _collect_document_vector_ids(db: FAISS, document_id: str, filename: str = "") -> Dict[str, Any]:
    """Split the indexed chunks into "belongs to this document" and "keep".

    Returns {"remove_ids": [...], "kept_chunks": [...], "total": int}, where
    remove_ids are the vector-store ids that FAISS.delete() expects.
    """
    remove_ids: List[str] = []
    kept_chunks: List[Document] = []
    total = 0

    for docstore_id in list(db.index_to_docstore_id.values()):
        try:
            doc = db.docstore.search(docstore_id)
        except Exception:
            continue
        if doc is None:
            continue
        total += 1
        if _chunk_belongs_to(doc, document_id, filename):
            remove_ids.append(docstore_id)
        else:
            kept_chunks.append(doc)

    return {"remove_ids": remove_ids, "kept_chunks": kept_chunks, "total": total}


def _delete_vectors_by_id(db: FAISS, docstore_ids: List[str]) -> bool:
    """Remove vectors by id (fast: no re-embedding). False => caller must rebuild."""
    try:
        deleted = db.delete(list(docstore_ids))
        logger.info(
            f"FAISS id-based delete done ({len(docstore_ids)} chunk(s), no re-embedding)."
        )
        return bool(deleted)
    except Exception as e:
        logger.warning(f"Id-based FAISS deletion unavailable ({e}); falling back to rebuild.")
        return False


def _rebuild_index_without(document_id: str, filename: str) -> int:
    """Last-resort slow path when id-based deletion is unsupported.

    Re-embeds every remaining chunk (minutes on large indexes), so it is only
    reached if the loaded index type cannot remove vectors by id.
    """
    db = load_vector_store()
    if db is None:
        return 0

    kept_chunks = [
        d for d in _load_all_indexed_documents(db)
        if not _chunk_belongs_to(d, document_id, filename)
    ]
    logger.info(f"Rebuilding FAISS from {len(kept_chunks)} remaining chunk(s)...")

    if kept_chunks:
        new_db = FAISS.from_documents(kept_chunks, get_embedding_model())
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        new_db.save_local(DB_FAISS_PATH)
        return len(kept_chunks)

    # Nothing left: remove the index entirely so it cannot serve stale vectors
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH, ignore_errors=True)
    return 0


def remove_document_from_vector_store(document_id: str, filename: str = "") -> Dict[str, Any]:
    """
    Fully remove a document's vectors from FAISS.

    Strategy (fast path first):
        1. Load the current index
        2. Map every stored chunk to its vector-store id
        3. Delete the document's vectors by id (seconds, no re-embedding)
        4. Only when unsupported: rebuild the index from the remaining chunks
        5. Persist and verify the document can no longer be retrieved

    Returns {"removed_chunks": int, "remaining_chunks": int}.
    """
    with _DOC_MUTATION_LOCK:
        db = load_vector_store()
        if db is None:
            logger.info("No vector store present; nothing to remove from FAISS.")
            return {"removed_chunks": 0, "remaining_chunks": 0}

        summary = _collect_document_vector_ids(db, document_id, filename)
        remove_ids = summary["remove_ids"]
        kept_chunks = summary["kept_chunks"]
        total = summary["total"]

        if not total:
            logger.warning("Vector store exists but contains no readable chunks.")
            return {"removed_chunks": 0, "remaining_chunks": 0}

        logger.info(
            f"Removing {len(remove_ids)} chunk(s) of '{document_id}' from FAISS "
            f"({total} indexed, {len(kept_chunks)} kept)..."
        )

        if remove_ids:
            if _delete_vectors_by_id(db, remove_ids):
                if db.index is not None and db.index.ntotal > 0:
                    os.makedirs(DB_FAISS_PATH, exist_ok=True)
                    db.save_local(DB_FAISS_PATH)
                elif os.path.exists(DB_FAISS_PATH):
                    # Nothing left: drop the index so it cannot serve stale vectors
                    shutil.rmtree(DB_FAISS_PATH, ignore_errors=True)
            else:
                _rebuild_index_without(document_id, filename)

        # ---- Verification: deleted document must NOT be retrievable ----
        verify_db = load_vector_store()
        if verify_db is not None:
            remaining_docs = _load_all_indexed_documents(verify_db)
            leftover = sum(1 for d in remaining_docs if _chunk_belongs_to(d, document_id, filename))
            if leftover:
                raise CustomException(
                    f"Deletion verification failed: {leftover} chunk(s) of '{document_id}' still present in FAISS."
                )
            logger.info(f"Verified: no vectors remain for document '{document_id}'.")
            remaining = len(remaining_docs)
        else:
            remaining = 0

        return {"removed_chunks": len(remove_ids), "remaining_chunks": remaining}


def delete_physical_file(file_path: str) -> bool:
    """
    Safely delete the uploaded file. Only paths inside UPLOAD_FOLDER are allowed,
    so a manipulated document record can never delete arbitrary server files.
    """
    if not file_path:
        return False
    try:
        upload_root = os.path.abspath(UPLOAD_FOLDER)
        target = os.path.abspath(file_path)
        # Must live inside the uploads folder (no path traversal)
        if not target.startswith(upload_root + os.sep):
            logger.warning(f"Refusing to delete file outside upload folder: {target}")
            return False
        if os.path.exists(target):
            os.remove(target)
            logger.info(f"Deleted physical file: {target}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete physical file '{file_path}': {e}")
    return False


def delete_document(document_id: str) -> Dict[str, Any]:
    """
    Complete document deletion pipeline.

    Order matters: the registry entry is removed FIRST because the registry is
    the single source of truth for both the sidebar listing and retrieval
    filtering (see retriver.query_rag). That makes the document disappear
    immediately and guarantees it can never be cited again, even if the
    physical-file or vector cleanup fails afterwards.
    """
    doc = get_document_by_id(document_id)
    if doc is None:
        raise CustomException(f"Document '{document_id}' not found.")

    filename = doc.get("filename", "")

    # 1) Remove metadata entry (UI + retrieval source of truth)
    remove_document_from_registry(document_id)

    # 2) Resolve and delete the physical file (validated against uploads root)
    file_path = doc.get("file_path") or ""
    if not file_path and filename:
        # Legacy entries without stored path: try safe join with uploads folder
        try:
            from werkzeug.utils import secure_filename
            candidate = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
            if os.path.exists(candidate):
                file_path = candidate
        except Exception:
            pass
    delete_physical_file(file_path)

    # 3) Remove vectors. Never fatal: the document is already unreachable, and a
    #    leftover vector must never resurrect it in the UI.
    result = {"removed_chunks": 0, "remaining_chunks": 0}
    try:
        result = remove_document_from_vector_store(document_id, filename=filename)
    except Exception as e:
        logger.error(
            f"Vector cleanup for '{document_id}' failed (metadata already removed): {e}"
        )

    logger.info(
        f"Document '{document_id}' ({filename}) deleted: "
        f"{result['removed_chunks']} vector(s) removed, {result['remaining_chunks']} remaining."
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "removed_chunks": result["removed_chunks"],
        "remaining_chunks": result["remaining_chunks"],
    }