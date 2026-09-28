import os
import json
import hashlib
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from markupsafe import Markup
from dotenv import load_dotenv

from app.componenets.retriver import query_rag
from app.componenets.document_loader import (
    load_single_document,
    create_text_chunks,
    generate_document_id
)
from app.componenets.vector_store import (
    add_documents_to_vector_store,
    get_uploaded_documents_info,
    register_document_info,
    reset_vector_store,
    delete_document,
    get_document_by_id,
    find_duplicate_by_hash,
    load_vector_store,
)
from app.componenets.history import (
    create_table,
    save_chat,
    get_history,
    delet_chat,
    clear_history,
    create_chat,
    get_chats,
    get_chat,
    get_messages,
    save_message,
    update_chat_documents,
    delete_conversation,
    clear_all_chats,
    generate_chat_title,
)
from app.config.config import (
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER
)
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

load_dotenv()
logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure required runtime folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
create_table()


def _startup_warmup():
    """
    Startup sequence: load document metadata, load FAISS index, chat DB ready.
    Must never crash the app when no documents exist yet.
    """
    try:
        docs = get_uploaded_documents_info()
        logger.info(f"Startup: loaded {len(docs)} document record(s) from registry.")
    except Exception as e:
        logger.warning(f"Startup: could not load document metadata: {e}")
    try:
        db = load_vector_store()
        if db is not None:
            logger.info("Startup: FAISS index loaded and ready.")
        else:
            logger.info("Startup: no FAISS index yet; one will be created on first upload.")
    except Exception as e:
        logger.warning(f"Startup: could not load FAISS index: {e}")


_startup_warmup()


def allowed_file(filename: str) -> bool:
    """Validate file extension against allowed types (.pdf, .docx)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def br_tag(value: str) -> Markup:
    """Format line breaks for HTML rendering."""
    if not value:
        return Markup("")
    # Basic markdown bold rendering if present, then line breaks
    formatted = Markup.escape(value)
    return Markup(formatted.replace("\n", "<br>\n"))


app.jinja_env.filters["br_tag"] = br_tag


def _format_sources(sources_raw):
    """Parse stored sources JSON into a list for template rendering."""
    if not sources_raw:
        return []
    if isinstance(sources_raw, list):
        return sources_raw
    try:
        return json.loads(sources_raw)
    except Exception:
        return [{"label": sources_raw}]


@app.route("/", methods=["GET", "POST"])
def index():
    """Main dashboard rendering chat interface and document controls."""
    if request.method == "POST":
        user_input = request.form.get("prompts", "").strip()
        if user_input:
            try:
                # Legacy form fallback: keep working via a session chat
                chat = create_chat(title=generate_chat_title(user_input))
                chat_id = chat["chat_id"]
                rag_result = query_rag(user_input)
                answer = rag_result.get("answer", "No response generated.")
                sources = rag_result.get("sources", [])
                save_message(chat_id, "user", user_input)
                save_message(chat_id, "assistant", answer, sources)
                used_ids = [s.get("document_id") for s in sources if s.get("document_id")]
                if used_ids:
                    update_chat_documents(chat_id, used_ids)
            except Exception as e:
                logger.error(f"Error handling user question: {e}")
                return render_template(
                    "index.html",
                    messages=[],
                    chats=get_chats(),
                    documents=get_uploaded_documents_info(),
                    error="An error occurred while processing your question. Please try again."
                )
        return redirect(url_for("index"))

    documents = get_uploaded_documents_info()
    chats = get_chats()
    return render_template(
        "index.html",
        messages=[],
        chats=chats,
        documents=documents
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle PDF and DOCX document uploads, text extraction, chunking, and vector embedding.
    """
    is_ajax = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.is_json
        or request.accept_mimetypes.best == "application/json"
    )

    if "file" not in request.files:
        error_msg = "No file part in the request."
        if is_ajax:
            return jsonify({"success": False, "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        error_msg = "No file selected."
        if is_ajax:
            return jsonify({"success": False, "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        error_msg = "Unsupported file type. Please upload a .pdf or .docx document."
        if is_ajax:
            return jsonify({"success": False, "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    try:
        original_name = file.filename
        safe_name = secure_filename(original_name)
        if not safe_name:
            safe_name = f"uploaded_doc.{original_name.rsplit('.', 1)[1].lower()}"

        # Read bytes once for hashing (duplicate detection), then persist
        file_bytes = file.read()
        if not file_bytes:
            error_msg = "Uploaded file is empty (0 bytes)."
            if is_ajax:
                return jsonify({"success": False, "message": error_msg}), 400
            flash(error_msg, "error")
            return redirect(url_for("index"))

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        duplicate = find_duplicate_by_hash(content_hash)
        if duplicate:
            dup_msg = (
                f"This file is already uploaded as '{duplicate.get('filename')}' "
                f"(ID: {duplicate.get('document_id')})."
            )
            if is_ajax:
                return jsonify({
                    "success": False,
                    "duplicate": True,
                    "message": dup_msg,
                    "document_id": duplicate.get("document_id")
                }), 400
            flash(dup_msg, "error")
            return redirect(url_for("index"))

        # Unique document_id + collision-proof stored filename
        doc_id = generate_document_id(original_name)
        stored_name = f"{doc_id}_{safe_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Saved uploaded file to '{save_path}' (document_id={doc_id})")

        # Extract text & pages
        docs = load_single_document(save_path, original_filename=original_name, document_id=doc_id)
        file_ext = original_name.rsplit(".", 1)[1].lower()
        pages_count = len(docs) if file_ext == "pdf" else "N/A"

        # Chunk document
        chunks = create_text_chunks(docs)

        # Generate embeddings & store in vector database
        add_documents_to_vector_store(chunks)

        # Record metadata
        register_document_info(
            filename=original_name,
            file_type=file_ext,
            pages=pages_count,
            chunks_count=len(chunks),
            document_id=doc_id,
            file_path=save_path,
            content_hash=content_hash
        )

        success_response = {
            "success": True,
            "filename": original_name,
            "file_type": file_ext,
            "pages": pages_count,
            "page_count": pages_count,
            "chunks": len(chunks),
            "chunk_count": len(chunks),
            "document_id": doc_id,
            "message": "Document processed successfully"
        }
        logger.info(f"Upload successful: {success_response}")

        if is_ajax:
            return jsonify(success_response)

        flash(f"Document '{original_name}' processed successfully ({len(chunks)} chunks).", "success")
        return redirect(url_for("index"))

    except Exception as e:
        logger.error(f"Document upload processing failed: {e}")
        # Friendly message only — strip internal CustomException details
        # (format: "message | Details: ... | File: ... | Line: ...")
        error_msg = str(e).split(" | ")[0].strip() or \
            "Failed to process the document. Please verify it is a valid PDF/DOCX and try again."
        if is_ajax:
            return jsonify({"success": False, "message": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
def chat_api():
    """
    Chat API: accepts JSON {question, chat_id?, document_ids?} (or form fallback).
    Creates/reuses a conversation session, runs RAG, persists both messages.
    """
    is_json = request.is_json
    if is_json:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or data.get("prompts") or "").strip()
        chat_id = (data.get("chat_id") or "").strip()
        document_ids = data.get("document_ids") or []
    else:
        question = (request.form.get("question") or request.form.get("prompts") or "").strip()
        chat_id = (request.form.get("chat_id") or "").strip()
        document_ids = []

    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty.",
                        "answer": "Please enter a question."}), 400

    if isinstance(document_ids, str):
        document_ids = [document_ids]

    try:
        # Resolve conversation session (create one for the first message)
        chat = get_chat(chat_id) if chat_id else None
        if chat is None:
            chat = create_chat(title=generate_chat_title(question),
                               document_ids=[d for d in document_ids if d])
            chat_id = chat["chat_id"]
            created_new = True
        else:
            created_new = False
            # First question sets the title if the chat is still untitled
            if chat.get("title") in ("", "New Chat"):
                from app.componenets.history import set_chat_title
                set_chat_title(chat_id, generate_chat_title(question))

        result = query_rag(question, document_ids=document_ids or None)
        answer = result.get("answer", "No answer found.")
        sources = result.get("sources", [])

        # Persist user + assistant messages in this conversation
        save_message(chat_id, "user", question)
        save_message(chat_id, "assistant", answer, sources)

        # Track which documents this conversation used (for later filtering)
        used_ids = [s.get("document_id") for s in sources if s.get("document_id")]
        if document_ids:
            used_ids = list(set(used_ids) | set(document_ids))
        if used_ids:
            update_chat_documents(chat_id, used_ids)

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "chat_id": chat_id,
            "chat_title": (get_chat(chat_id) or {}).get("title", ""),
            "new_chat": created_new
        })

    except Exception as e:
        logger.error(f"Chat API processing error: {e}")
        # Friendly message only — never expose stack traces / API keys
        return jsonify({
            "success": False,
            "answer": "An error occurred while analyzing the document. Please try again.",
            "error": "An error occurred while analyzing the document.",
            "sources": []
        }), 500


@app.route("/documents", methods=["GET"])
def list_documents():
    """Return JSON list of currently processed documents."""
    docs = get_uploaded_documents_info()
    # Normalize field names for the frontend contract
    normalized = []
    for d in docs:
        normalized.append({
            "document_id": d.get("document_id"),
            "filename": d.get("filename"),
            "file_type": d.get("file_type"),
            "page_count": d.get("page_count", d.get("pages", "N/A")),
            "chunk_count": d.get("chunk_count", d.get("chunks", 0)),
            "upload_time": d.get("upload_time", ""),
            "status": d.get("status", "Ready")
        })
    return jsonify({"success": True, "documents": normalized})


@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document_api(document_id):
    """
    Full document deletion: physical file + metadata + FAISS vectors (rebuild).
    Uses document_id only — never a raw filesystem path from the client.
    """
    if not get_document_by_id(document_id):
        return jsonify({"success": False, "message": "Document not found."}), 404

    try:
        result = delete_document(document_id)
        return jsonify({
            "success": True,
            "message": "Document deleted successfully",
            "document_id": document_id,
            "removed_chunks": result.get("removed_chunks", 0)
        })
    except CustomException as e:
        logger.error(f"Document deletion failed for '{document_id}': {e}")
        return jsonify({"success": False, "message": "Unable to delete document."}), 500
    except Exception as e:
        logger.error(f"Unexpected error deleting document '{document_id}': {e}")
        # Friendly message only — no internal details
        return jsonify({"success": False, "message": "Unable to delete document."}), 500


# ============================================================
# Chat history (session) API
# ============================================================

@app.route("/chats", methods=["GET"])
def list_chats_api():
    """Return all conversation summaries (chat_id, title, timestamps)."""
    try:
        chats = get_chats()
        return jsonify({
            "success": True,
            "chats": [
                {
                    "chat_id": c["chat_id"],
                    "title": c["title"],
                    "created_at": c["created_at"],
                    "updated_at": c["updated_at"],
                    "document_ids": c.get("document_ids", [])
                }
                for c in chats
            ]
        })
    except Exception as e:
        logger.error(f"Error listing chats: {e}")
        return jsonify({"success": False, "chats": [], "message": "Unable to load chats."}), 500


@app.route("/chats", methods=["DELETE"])
def clear_chats_api():
    """Clear ALL conversations and messages."""
    try:
        count = clear_all_chats()
        return jsonify({"success": True, "message": f"Cleared {count} conversation(s)."})
    except Exception as e:
        logger.error(f"Error clearing chats: {e}")
        return jsonify({"success": False, "message": "Unable to clear chat history."}), 500


@app.route("/chats/<chat_id>", methods=["GET"])
def get_chat_api(chat_id):
    """Return one conversation with its ordered messages."""
    chat = get_chat(chat_id)
    if chat is None:
        return jsonify({"success": False, "message": "Chat not found."}), 404
    try:
        return jsonify({
            "success": True,
            "chat": chat,
            "messages": get_messages(chat_id)
        })
    except Exception as e:
        logger.error(f"Error loading chat '{chat_id}': {e}")
        return jsonify({"success": False, "message": "Unable to load chat."}), 500


@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat_api(chat_id):
    """Delete one conversation and all of its messages."""
    try:
        deleted = delete_conversation(chat_id)
        if not deleted:
            return jsonify({"success": False, "message": "Chat not found."}), 404
        return jsonify({"success": True, "message": "Chat deleted successfully."})
    except Exception as e:
        logger.error(f"Error deleting chat '{chat_id}': {e}")
        return jsonify({"success": False, "message": "Unable to delete chat."}), 500


@app.route("/reset-documents", methods=["POST"])
def reset_all_documents():
    """Clear vector database and document index."""
    try:
        reset_vector_store()
        return jsonify({"success": True, "message": "Knowledge base reset successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/delete/chat/<int:chat_id>")
def delete(chat_id):
    delet_chat(chat_id)
    return redirect(url_for("index"))


@app.route("/clear", methods=["GET", "POST"])
def clear():
    clear_history()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)