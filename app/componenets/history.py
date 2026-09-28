import sqlite3
import os
import json
import uuid
from datetime import datetime
from typing import List, Tuple, Any, Optional, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "chat_history.db")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_chat_title(question: str) -> str:
    """Build a short chat title from the first user question (no LLM needed)."""
    cleaned = " ".join((question or "").split())
    if not cleaned:
        return "New Chat"
    return cleaned if len(cleaned) <= 42 else cleaned[:40].rstrip() + "..."


def create_table():
    """Create session-based chat tables and migrate legacy flat history once."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    # Legacy flat table (kept so old code paths keep working)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            time TEXT,
            sources TEXT
        )
    """)

    # Session-based conversations
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            document_ids TEXT DEFAULT '[]'
        )
    """)

    # Messages belonging to a conversation
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '',
            timestamp TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
    conn.commit()

    # Gracefully add sources column if older legacy schema is present
    try:
        c.execute("PRAGMA table_info(chats)")
        columns = [row[1] for row in c.fetchall()]
        if "sources" not in columns:
            c.execute("ALTER TABLE chats ADD COLUMN sources TEXT")
            conn.commit()
    except Exception:
        pass

    # One-time migration: import legacy Q/A rows as individual conversations
    try:
        c.execute("SELECT question, answer, time, sources FROM chats ORDER BY id ASC")
        legacy_rows = c.fetchall()
        for question, answer, when, sources in legacy_rows:
            if not question:
                continue
            chat_id = "chat_" + uuid.uuid4().hex[:6]
            ts = when or _now()
            c.execute(
                "INSERT INTO conversations (chat_id, title, created_at, updated_at, document_ids) VALUES (?,?,?,?,?)",
                (chat_id, generate_chat_title(question), ts, ts, "[]")
            )
            c.execute(
                "INSERT INTO messages (chat_id, role, content, sources, timestamp) VALUES (?,?,?,?,?)",
                (chat_id, "user", question, "", ts)
            )
            c.execute(
                "INSERT INTO messages (chat_id, role, content, sources, timestamp) VALUES (?,?,?,?,?)",
                (chat_id, "assistant", answer or "", sources or "", ts)
            )
        if legacy_rows:
            c.execute("DELETE FROM chats")
            conn.commit()
    except Exception:
        pass

    conn.close()


def save_chat(question: str, answer: str, sources: Optional[Any] = None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    sources_str = ""
    if sources:
        if isinstance(sources, (list, dict)):
            sources_str = json.dumps(sources)
        else:
            sources_str = str(sources)

    c.execute("""
        INSERT INTO chats (question, answer, time, sources)
        VALUES (?, ?, ?, ?)
    """, (question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sources_str))

    conn.commit()
    conn.close()


def get_history() -> List[Tuple]:
    """Retrieve chat history, returning tuples of (id, question, answer, time, sources)."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        SELECT id, question, answer, time, COALESCE(sources, '') FROM chats
        ORDER BY id ASC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def delet_chat(chat_id: int):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def clear_history():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM chats")
    conn.commit()
    conn.close()


# ============================================================
# Session-based chat history (chat_id -> messages)
# ============================================================

def _row_to_chat(row) -> Dict[str, Any]:
    chat_id, title, created_at, updated_at, document_ids = row
    try:
        doc_ids = json.loads(document_ids) if document_ids else []
    except Exception:
        doc_ids = []
    return {
        "chat_id": chat_id,
        "title": title or "New Chat",
        "created_at": created_at,
        "updated_at": updated_at,
        "document_ids": doc_ids,
    }


def create_chat(title: Optional[str] = None, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new chat conversation and return its record."""
    chat_id = "chat_" + uuid.uuid4().hex[:6]
    now = _now()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (chat_id, title, created_at, updated_at, document_ids) VALUES (?,?,?,?,?)",
        (chat_id, title or "New Chat", now, now, json.dumps(document_ids or []))
    )
    conn.commit()
    conn.close()
    return {
        "chat_id": chat_id,
        "title": title or "New Chat",
        "created_at": now,
        "updated_at": now,
        "document_ids": document_ids or [],
    }


def get_chats() -> List[Dict[str, Any]]:
    """Return all conversations ordered by most recent activity."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT chat_id, title, created_at, updated_at, document_ids FROM conversations ORDER BY updated_at DESC"
    )
    rows = c.fetchall()
    conn.close()
    return [_row_to_chat(r) for r in rows]


def get_chat(chat_id: str) -> Optional[Dict[str, Any]]:
    """Return a single conversation record or None."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT chat_id, title, created_at, updated_at, document_ids FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )
    row = c.fetchone()
    conn.close()
    return _row_to_chat(row) if row else None


def get_messages(chat_id: str) -> List[Dict[str, Any]]:
    """Return all messages for a conversation in chronological order."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT id, role, content, sources, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,)
    )
    rows = c.fetchall()
    conn.close()

    messages = []
    for msg_id, role, content, sources_str, timestamp in rows:
        parsed_sources = []
        if sources_str:
            try:
                parsed_sources = json.loads(sources_str)
            except Exception:
                parsed_sources = [{"label": sources_str}]
        messages.append({
            "id": msg_id,
            "chat_id": chat_id,
            "role": role,
            "message": content,
            "content": content,
            "sources": parsed_sources,
            "timestamp": timestamp,
            "time": timestamp,
        })
    return messages


def save_message(chat_id: str, role: str, content: str, sources: Optional[Any] = None) -> bool:
    """Persist one message in a conversation and bump its updated_at."""
    try:
        sources_str = ""
        if sources:
            if isinstance(sources, (list, dict)):
                sources_str = json.dumps(sources)
            else:
                sources_str = str(sources)

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (chat_id, role, content, sources, timestamp) VALUES (?,?,?,?,?)",
            (chat_id, role, content, sources_str, _now())
        )
        c.execute("UPDATE conversations SET updated_at = ? WHERE chat_id = ?", (_now(), chat_id))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def set_chat_title(chat_id: str, title: str):
    """Update the chat title (used when the first question arrives)."""
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("UPDATE conversations SET title = ? WHERE chat_id = ?", (title, chat_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


def update_chat_documents(chat_id: str, document_ids: List[str]):
    """Merge document IDs used in a conversation (stored for later filtering)."""
    try:
        chat = get_chat(chat_id)
        if not chat:
            return
        merged = list(dict.fromkeys(list(chat.get("document_ids", [])) + list(document_ids or [])))
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("UPDATE conversations SET document_ids = ? WHERE chat_id = ?", (json.dumps(merged), chat_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


def delete_conversation(chat_id: str) -> bool:
    """Delete one conversation and all of its messages. Returns True if found."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT chat_id FROM conversations WHERE chat_id = ?", (chat_id,))
    exists = c.fetchone() is not None
    if exists:
        c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        c.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
        conn.commit()
    conn.close()
    return exists


def clear_all_chats() -> int:
    """Delete ALL conversations and messages. Returns number of chats removed."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM conversations")
    count = c.fetchone()[0]
    c.execute("DELETE FROM messages")
    c.execute("DELETE FROM conversations")
    c.execute("DELETE FROM chats")  # also clear legacy flat history
    conn.commit()
    conn.close()
    return count