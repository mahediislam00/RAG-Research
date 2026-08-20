"""SQLite persistence for users, sessions, the per-user document registry, and
persistent multi-chat conversations.

The registry is deliberately the *authorization* store, not the search store:
authorization decisions are made here, in SQL, before any vector call runs. The
same rule extends to chats and messages — every read/write is filtered by
user_id in the same statement that touches the row.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from .config import DB_PATH

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    pages       INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    upload_time REAL NOT NULL,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

-- One row per conversation. Every chat belongs to exactly one user; the FK
-- cascade means deleting a user erases their whole history. doc_scope is the
-- JSON-encoded list of document_ids the conversation is pinned to (NULL/empty
-- means "all of the user's documents"), so reopening a chat restores not just
-- its messages but the retrieval scope it was held to. updated_at is bumped on
-- every new turn, which is the column the chat list is ordered by.
CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title      TEXT,
    doc_scope  TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_user_updated
    ON chats(user_id, updated_at DESC);

-- One row per message. chat_id scopes it to a conversation and user_id is
-- carried redundantly so an ownership check never has to join through chats.
-- Both FKs cascade, so deleting a chat (or a user) removes the transcript.
CREATE TABLE IF NOT EXISTS messages (
    message_id  TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    model_label TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
"""


# Additive, idempotent migrations for databases created before a column/table
# existed. CREATE TABLE IF NOT EXISTS in SCHEMA already handles brand-new
# installs; this pass repairs older files without dropping data.
def _migrate(conn: sqlite3.Connection) -> None:
    def _safe(sql: str) -> None:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # already applied (duplicate column / existing object)

    _safe(
        "CREATE TABLE IF NOT EXISTS chats ("
        " chat_id TEXT PRIMARY KEY,"
        " user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,"
        " title TEXT, doc_scope TEXT,"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    _safe(
        "CREATE TABLE IF NOT EXISTS messages ("
        " message_id TEXT PRIMARY KEY,"
        " chat_id TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,"
        " user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,"
        " role TEXT NOT NULL, content TEXT NOT NULL, model_label TEXT,"
        " created_at REAL NOT NULL)"
    )
    _safe("ALTER TABLE chats ADD COLUMN doc_scope TEXT")
    _safe("ALTER TABLE chats ADD COLUMN title TEXT")
    _safe("ALTER TABLE messages ADD COLUMN model_label TEXT")
    _safe("CREATE INDEX IF NOT EXISTS idx_chats_user_updated "
          "ON chats(user_id, updated_at DESC)")
    _safe("CREATE INDEX IF NOT EXISTS idx_messages_chat "
          "ON messages(chat_id, created_at)")
    _safe("CREATE INDEX IF NOT EXISTS idx_messages_user "
          "ON messages(user_id)")
    conn.commit()


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            _conn.commit()
            _migrate(_conn)   # repair older databases in place (additive only)
        return _conn


def _exec(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    conn = connect()
    with _lock:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = connect()
    with _lock:
        return conn.execute(sql, params).fetchall()


# --- users --------------------------------------------------------------
def create_user(user_id: str, email: str, password_hash: str) -> None:
    _exec(
        "INSERT INTO users (user_id, email, password_hash, created_at) VALUES (?,?,?,?)",
        (user_id, email.strip(), password_hash, time.time()),
    )


def get_user_by_email(email: str) -> sqlite3.Row | None:
    rows = _query("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),))
    return rows[0] if rows else None


def get_user(user_id: str) -> sqlite3.Row | None:
    rows = _query("SELECT * FROM users WHERE user_id = ?", (user_id,))
    return rows[0] if rows else None


# --- sessions -----------------------------------------------------------
def create_session(session_id: str, user_id: str, ttl_seconds: int) -> None:
    now = time.time()
    _exec(
        "INSERT INTO sessions (session_id, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        (session_id, user_id, now, now + ttl_seconds),
    )


def session_is_valid(session_id: str, user_id: str) -> bool:
    rows = _query(
        "SELECT 1 FROM sessions WHERE session_id=? AND user_id=? "
        "AND revoked=0 AND expires_at > ?",
        (session_id, user_id, time.time()),
    )
    return bool(rows)


def revoke_session(session_id: str, user_id: str) -> None:
    _exec(
        "UPDATE sessions SET revoked=1 WHERE session_id=? AND user_id=?",
        (session_id, user_id),
    )


def purge_expired_sessions() -> int:
    cur = _exec("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
    return cur.rowcount


# --- documents ----------------------------------------------------------
def add_document(document_id: str, user_id: str, filename: str, stored_path: str,
                 pages: int, chunks: int, upload_time: float,
                 session_id: str | None) -> None:
    _exec(
        "INSERT OR REPLACE INTO documents "
        "(document_id, user_id, filename, stored_path, pages, chunks, upload_time, session_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (document_id, user_id, filename, stored_path, pages, chunks, upload_time, session_id),
    )


def list_documents(user_id: str) -> list[dict]:
    rows = _query(
        "SELECT document_id, filename, pages, chunks, upload_time, session_id "
        "FROM documents WHERE user_id=? ORDER BY upload_time DESC",
        (user_id,),
    )
    return [{**dict(r), "doc_id": r["document_id"]} for r in rows]


def get_document(document_id: str, user_id: str) -> dict | None:
    rows = _query(
        "SELECT * FROM documents WHERE document_id=? AND user_id=?",
        (document_id, user_id),
    )
    return dict(rows[0]) if rows else None


def owned_document_ids(user_id: str, requested: list[str] | None) -> list[str]:
    rows = _query("SELECT document_id FROM documents WHERE user_id=?", (user_id,))
    owned = {r["document_id"] for r in rows}
    if not requested:
        return sorted(owned)
    return sorted(owned & set(requested))


def all_document_ids() -> set[str]:
    return {r["document_id"] for r in _query("SELECT document_id FROM documents")}


def delete_document(document_id: str, user_id: str) -> bool:
    cur = _exec(
        "DELETE FROM documents WHERE document_id=? AND user_id=?",
        (document_id, user_id),
    )
    return cur.rowcount > 0


# --- chats --------------------------------------------------------------
# Every read and write here is filtered by user_id in the same query that
# touches the row, so ownership is enforced at the storage layer and not just
# in a service check above it. Naming another user's chat_id matches nothing.
def create_chat(chat_id: str, user_id: str, title: str | None,
                doc_scope: str | None, now: float) -> None:
    _exec(
        "INSERT INTO chats (chat_id, user_id, title, doc_scope, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (chat_id, user_id, title, doc_scope, now, now),
    )


def list_chats(user_id: str) -> list[dict]:
    rows = _query(
        "SELECT chat_id, title, doc_scope, created_at, updated_at "
        "FROM chats WHERE user_id=? ORDER BY updated_at DESC, created_at DESC",
        (user_id,),
    )
    return [dict(r) for r in rows]


def get_chat(chat_id: str, user_id: str) -> dict | None:
    rows = _query(
        "SELECT * FROM chats WHERE chat_id=? AND user_id=?",
        (chat_id, user_id),
    )
    return dict(rows[0]) if rows else None


def rename_chat(chat_id: str, user_id: str, title: str) -> bool:
    cur = _exec(
        "UPDATE chats SET title=? WHERE chat_id=? AND user_id=?",
        (title, chat_id, user_id),
    )
    return cur.rowcount > 0


def set_chat_scope(chat_id: str, user_id: str, doc_scope: str | None) -> bool:
    cur = _exec(
        "UPDATE chats SET doc_scope=? WHERE chat_id=? AND user_id=?",
        (doc_scope, chat_id, user_id),
    )
    return cur.rowcount > 0


def touch_chat(chat_id: str, user_id: str, now: float,
               title_if_empty: str | None = None) -> None:
    if title_if_empty is not None:
        _exec(
            "UPDATE chats SET updated_at=?, "
            "title = COALESCE(NULLIF(title, ''), ?) "
            "WHERE chat_id=? AND user_id=?",
            (now, title_if_empty, chat_id, user_id),
        )
    else:
        _exec(
            "UPDATE chats SET updated_at=? WHERE chat_id=? AND user_id=?",
            (now, chat_id, user_id),
        )


def delete_chat(chat_id: str, user_id: str) -> bool:
    cur = _exec(
        "DELETE FROM chats WHERE chat_id=? AND user_id=?",
        (chat_id, user_id),
    )
    return cur.rowcount > 0


# --- messages -----------------------------------------------------------
def add_message(message_id: str, chat_id: str, user_id: str, role: str,
                content: str, model_label: str | None, now: float) -> None:
    _exec(
        "INSERT INTO messages "
        "(message_id, chat_id, user_id, role, content, model_label, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (message_id, chat_id, user_id, role, content, model_label, now),
    )


def list_messages(chat_id: str, user_id: str) -> list[dict]:
    rows = _query(
        "SELECT message_id, role, content, model_label, created_at "
        "FROM messages WHERE chat_id=? AND user_id=? ORDER BY created_at ASC, rowid ASC",
        (chat_id, user_id),
    )
    return [dict(r) for r in rows]


def count_messages(chat_id: str, user_id: str) -> int:
    rows = _query(
        "SELECT COUNT(*) AS n FROM messages WHERE chat_id=? AND user_id=?",
        (chat_id, user_id),
    )
    return int(rows[0]["n"]) if rows else 0
