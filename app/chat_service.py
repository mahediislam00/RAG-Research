"""Conversation persistence: create, list, load, rename, delete chats and the
messages inside them.

This module is the single place that turns "a request from an authenticated
principal" into "a row in SQLite", and it inherits the tenancy rule the rest of
the app lives by:

    user_id is passed in from the caller (derived from the JWT, never a request
    body) and threaded through EVERY db call. There is no function here that
    reads or writes a chat or a message without the owner's id in the same
    query, so a caller has no vocabulary for naming another user's data.

Ownership is enforced twice, defence in depth:
  * `require_chat()` first confirms the chat exists AND belongs to the caller,
    raising 404 (not 403) otherwise — a missing chat and someone else's chat are
    indistinguishable, so there is no existence oracle.
  * The underlying db.* calls are themselves filtered by user_id, so even a bug
    that skipped `require_chat()` could not reach across tenants.

The service also owns two conveniences the API layer should not re-implement:
title auto-generation from the first user message, and assembling recent turns
into the message list handed to the LLM.
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import HTTPException

from . import db

# How many prior turns (user+assistant messages) to feed back into the model as
# conversational context. Kept small so the retrieved passages — not the history
# — remain the dominant signal, and so the prompt stays well inside budget.
HISTORY_TURNS = 8
TITLE_MAX_CHARS = 60


# --- ids ----------------------------------------------------------------
def _chat_id() -> str:
    return f"chat_{uuid.uuid4().hex[:16]}"


def _message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:16]}"


# --- doc scope (de)serialization ---------------------------------------
def _encode_scope(doc_ids: list[str] | None) -> str | None:
    """A pinned, de-duplicated list of document ids, or None for 'all docs'.
    Stored as JSON so a chat reopened later retrieves against the same scope."""
    if not doc_ids:
        return None
    cleaned = sorted({d for d in doc_ids if isinstance(d, str) and d})
    return json.dumps(cleaned) if cleaned else None


def _decode_scope(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(val, list):
        return [d for d in val if isinstance(d, str) and d] or None
    return None


# --- title generation ---------------------------------------------------
def derive_title(first_message: str) -> str:
    """Condense the first user message into a short, single-line title —
    ChatGPT-style. Never empty (falls back to 'New chat')."""
    text = " ".join((first_message or "").split())
    if not text:
        return "New chat"
    if len(text) <= TITLE_MAX_CHARS:
        return text
    # Cut on a word boundary near the limit rather than mid-word.
    clipped = text[:TITLE_MAX_CHARS].rsplit(" ", 1)[0].strip()
    clipped = clipped or text[:TITLE_MAX_CHARS].strip()
    return clipped + "…"


# --- serialization for the API -----------------------------------------
def _chat_public(row: dict) -> dict:
    return {
        "chat_id": row["chat_id"],
        "title": row.get("title") or "New chat",
        "doc_ids": _decode_scope(row.get("doc_scope")),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _message_public(row: dict) -> dict:
    return {
        "message_id": row["message_id"],
        "role": row["role"],
        "content": row["content"],
        "model_label": row.get("model_label"),
        "created_at": row["created_at"],
    }


# --- ownership gate -----------------------------------------------------
def require_chat(chat_id: str, user_id: str) -> dict:
    """Return the chat row iff the caller owns it, else 404. Every mutating and
    reading operation on a specific chat funnels through here first."""
    row = db.get_chat(chat_id, user_id)
    if row is None:
        raise HTTPException(404, "Chat not found")
    return row


# --- CRUD ---------------------------------------------------------------
def create_chat(user_id: str, title: str | None = None,
                doc_ids: list[str] | None = None) -> dict:
    now = time.time()
    chat_id = _chat_id()
    clean_title = (title or "").strip() or None
    db.create_chat(chat_id, user_id, clean_title, _encode_scope(doc_ids), now)
    return _chat_public(db.get_chat(chat_id, user_id))


def list_chats(user_id: str) -> list[dict]:
    """Sidebar payload: the user's chats, most recent first."""
    return [_chat_public(r) for r in db.list_chats(user_id)]


def load_chat(chat_id: str, user_id: str) -> dict:
    """Everything needed to restore a conversation on the screen: the chat's
    metadata (including its pinned doc scope) and the full message transcript."""
    chat = require_chat(chat_id, user_id)
    messages = [_message_public(m) for m in db.list_messages(chat_id, user_id)]
    return {"chat": _chat_public(chat), "messages": messages}


def rename_chat(chat_id: str, user_id: str, title: str) -> dict:
    require_chat(chat_id, user_id)
    clean = (title or "").strip()
    if not clean:
        raise HTTPException(400, "Title cannot be empty.")
    clean = clean[:120]
    db.rename_chat(chat_id, user_id, clean)
    return _chat_public(db.get_chat(chat_id, user_id))


def delete_chat(chat_id: str, user_id: str) -> None:
    require_chat(chat_id, user_id)
    db.delete_chat(chat_id, user_id)  # messages cascade


def set_scope(chat_id: str, user_id: str, doc_ids: list[str] | None) -> None:
    require_chat(chat_id, user_id)
    db.set_chat_scope(chat_id, user_id, _encode_scope(doc_ids))


def get_scope(chat_id: str, user_id: str) -> list[str] | None:
    chat = require_chat(chat_id, user_id)
    return _decode_scope(chat.get("doc_scope"))


# --- messages -----------------------------------------------------------
def add_user_message(chat_id: str, user_id: str, content: str) -> dict:
    """Persist the human turn, bump updated_at, and — if this is the first
    message — auto-name the chat from it. Bumping here (not only when the
    assistant replies) means a turn that produces no answer, e.g. no documents
    matched, still floats the chat to the top and is honestly recorded."""
    require_chat(chat_id, user_id)
    now = time.time()
    is_first = db.count_messages(chat_id, user_id) == 0
    mid = _message_id()
    db.add_message(mid, chat_id, user_id, "user", content, None, now)
    # touch_chat bumps updated_at; title_if_empty only fills a still-empty title,
    # so a user-chosen name always survives.
    db.touch_chat(chat_id, user_id, now,
                  title_if_empty=derive_title(content) if is_first else None)
    return {"message_id": mid, "created_at": now, "is_first": is_first}


def add_assistant_message(chat_id: str, user_id: str, content: str,
                          model_label: str | None = None) -> dict:
    """Persist the assistant turn and float the chat to the top of the list by
    bumping updated_at. Called after the stream to the client completes."""
    require_chat(chat_id, user_id)
    now = time.time()
    mid = _message_id()
    db.add_message(mid, chat_id, user_id, "assistant", content, model_label, now)
    db.touch_chat(chat_id, user_id, now)
    return {"message_id": mid, "created_at": now}


def recent_history(chat_id: str, user_id: str,
                   turns: int = HISTORY_TURNS) -> list[dict]:
    """The last `turns` messages as plain {role, content} dicts, oldest first —
    ready to interleave into the LLM prompt. The just-saved user message is
    included, so callers pass the tail of the transcript straight through."""
    msgs = db.list_messages(chat_id, user_id)
    tail = msgs[-turns:] if turns and turns > 0 else msgs
    return [{"role": m["role"], "content": m["content"]} for m in tail]
