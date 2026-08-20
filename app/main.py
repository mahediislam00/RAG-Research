"""FastAPI application: sign in, upload documents, run hybrid retrieval, and
stream grounded answers from the fallback LLM chain — now with persistent,
per-user multi-chat conversations.

Tenancy model
-------------
Identity comes from a signed bearer token and nothing else. `user_id` is never
read from a request body, path, or query string, so a client has no vocabulary
for naming another user. Every data route depends on `current_user`, and every
store call takes that principal's id:

    upload  -> store.add_document(user.user_id, …)     stamps the owner on each chunk
    chat    -> store.dense_search(user.user_id, …)     server-side filter in Qdrant
    delete  -> store.delete_document(user.user_id, …)  AND(user_id, document_id)

Conversation persistence follows the same rule. Every chat belongs to a single
user_id and every message to a chat_id; the chat service enforces ownership in
the same SQL that touches the row, so a guessed chat_id belonging to another
user resolves to 404, never someone else's transcript.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, chat_service, config, db
from . import eval_routes
from .auth import Credentials, Principal, current_user
from .embeddings import embed_passages
from .ingestion import extract_pages, chunk_pages
from .llm import AllModelsUnavailable, ChatRouter
from .prompts import build_messages
from .retrieval import HybridRetriever
from .vectorstore import Store

log = logging.getLogger("raglab")

app = FastAPI(title="Gov Contracting RAG")

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,   # never "*" — we send credentials
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

store = Store()
retriever = HybridRetriever(store)
router = ChatRouter()

app.include_router(eval_routes.router)

ALLOWED = {".pdf", ".docx", ".txt", ".md", ".text"}
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    db.purge_expired_sessions()
    try:
        removed = store.purge_orphans()
        if removed:
            log.warning("purged %d orphaned chunk(s) from Qdrant", removed)
    except Exception as e:  # never block startup on the sweep
        log.warning("orphan sweep skipped: %s", e)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# --- auth ---------------------------------------------------------------
@app.post("/api/auth/register")
def register(creds: Credentials):
    principal, token, exp = auth.register(creds)
    return {"token": token, "expires_at": exp,
            "user": {"user_id": principal.user_id, "email": principal.email}}


@app.post("/api/auth/login")
def login(creds: Credentials):
    principal, token, exp = auth.login(creds)
    return {"token": token, "expires_at": exp,
            "user": {"user_id": principal.user_id, "email": principal.email}}


@app.post("/api/auth/logout")
def logout(user: Principal = Depends(current_user)):
    db.revoke_session(user.session_id, user.user_id)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: Principal = Depends(current_user)):
    return {"user_id": user.user_id, "email": user.email}


# --- documents ----------------------------------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 user: Principal = Depends(current_user)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported type {suffix}. Allowed: {sorted(ALLOWED)}")

    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_BYTES // (1024*1024)} MB.")
    if not raw:
        raise HTTPException(400, "Empty file.")

    document_id = uuid.uuid4().hex[:12]
    user_dir = config.UPLOAD_DIR / user.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / f"{document_id}{suffix}"
    dest.write_bytes(raw)
    filename = Path(file.filename).name
    upload_time = time.time()

    def gen():
        def progress(percent: int, stage: str):
            return _sse({"type": "progress",
                         "percent": max(0, min(100, int(percent))),
                         "stage": stage})

        try:
            yield progress(4, "Reading document")
            pages = [p for p in extract_pages(dest) if p.text and p.text.strip()]
            if not pages:
                raise ValueError(
                    "No extractable text. If this is a scanned PDF, OCR it first "
                    "(e.g. `ocrmypdf in.pdf out.pdf`) and re-upload."
                )
            yield progress(22, f"Extracted {len(pages)} page(s)")

            chunks = chunk_pages(pages, filename=filename, doc_id=document_id)
            if not chunks:
                raise ValueError("Document produced no usable chunks.")
            yield progress(38, f"Split into {len(chunks)} chunk(s)")

            events: list[str] = []

            def emb_cb(done: int, total: int):
                frac = (done / total) if total else 1.0
                events.append(progress(38 + int(frac * 54),
                                       f"Embedding chunks {done}/{total}"))

            vectors = embed_passages([c.text for c in chunks], progress_cb=emb_cb)
            for ev in events:
                yield ev

            yield progress(95, "Writing to index")
            store.add_document(
                user_id=user.user_id,
                document_id=document_id,
                filename=filename,
                chunks=chunks,
                vectors=vectors,
                upload_time=upload_time,
                session_id=user.session_id,
            )
            db.add_document(
                document_id=document_id,
                user_id=user.user_id,
                filename=filename,
                stored_path=str(dest),
                pages=len(pages),
                chunks=len(chunks),
                upload_time=upload_time,
                session_id=user.session_id,
            )
            yield progress(100, "Indexed")
            yield _sse({"type": "done", "doc_id": document_id, "filename": filename,
                        "pages": len(pages), "chunks": len(chunks)})
        except Exception as e:
            log.exception("upload failed for user=%s doc=%s", user.user_id, document_id)
            dest.unlink(missing_ok=True)
            try:
                store.delete_document(user.user_id, document_id)
            except Exception:
                pass
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/documents")
def documents(user: Principal = Depends(current_user)):
    return {"documents": db.list_documents(user.user_id)}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str, user: Principal = Depends(current_user)):
    """Remove a document: its vectors, its ownership row, and its file."""
    row = db.get_document(document_id, user.user_id)

    try:
        removed = store.delete_document(user.user_id, document_id)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    if row is None and removed == 0:
        raise HTTPException(404, "Document not found")

    db.delete_document(document_id, user.user_id)
    if row:
        Path(row["stored_path"]).unlink(missing_ok=True)

    return {"deleted": document_id, "chunks_removed": removed}


@app.delete("/api/me/documents")
def delete_all_documents(user: Principal = Depends(current_user)):
    """Erase everything this user owns — vectors, rows, and files."""
    removed = store.delete_user(user.user_id)
    for d in db.list_documents(user.user_id):
        row = db.get_document(d["document_id"], user.user_id)
        if row:
            Path(row["stored_path"]).unlink(missing_ok=True)
        db.delete_document(d["document_id"], user.user_id)
    return {"chunks_removed": removed}


@app.get("/api/models")
def models(user: Principal = Depends(current_user)):
    return {"models": router.status()}


# --- chats (conversation persistence) -----------------------------------
# Every route below derives its user_id from the JWT via `current_user` and
# hands it to chat_service, which enforces ownership in the same SQL query that
# touches the row. A chat_id or message_id from the client is only ever a
# *lookup key scoped to the caller* — naming another user's chat matches nothing
# and returns 404 (never someone else's data, never a 403 existence oracle).
class CreateChatRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    doc_ids: list[str] | None = None


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.post("/api/chats")
def create_chat(req: CreateChatRequest, user: Principal = Depends(current_user)):
    # Pin scope only to documents the caller actually owns.
    scope = db.owned_document_ids(user.user_id, req.doc_ids) if req.doc_ids else None
    chat = chat_service.create_chat(user.user_id, title=req.title, doc_ids=scope)
    return {"chat": chat}


@app.get("/api/chats")
def list_chats(user: Principal = Depends(current_user)):
    # The sidebar payload: this user's chats, most-recently-updated first.
    return {"chats": chat_service.list_chats(user.user_id)}


@app.get("/api/chats/{chat_id}")
def load_chat(chat_id: str, user: Principal = Depends(current_user)):
    # Restore a full conversation: metadata (incl. pinned doc scope) + transcript.
    return chat_service.load_chat(chat_id, user.user_id)


@app.get("/api/chats/{chat_id}/messages")
def chat_messages(chat_id: str, user: Principal = Depends(current_user)):
    chat_service.require_chat(chat_id, user.user_id)  # ownership gate
    return {"messages": [
        chat_service._message_public(m)
        for m in db.list_messages(chat_id, user.user_id)
    ]}


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, req: RenameChatRequest,
                user: Principal = Depends(current_user)):
    return {"chat": chat_service.rename_chat(chat_id, user.user_id, req.title)}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user: Principal = Depends(current_user)):
    chat_service.delete_chat(chat_id, user.user_id)
    return {"deleted": chat_id}


# --- chat (retrieval + grounded streaming answer, persisted) ------------
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    doc_ids: list[str] | None = None
    mode: str = "qa"           # qa | proposal | sow
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    chat_id: str | None = None  # continue an existing conversation; None = new


@app.post("/api/chat")
def chat(req: ChatRequest, user: Principal = Depends(current_user)):
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    # Resolve the conversation up front. A supplied chat_id must belong to the
    # caller (require_chat -> 404 otherwise); with none, we open a fresh chat so
    # every /chat turn is always recorded somewhere the user owns.
    if req.chat_id:
        chat_service.require_chat(req.chat_id, user.user_id)
        chat_id = req.chat_id
        created = False
    else:
        chat_id = chat_service.create_chat(user.user_id)["chat_id"]
        created = True

    # Client-supplied ids are a *narrowing* hint, never an authorization claim:
    # anything the caller does not own is dropped here, and Qdrant re-checks
    # ownership on the query anyway. The resulting scope is remembered on the
    # chat so reopening it retrieves against the same documents.
    scope = db.owned_document_ids(user.user_id, req.doc_ids)
    chat_service.set_scope(chat_id, user.user_id,
                           req.doc_ids if req.doc_ids else None)

    # Prior turns BEFORE this one become conversational context for the model.
    history = chat_service.recent_history(chat_id, user.user_id)

    # Persist the human turn now (auto-titles a new chat, bumps updated_at).
    chat_service.add_user_message(chat_id, user.user_id, req.query)

    passages = retriever.retrieve(user.user_id, req.query, scope) if scope else []

    def gen():
        # Tell the client which conversation this belongs to, so a brand-new
        # chat can be tracked and slotted into the sidebar immediately.
        yield _sse({"type": "chat", "chat_id": chat_id, "created": created})

        yield _sse({"type": "sources", "passages": [
            {k: p.get(k) for k in ("filename", "page_start", "page_end", "section",
                                   "matched_by", "fused_score", "dense_score",
                                   "bm25_score", "text")}
            for p in passages
        ]})

        if not passages:
            yield _sse({"type": "error",
                        "message": "No indexed documents matched. Upload a "
                                   "document or broaden your question."})
            yield _sse({"type": "done"})
            return

        messages = build_messages(req.mode, req.query, passages, history=history)
        answer_parts: list[str] = []
        model_label: str | None = None
        try:
            for event in router.stream_chat(messages, temperature=req.temperature):
                if event.get("type") == "token":
                    answer_parts.append(event.get("text", ""))
                elif event.get("type") == "model":
                    model_label = event.get("label")
                yield _sse(event)
        except AllModelsUnavailable as e:
            yield _sse({"type": "error", "message": str(e)})
            yield _sse({"type": "done"})

        # Persist the assistant turn once the stream is complete. Even a partial
        # answer (client disconnect, model cut off) is saved so the transcript
        # matches what the user saw and the chat stays continuable.
        answer = "".join(answer_parts).strip()
        if answer:
            chat_service.add_assistant_message(
                chat_id, user.user_id, answer, model_label=model_label
            )

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- frontend -----------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/evaluate")
def evaluate_page():
    return FileResponse(STATIC_DIR / "evaluate.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
