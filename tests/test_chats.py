"""End-to-end checks for persistent multi-chat conversations:

  * a user's chats are private to them (no cross-user read / rename / delete),
  * messages are saved on every /chat turn (user + assistant),
  * a title is auto-generated from the first user message and survives rename,
  * the chat list is ordered by most-recently-updated,
  * a conversation's document scope is remembered and restored,
  * deleting a chat removes its messages (cascade).

The LLM router is stubbed so no network / HF token is needed; the embedder is
stubbed exactly as in test_tenancy.py so sentence-transformers isn't downloaded.
"""
import os, sys, tempfile, time
import numpy as np

TMP = tempfile.mkdtemp()
os.environ.update(
    DATA_DIR=TMP,
    QDRANT_MODE="local",
    QDRANT_PATH=os.path.join(TMP, "qdrant"),
    JWT_SECRET="x" * 40,
    HF_TOKEN="test-token",   # non-empty so the router doesn't short-circuit
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- stub the embedder (same trick as test_tenancy) ---
from app import embeddings

_rng = np.random.default_rng(0)
_cache: dict[str, np.ndarray] = {}


def _vec(t: str) -> np.ndarray:
    if t not in _cache:
        v = _rng.standard_normal(32).astype(np.float32)
        _cache[t] = v / np.linalg.norm(v)
    return _cache[t]


embeddings.embed_passages = lambda texts, batch_size=64, progress_cb=None: (
    progress_cb and progress_cb(len(texts), len(texts)),
    np.vstack([_vec(t) for t in texts]),
)[1]
embeddings.embed_query = lambda text: _vec(text)

# --- stub the LLM router so /chat produces a deterministic answer ---
from app.llm import ChatRouter

_captured = {"messages": None}


def _fake_stream(self, messages, temperature=0.2):
    _captured["messages"] = messages          # so we can assert history threading
    yield {"type": "model", "id": "stub", "label": "StubModel"}
    yield {"type": "token", "text": "Grounded answer. [alpha.txt p.1]"}
    yield {"type": "done", "label": "StubModel"}


ChatRouter.stream_chat = _fake_stream

from fastapi.testclient import TestClient
from app import main as app_main

client = TestClient(app_main.app)

PASS = "correcthorse9"
ok = lambda m: print(f"  PASS  {m}")


def auth(email):
    r = client.post("/api/auth/register", json={"email": email, "password": PASS})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}, r.json()["user"]["user_id"]


def upload(headers, name, body):
    r = client.post("/api/upload", headers=headers,
                    files={"file": (name, body.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    import json
    last = [l for l in r.text.splitlines() if l.startswith("data:")][-1]
    assert '"done"' in last, last
    return json.loads(last[5:])["doc_id"]


def sse_events(text):
    import json
    out = []
    for l in text.splitlines():
        if l.startswith("data:"):
            try: out.append(json.loads(l[5:]))
            except Exception: pass
    return out


text_a = "Alpha corp past performance. FAR 52.219-14 limitations on subcontracting. " * 40

a_hdr, a_id = auth("alice@example.gov")
b_hdr, b_id = auth("bob@example.gov")
doc_a = upload(a_hdr, "alpha.txt", text_a)

print("\n=== 1. auth required on all chat routes ===")
for method, path in [("get", "/api/chats"), ("post", "/api/chats"),
                     ("get", "/api/chats/whatever"),
                     ("delete", "/api/chats/whatever")]:
    assert getattr(client, method)(path).status_code in (401, 403)
ok("unauthenticated chat calls are rejected")

print("\n=== 2. create / list / empty ===")
assert client.get("/api/chats", headers=a_hdr).json()["chats"] == []
ok("new user starts with no chats")
c1 = client.post("/api/chats", headers=a_hdr, json={}).json()["chat"]
assert c1["title"] == "New chat" and c1["chat_id"].startswith("chat_")
chats = client.get("/api/chats", headers=a_hdr).json()["chats"]
assert [c["chat_id"] for c in chats] == [c1["chat_id"]]
ok("chat is created and listed for its owner")

print("\n=== 3. /chat persists user + assistant messages ===")
r = client.post("/api/chat", headers=a_hdr,
                json={"query": "What is the past performance requirement?",
                      "chat_id": c1["chat_id"], "doc_ids": [doc_a]})
evs = sse_events(r.text)
assert any(e.get("type") == "chat" and e["chat_id"] == c1["chat_id"] for e in evs)
assert any(e.get("type") == "token" for e in evs)
loaded = client.get(f"/api/chats/{c1['chat_id']}", headers=a_hdr).json()
roles = [m["role"] for m in loaded["messages"]]
assert roles == ["user", "assistant"], roles
assert "Grounded answer" in loaded["messages"][1]["content"]
assert loaded["messages"][1]["model_label"] == "StubModel"
ok("user message and streamed assistant answer are both saved")

print("\n=== 4. auto-title from first message ===")
title = client.get("/api/chats", headers=a_hdr).json()["chats"][0]["title"]
assert title.startswith("What is the past performance"), title
ok(f"title auto-generated from first message: {title!r}")

print("\n=== 5. /chat with no chat_id auto-creates a conversation ===")
r = client.post("/api/chat", headers=a_hdr,
                json={"query": "Which clause governs subcontracting?", "doc_ids": [doc_a]})
evs = sse_events(r.text)
new_id = next(e["chat_id"] for e in evs if e.get("type") == "chat")
assert new_id != c1["chat_id"]
assert client.get(f"/api/chats/{new_id}", headers=a_hdr).status_code == 200
ok("a turn without chat_id opens and records a fresh chat")

print("\n=== 6. list ordered by most-recently-updated ===")
# new_id was just used, so it should sort ahead of c1.
order = [c["chat_id"] for c in client.get("/api/chats", headers=a_hdr).json()["chats"]]
assert order[0] == new_id and c1["chat_id"] in order, order
# Touch c1 again; it must jump back to the top.
client.post("/api/chat", headers=a_hdr,
            json={"query": "Follow-up question about Section L.",
                  "chat_id": c1["chat_id"], "doc_ids": [doc_a]})
order = [c["chat_id"] for c in client.get("/api/chats", headers=a_hdr).json()["chats"]]
assert order[0] == c1["chat_id"], order
ok("chats are ordered by updated_at, newest turn first")

print("\n=== 7. conversation history is threaded into the prompt ===")
msgs = _captured["messages"]
# system + prior user + prior assistant + current user
assert msgs[0]["role"] == "system"
joined = " ".join(m["content"] for m in msgs)
assert "past performance requirement" in joined  # earlier turn is present
assert msgs[-1]["role"] == "user" and "Section L" in msgs[-1]["content"]
ok("prior turns are included as context, current question is last")

print("\n=== 8. document scope is remembered per chat ===")
scoped = client.get(f"/api/chats/{c1['chat_id']}", headers=a_hdr).json()["chat"]
assert scoped["doc_ids"] == [doc_a], scoped["doc_ids"]
ok("chat restores the document scope it was held to")

print("\n=== 9. rename (and auto-title never clobbers it) ===")
client.patch(f"/api/chats/{c1['chat_id']}", headers=a_hdr, json={"title": "Subcontracting Q&A"})
assert client.get("/api/chats", headers=a_hdr).json()["chats"][0]["title"] == "Subcontracting Q&A"
# A further turn must not overwrite the chosen title.
client.post("/api/chat", headers=a_hdr,
            json={"query": "another turn", "chat_id": c1["chat_id"], "doc_ids": [doc_a]})
titles = {c["chat_id"]: c["title"] for c in client.get("/api/chats", headers=a_hdr).json()["chats"]}
assert titles[c1["chat_id"]] == "Subcontracting Q&A"
ok("rename sticks and survives subsequent messages")
assert client.patch(f"/api/chats/{c1['chat_id']}", headers=a_hdr, json={"title": "   "}).status_code == 422 \
    or client.patch(f"/api/chats/{c1['chat_id']}", headers=a_hdr, json={"title": ""}).status_code == 422
ok("empty rename is rejected")

print("\n=== 10. cross-user isolation (the core security property) ===")
# Bob cannot see, load, rename, delete, or post into Alice's chat.
assert client.get("/api/chats", headers=b_hdr).json()["chats"] == []
ok("bob's chat list does not include alice's chats")
assert client.get(f"/api/chats/{c1['chat_id']}", headers=b_hdr).status_code == 404
ok("bob loading alice's chat -> 404")
assert client.patch(f"/api/chats/{c1['chat_id']}", headers=b_hdr, json={"title": "hijack"}).status_code == 404
ok("bob renaming alice's chat -> 404")
assert client.delete(f"/api/chats/{c1['chat_id']}", headers=b_hdr).status_code == 404
ok("bob deleting alice's chat -> 404")
# Bob posting a turn "into" Alice's chat_id must 404, not append to her history.
before = len(client.get(f"/api/chats/{c1['chat_id']}", headers=a_hdr).json()["messages"])
r = client.post("/api/chat", headers=b_hdr,
                json={"query": "inject", "chat_id": c1["chat_id"]})
assert r.status_code == 404, r.status_code
after = len(client.get(f"/api/chats/{c1['chat_id']}", headers=a_hdr).json()["messages"])
assert after == before, (before, after)
ok("bob posting into alice's chat -> 404, her transcript untouched")

print("\n=== 11. delete cascades to messages ===")
victim = client.post("/api/chats", headers=a_hdr, json={}).json()["chat"]
client.post("/api/chat", headers=a_hdr,
            json={"query": "temp", "chat_id": victim["chat_id"], "doc_ids": [doc_a]})
from app import db as _db
assert _db.count_messages(victim["chat_id"], a_id) > 0
assert client.delete(f"/api/chats/{victim['chat_id']}", headers=a_hdr).status_code == 200
assert client.get(f"/api/chats/{victim['chat_id']}", headers=a_hdr).status_code == 404
assert _db.count_messages(victim["chat_id"], a_id) == 0
ok("deleting a chat removes it and all its messages")

print("\n=== 12. deleting the user cascades to chats + messages ===")
# Sanity on the schema's ON DELETE CASCADE from users.
c = client.post("/api/chats", headers=b_hdr, json={}).json()["chat"]
_db._exec("DELETE FROM users WHERE user_id=?", (b_id,))
assert _db.get_chat(c["chat_id"], b_id) is None
ok("removing a user erases their chats (FK cascade)")

print("\nALL CHAT CHECKS PASSED\n")
