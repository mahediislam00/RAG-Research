"""End-to-end checks: tenant isolation + documents actually leaving Qdrant."""
import os, sys, tempfile, uuid
import numpy as np

TMP = tempfile.mkdtemp()
os.environ.update(
    DATA_DIR=TMP,
    QDRANT_MODE="local",
    QDRANT_PATH=os.path.join(TMP, "qdrant"),
    JWT_SECRET="x" * 40,
    HF_TOKEN="",
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub the embedder so the test doesn't download sentence-transformers.
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

from fastapi.testclient import TestClient
from app import main as app_main

client = TestClient(app_main.app)
store = app_main.store

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
    events = [l for l in r.text.splitlines() if l.startswith("data:")]
    last = events[-1]
    assert '"done"' in last, last
    import json
    return json.loads(last[5:])["doc_id"]


text_a = "Alpha corp past performance. FAR 52.219-14 limitations on subcontracting. " * 40
text_b = "Bravo llc classified pricing. CLIN-0001 unit rate is 42 dollars. " * 40

print("\n=== 1. auth is required ===")
assert client.get("/api/documents").status_code in (401, 403)
assert client.post("/api/chat", json={"query": "hi"}).status_code in (401, 403)
ok("unauthenticated calls are rejected")

a_hdr, a_id = auth("alice@example.gov")
b_hdr, b_id = auth("bob@example.gov")
bad = {"Authorization": "Bearer " + "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyX3gifQ.zzz"}
assert client.get("/api/documents", headers=bad).status_code == 401
ok("forged token is rejected")
assert client.post("/api/auth/login",
                   json={"email": "alice@example.gov", "password": "wrong-pass1"}
                   ).status_code == 401
ok("wrong password is rejected")

print("\n=== 2. upload + metadata ===")
doc_a = upload(a_hdr, "alpha.txt", text_a)
doc_b = upload(b_hdr, "bravo.txt", text_b)
pts, _ = store._client.scroll(collection_name=app_main.store.__class__ and
                              __import__("app.vectorstore", fromlist=["x"]).COLLECTION_NAME,
                              limit=1, with_payload=True)
payload = pts[0].payload
for key in ("user_id", "document_id", "filename", "upload_time", "session_id"):
    assert key in payload, f"missing {key} in payload"
ok(f"every chunk carries {sorted(k for k in payload if k in ('user_id','document_id','filename','upload_time','session_id'))}")
assert store.count_chunks(a_id, doc_a) > 0 and store.count_chunks(b_id, doc_b) > 0
ok("chunks are stamped with their owner")

print("\n=== 3. isolation ===")
docs_a = client.get("/api/documents", headers=a_hdr).json()["documents"]
assert [d["doc_id"] for d in docs_a] == [doc_a]
ok("alice lists only her own document")

# Alice searches, explicitly naming Bob's document id.
r = client.post("/api/chat", headers=a_hdr,
                json={"query": "CLIN-0001 unit rate", "doc_ids": [doc_b]})
import json as _json
srcs = _json.loads([l for l in r.text.splitlines() if '"sources"' in l][0][5:])
assert srcs["passages"] == [], srcs
ok("alice naming bob's doc_id retrieves nothing")

# Even unscoped, Alice never sees Bob's text.
r = client.post("/api/chat", headers=a_hdr, json={"query": "CLIN-0001 unit rate"})
srcs = _json.loads([l for l in r.text.splitlines() if '"sources"' in l][0][5:])
assert all(p["filename"] == "alpha.txt" for p in srcs["passages"]), srcs
ok("alice's unscoped search returns only alpha.txt")

# Vector-level check: Bob's chunks are invisible to Alice's filter.
assert store.count_chunks(a_id, doc_b) == 0
ok("qdrant filter (user_id AND document_id) excludes bob's chunks from alice")

print("\n=== 4. cross-tenant delete ===")
before = store.count_chunks(b_id, doc_b)
r = client.delete(f"/api/documents/{doc_b}", headers=a_hdr)
assert r.status_code == 404, r.text
assert store.count_chunks(b_id, doc_b) == before
ok(f"alice deleting bob's doc -> 404, bob's {before} chunks untouched")

print("\n=== 5. delete actually removes vectors ===")
n = store.count_chunks(a_id, doc_a)
r = client.delete(f"/api/documents/{doc_a}", headers=a_hdr)
assert r.status_code == 200, r.text
assert r.json()["chunks_removed"] == n
assert store.count_chunks(a_id, doc_a) == 0
ok(f"delete removed all {n} chunks from qdrant (verified by count, not assumed)")
r = client.post("/api/chat", headers=a_hdr, json={"query": "past performance FAR"})
srcs = _json.loads([l for l in r.text.splitlines() if '"sources"' in l][0][5:])
assert srcs["passages"] == []
ok("deleted document is no longer retrievable")
assert not list((__import__("pathlib").Path(TMP) / "uploads" / a_id).glob("*")), "file left on disk"
ok("uploaded file removed from disk")

print("\n=== 6. orphan sweep (the old stale-document state) ===")
from qdrant_client.models import PointStruct
from app.vectorstore import COLLECTION_NAME
store._client.upsert(collection_name=COLLECTION_NAME, wait=True, points=[
    PointStruct(id=str(uuid.uuid4()), vector=_vec("ghost").tolist(),
                payload={"user_id": a_id, "document_id": "ghost_doc",
                         "filename": "ghost.pdf", "text": "leftover chunk",
                         "page_start": 1, "page_end": 1, "section": "",
                         "chunk_index": 0, "upload_time": 0, "session_id": None})])
assert store.count_chunks(a_id, "ghost_doc") == 1
removed = store.purge_orphans()
assert removed == 1 and store.count_chunks(a_id, "ghost_doc") == 0
ok("chunks with no ownership row are swept (repairs pre-existing stale data)")

print("\n=== 7. logout revokes the token ===")
assert client.post("/api/auth/logout", headers=b_hdr).status_code == 200
assert client.get("/api/documents", headers=b_hdr).status_code == 401
ok("token stops working the moment the session is revoked")

print("\nALL CHECKS PASSED\n")
