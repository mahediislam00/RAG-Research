# Security & tenancy

## Why documents weren't getting deleted

Chunks carried no owner and no reliable handle. The registry lived in
`data/index/documents.json`, and `Store.delete_document()` refused to touch
Qdrant unless the id was in that file:

```python
if doc_id not in self.documents:   # ← registry, not the vector DB
    return False                   # → HTTP 404, points stay forever
```

Any drift between the file and the collection stranded points permanently:

* an upload that upserted points and then crashed before the JSON was written;
* `QDRANT_MODE=memory` (the default) — vectors die on restart, the JSON survives,
  so the registry and the collection disagree in both directions;
* concurrent uploads racing on a whole-file JSON rewrite, losing an entry;
* a wiped or restored `data/` directory.

Stranded points stayed *searchable* while being *unreachable* by any delete call.

## What changed

**Metadata on every chunk.** Each Qdrant point now carries:

```json
{
  "user_id":     "user_9f3c…",
  "document_id": "invoice_543",
  "filename":    "report.pdf",
  "upload_time": 1752480000.0,
  "session_id":  "sess_…",
  "page_start": 4, "page_end": 5, "section": "C.3.1", "chunk_index": 12,
  "text": "…"
}
```

**Delete is scoped and verified.** `AND(user_id, document_id)`, `wait=True`, then
a `count()` on the same filter. If anything survived, the store raises instead of
dropping the registry row — a failed delete can no longer masquerade as success.
Deletion also runs even when the registry row is missing, which is exactly the
state old stale documents are stuck in.

**Orphan sweep on startup.** `Store.purge_orphans()` scrolls the collection and
removes points whose `document_id` has no owner row in SQLite. This cleans up
data left behind by the previous version. It runs at startup and is safe to run
any time.

**Search can't resurrect them.** Retrieval is always scoped to document ids that
exist in the ownership table, so an orphaned point is invisible even before the
sweep reaches it.

## Tenancy model

| Layer | Control |
|---|---|
| Identity | Session-bound JWT (HS256). `user_id` comes from the signature, never from a request body, path, or query. |
| Authorization | SQLite `documents` table is the ownership record. Client-supplied `doc_ids` are intersected with the caller's rows (`db.owned_document_ids`) before any vector call. |
| Data plane | Qdrant re-checks: every search and delete carries a server-side `user_id` filter. `Store` takes `user_id` as a required positional arg, so a forgotten check is a `TypeError`, not a leak. |
| Files | Uploads land in `data/uploads/<user_id>/<document_id><ext>` — server-chosen names, no client-controlled paths. |

Defense in depth is the point: a guessed `document_id` is dropped by the SQL
intersection, and would be rejected by the Qdrant filter even if it weren't.

## Auth details

* Passwords: `hashlib.scrypt` (N=2^14, r=8, p=1), per-password salt, constant-time
  compare. Login verifies against a dummy hash for unknown emails so response
  time doesn't reveal whether an account exists.
* Tokens: HS256, 12h TTL, signed with `JWT_SECRET`, and bound to a row in
  `sessions`. Logout revokes the row, so a leaked token dies immediately rather
  than at expiry.
* Login throttle: 8 attempts per email per 60s (in-process — move to Redis if you
  run more than one worker).
* Uploads capped at 50 MB (`MAX_UPLOAD_MB`).

## Before you run this in production

1. **Set `JWT_SECRET`** (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
   Without it the app generates one into `data/.jwt_secret`; two app servers would
   then sign with different keys and reject each other's tokens.
2. **Don't ship `QDRANT_MODE=memory`.** Use `remote` against a real Qdrant so
   payload indexes, durable deletes, and concurrent access all work.
3. **Lock down Qdrant itself.** It has no per-user auth — anyone who can reach
   port 6333 reads every tenant's chunks, filters or not. Bind it to a private
   network, set `QDRANT_API_KEY` (and a read-only key for anything that only
   queries), and terminate TLS.
4. **Serve the app over HTTPS.** Bearer tokens in cleartext are bearer tokens for
   whoever is listening.
5. **Set `CORS_ORIGINS`** to your exact frontend origin, or leave it empty for
   same-origin. Never `*` — the API takes credentials.
6. **Back up `data/raglab.db`.** It is the ownership record; losing it turns every
   document into an orphan (recoverable data, but the sweep will delete it).
7. Consider per-user Qdrant collections if you ever need hard physical isolation
   rather than filter-based isolation.

## Tests

```bash
python tests/test_tenancy.py
```

Covers: unauthenticated rejection, forged tokens, wrong passwords, per-chunk
metadata, list/search/delete isolation across two users, cross-tenant delete →
404 with the victim's chunks intact, delete actually emptying Qdrant (verified by
count), deleted documents disappearing from retrieval, the orphan sweep, and
logout revocation.
