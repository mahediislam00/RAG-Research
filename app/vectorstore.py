"""Qdrant-backed vector store — multi-tenant, with real deletes.

Two problems this module now solves.

1. Documents were not actually leaving the vector DB.
   The old store kept its registry in ``documents.json`` and refused to delete
   anything whose id was missing from that file. Any registry/collection drift
   (a crashed upload that upserted points before the registry was written, a
   wiped data dir, the default in-memory Qdrant losing vectors while the JSON
   survived) left points stranded with no way to reach them — the delete path
   returned 404 while the chunks stayed searchable forever.

   Fixes:
     * every point carries ``document_id`` and ``user_id``, both payload-indexed;
     * delete issues a filtered delete with ``wait=True`` and then *counts* the
       remaining matches, so a delete that didn't take raises instead of lying;
     * delete runs against Qdrant even when the registry row is already gone;
     * ``purge_orphans()`` sweeps points whose document_id no longer exists in
       SQLite, cleaning up drift left behind by earlier versions.

2. Every user could read every document.
   There is no longer a code path that queries the collection without a
   ``user_id`` filter: it is a required positional argument on both search and
   delete, and it is applied server-side by Qdrant. A caller who guesses another
   tenant's document_id gets an empty result, because the filter is an AND of
   (user_id, document_id) and user_id comes from a signed token, never from the
   request body.

Payload written per chunk::

    {
      "user_id":     "user_9f3c…",   # owner — the tenancy key
      "document_id": "invoice_543",  # what delete targets
      "filename":    "report.pdf",
      "upload_time": 1752480000.0,
      "session_id":  "sess_…",       # login that ingested it
      "page_start": 4, "page_end": 5, "section": "C.3.1", "chunk_index": 12,
      "text":        "…"
    }
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np

from . import config, db
from .ingestion import Chunk

COLLECTION_NAME = config.QDRANT_COLLECTION


def _build_client():
    """Return an initialised QdrantClient for the configured mode."""
    from qdrant_client import QdrantClient

    mode = config.QDRANT_MODE.lower()
    if mode == "memory":
        return QdrantClient(":memory:")
    if mode == "local":
        Path(config.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=config.QDRANT_PATH)
    kwargs: dict = {"url": config.QDRANT_URL}
    if config.QDRANT_API_KEY:
        kwargs["api_key"] = config.QDRANT_API_KEY
    return QdrantClient(**kwargs)


class Store:
    """Multi-tenant Qdrant store.

    Every public method that touches vectors takes ``user_id`` first. That is
    deliberate: this class cannot be called correctly without saying who is
    asking, so a missing tenancy check shows up as a TypeError, not a leak.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client = _build_client()
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        with self._lock:
            if not self._collection_exists():
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            # Payload indexes make the tenancy filter cheap instead of a full
            # scan, and are what let filtered deletes finish quickly on a large
            # collection. Embedded Qdrant ignores them (it always scans), so we
            # only ask for them on a real server.
            if config.QDRANT_MODE.lower() == "remote":
                for field in ("user_id", "document_id", "session_id"):
                    try:
                        self._client.create_payload_index(
                            collection_name=COLLECTION_NAME,
                            field_name=field,
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass  # already indexed
            self._dim = dim

    # ------------------------------------------------------------------
    # Filters — the only way this module ever reaches the collection
    # ------------------------------------------------------------------
    @staticmethod
    def _tenant_filter(user_id: str, document_ids: list[str] | None = None):
        """AND(user_id, [document_id IN …]). user_id is never optional."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        if not user_id:
            raise ValueError("user_id is required for every vector operation")
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if document_ids:
            must.append(
                FieldCondition(key="document_id",
                               match=MatchAny(any=list(document_ids)))
            )
        return Filter(must=must)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_document(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
        upload_time: float,
        session_id: str | None = None,
    ) -> int:
        """Upsert one document's chunks, each stamped with the owner's identity."""
        from qdrant_client.models import PointStruct

        if not len(chunks):
            return 0
        with self._lock:
            self._ensure_collection(int(vectors.shape[1]))

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "user_id": user_id,
                        "document_id": document_id,
                        "filename": filename,
                        "upload_time": upload_time,
                        "session_id": session_id,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
                for chunk, vec in zip(chunks, vectors)
            ]

            for i in range(0, len(points), 256):
                self._client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points[i:i + 256],
                    wait=True,   # never report "indexed" before it is durable
                )
            return len(points)

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------
    def delete_document(self, user_id: str, document_id: str) -> int:
        """Delete every chunk of `document_id` owned by `user_id`.

        Returns the number of points removed, and raises if any survive — a
        delete that silently does nothing is exactly the failure being fixed.
        """
        from qdrant_client.models import FilterSelector

        with self._lock:
            if not self._collection_exists():
                return 0

            flt = self._tenant_filter(user_id, [document_id])
            before = self._count(flt)
            if before == 0:
                return 0

            self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(filter=flt),
                wait=True,
            )
            after = self._count(flt)
            if after:
                raise RuntimeError(
                    f"Qdrant still holds {after} chunk(s) for {document_id} after "
                    "delete. Keeping the registry row: dropping it would make the "
                    "document unreachable but still searchable."
                )
            return before

    def delete_user(self, user_id: str) -> int:
        """Erase everything a user owns. Backs the 'delete my account' path."""
        from qdrant_client.models import FilterSelector

        with self._lock:
            if not self._collection_exists():
                return 0
            flt = self._tenant_filter(user_id)
            n = self._count(flt)
            if n:
                self._client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=FilterSelector(filter=flt),
                    wait=True,
                )
            return n

    def purge_orphans(self) -> int:
        """Drop points whose document_id has no owner row in SQLite.

        The repair pass for collections written by earlier versions: unowned
        points, points left by half-finished uploads, points whose registry
        entry was lost. Runs at startup; safe to run any time.
        """
        from qdrant_client.models import PointIdsList

        with self._lock:
            if not self._collection_exists():
                return 0
            known = db.all_document_ids()
            orphan_ids: list = []
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=512,
                    offset=offset,
                    with_payload=["document_id", "user_id"],
                    with_vectors=False,
                )
                for p in points:
                    payload = p.payload or {}
                    if (payload.get("document_id") not in known
                            or not payload.get("user_id")):
                        orphan_ids.append(p.id)
                if offset is None:
                    break

            for i in range(0, len(orphan_ids), 256):
                self._client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=orphan_ids[i:i + 256]),
                    wait=True,
                )
            return len(orphan_ids)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def dense_search(
        self,
        user_id: str,
        qvec: np.ndarray,
        document_ids: list[str] | None,
        k: int,
    ) -> list[tuple[dict, float]]:
        """Semantic search, hard-scoped to `user_id` by a server-side filter.

        Returns [(payload_dict, cosine_score)]. `document_ids` narrows further;
        it can never widen, because it is ANDed with the tenancy condition.
        """
        with self._lock:
            if not self._collection_exists():
                return []

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec.tolist(),
            query_filter=self._tenant_filter(user_id, document_ids),
            limit=k,
            with_payload=True,
        ).points
        return [(dict(r.payload or {}), float(r.score)) for r in results]

    def count_chunks(self, user_id: str, document_id: str | None = None) -> int:
        with self._lock:
            if not self._collection_exists():
                return 0
        ids = [document_id] if document_id else None
        return self._count(self._tenant_filter(user_id, ids))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collection_exists(self) -> bool:
        try:
            existing = {c.name for c in self._client.get_collections().collections}
        except Exception:
            return False
        return COLLECTION_NAME in existing

    def _count(self, flt) -> int:
        return self._client.count(
            collection_name=COLLECTION_NAME,
            count_filter=flt,
            exact=True,
        ).count
