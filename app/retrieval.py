"""Hybrid retrieval = BM25 (keyword) + Qdrant dense search (semantic), fused.

Why hybrid matters for government contracting:
  * Dense search captures *meaning* — "past performance" matches "prior
    contract experience" even with no shared words.
  * BM25 captures *exact tokens* — clause numbers (FAR 52.219-14), CLINs,
    section labels (Section L, C.3.1), agency acronyms. Dense models routinely
    blur these; BM25 nails them.
Fusing both with Reciprocal Rank Fusion (RRF) keeps a passage that either method
ranks highly, which is exactly the recall profile a solicitation needs.

Tenancy note: `retrieve()` takes a user_id and does nothing with it except hand
it to the store, which turns it into a server-side Qdrant filter. BM25 then runs
only over the passages that filter returned, so the keyword half of the hybrid
inherits the same scope — it never sees another tenant's text, and cannot
resurrect a deleted document.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .config import RETRIEVAL
from .embeddings import embed_query
from .vectorstore import Store

# Keep dotted/hyphenated codes intact: "52.219-14", "C.3.1", "CLIN-0001".
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class HybridRetriever:
    """BM25 + Qdrant dense search fused with RRF."""

    def __init__(self, store: Store) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # BM25 over the (already tenant-scoped) dense candidates
    # ------------------------------------------------------------------
    def _bm25_search(self, query: str, candidates: list[dict],
                     k: int) -> list[tuple[int, float]]:
        if not candidates:
            return []
        corpus = [tokenize(c["text"]) for c in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(query))
        k = min(k, len(candidates))
        ranked = sorted(range(len(scores)), key=lambda j: -scores[j])[:k]
        return [(j, float(scores[j])) for j in ranked if scores[j] > 0]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion (positional ids — no id() aliasing games)
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf(ranked_lists: list[list[int]], weights: list[float],
             rrf_k: int) -> dict[int, float]:
        fused: dict[int, float] = {}
        for lst, w in zip(ranked_lists, weights):
            for rank, idx in enumerate(lst):
                fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + rank + 1)
        return fused

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(self, user_id: str, query: str,
                 document_ids: list[str] | None = None) -> list[dict]:
        qvec = embed_query(query)

        # --- dense (Qdrant, filtered to this user) ---
        dense_raw = self.store.dense_search(
            user_id, qvec, document_ids, RETRIEVAL.dense_k
        )
        if not dense_raw:
            return []

        chunks: list[dict] = [payload for payload, _ in dense_raw]
        dense_scores: dict[int, float] = {i: s for i, (_, s) in enumerate(dense_raw)}
        dense_order: list[int] = list(range(len(chunks)))

        # --- sparse (BM25 over the same candidate set) ---
        sparse_raw = self._bm25_search(query, chunks, RETRIEVAL.sparse_k)
        sparse_scores: dict[int, float] = {i: s for i, s in sparse_raw}
        sparse_order: list[int] = [i for i, _ in sparse_raw]

        # --- RRF fusion ---
        fused = self._rrf(
            [dense_order, sparse_order],
            [RETRIEVAL.dense_weight, RETRIEVAL.sparse_weight],
            RETRIEVAL.rrf_k,
        )
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[: RETRIEVAL.final_k]

        results: list[dict] = []
        for idx, fused_score in ranked:
            c = dict(chunks[idx])
            c["fused_score"] = round(fused_score, 5)
            c["dense_score"] = round(dense_scores.get(idx, 0.0), 4)
            c["bm25_score"] = round(sparse_scores.get(idx, 0.0), 4)
            c["matched_by"] = (
                "both" if idx in dense_scores and idx in sparse_scores
                else "semantic" if idx in dense_scores
                else "keyword"
            )
            results.append(c)
        return results
