"""Ephemeral, in-process retrieval index for one experiment run, supporting
four retrieval modes:

- **dense**            — semantic search only (embedding cosine similarity).
- **sparse**            — BM25 keyword search only, over the full corpus.
- **hybrid_rrf**        — the production algorithm: dense candidates, BM25
                          re-scored over just those candidates, fused with
                          Reciprocal Rank Fusion. Mirrors
                          `app/retrieval.py::HybridRetriever` exactly, so a
                          run with "hybrid_rrf, no reranker, no noise, same
                          k/weights as config.py defaults" reproduces the
                          production pipeline's behaviour.
- **hybrid_weighted**   — an alternative fusion: dense and BM25 scores are
                          each min-max normalized to [0,1] over the current
                          candidate pool, then combined as a weighted sum
                          (`dense_weight * norm_dense + sparse_weight *
                          norm_bm25`). Unlike RRF, which only looks at rank
                          position, this lets the *margin* between a strong
                          and a weak match influence the fused score.

It never touches Qdrant or the production embedding cache: chunks and
vectors are rebuilt fresh, in memory, per run, using whatever
chunk/embedding config the run specifies.
"""
from __future__ import annotations

import random

import numpy as np
from rank_bm25 import BM25Okapi

from ..retrieval import HybridRetriever, tokenize

RETRIEVAL_MODES = [
    {"id": "hybrid_rrf", "label": "Hybrid — Reciprocal Rank Fusion (production default)"},
    {"id": "hybrid_weighted", "label": "Hybrid — weighted score fusion"},
    {"id": "dense", "label": "Dense (semantic) only"},
    {"id": "sparse", "label": "Sparse (BM25 keyword) only"},
    {"id": "mmr", "label": "Dense + MMR (diversity-aware reranking)"},
]


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class EphemeralIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        """`chunks[i]` corresponds to `vectors[i]`. Vectors must already be
        L2-normalized (embed_cache does this) so dot product == cosine."""
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32) if len(vectors) else vectors
        self._id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self._full_bm25: BM25Okapi | None = None  # built lazily, only for sparse-only mode

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    def _dense_search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.chunks) == 0 or k <= 0:
            return []
        scores = self.vectors @ qvec  # cosine, since both are L2-normalized
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def _bm25_over_candidates(self, query: str, candidate_idx: list[int],
                              k: int) -> list[tuple[int, float]]:
        """Same design as HybridRetriever._bm25_search: BM25 runs only over
        the dense candidate set, not the whole corpus, so both rankings
        operate on the same pool. Used by hybrid_rrf and hybrid_weighted."""
        if not candidate_idx:
            return []
        corpus = [tokenize(self.chunks[i]["text"]) for i in candidate_idx]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(query))
        k = min(k, len(candidate_idx))
        ranked_local = sorted(range(len(scores)), key=lambda j: -scores[j])[:k]
        return [(candidate_idx[j], float(scores[j])) for j in ranked_local if scores[j] > 0]

    def _full_corpus_bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Sparse-only mode has no dense stage to draw a candidate pool
        from, so it needs its own full-corpus BM25 index. Built once, on
        first use, and cached for the life of this index."""
        if self._full_bm25 is None:
            self._full_bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        scores = self._full_bm25.get_scores(tokenize(query))
        k = min(k, len(scores))
        ranked = sorted(range(len(scores)), key=lambda j: -scores[j])[:k]
        return [(j, float(scores[j])) for j in ranked if scores[j] > 0]

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        qvec: np.ndarray,
        dense_k: int,
        sparse_k: int,
        final_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
        retrieval_mode: str = "hybrid_rrf",
        mmr_lambda: float = 0.5,
    ) -> list[dict]:
        """Return up to `final_k` chunk dicts, ranked per `retrieval_mode`,
        each annotated with matched_by / dense_score / bm25_score /
        fused_score. Does not inject noise; call `inject_noise_into()` as a
        separate, explicit step (typically after any reranking) so the
        ordering of rerank-then-noise vs. noise-then-rerank is a pipeline
        decision, not something buried in the retriever."""
        if len(self.chunks) == 0:
            return []
        if retrieval_mode == "dense":
            return self._search_dense_only(qvec, final_k)
        if retrieval_mode == "sparse":
            return self._search_sparse_only(query, final_k)
        if retrieval_mode == "mmr":
            return self._search_mmr(qvec, dense_k, final_k, mmr_lambda)
        if retrieval_mode == "hybrid_weighted":
            return self._search_hybrid_weighted(query, qvec, dense_k, sparse_k, final_k,
                                                dense_weight, sparse_weight)
        return self._search_hybrid_rrf(query, qvec, dense_k, sparse_k, final_k, rrf_k,
                                       dense_weight, sparse_weight)

    def _annotate(self, idx: int, fused_score: float, dense_score: float,
                 bm25_score: float, matched_by: str) -> dict:
        c = dict(self.chunks[idx])
        c["fused_score"] = round(float(fused_score), 5)
        c["dense_score"] = round(float(dense_score), 4)
        c["bm25_score"] = round(float(bm25_score), 4)
        c["matched_by"] = matched_by
        c["injected_noise"] = False
        return c

    def _search_dense_only(self, qvec: np.ndarray, final_k: int) -> list[dict]:
        results = self._dense_search(qvec, final_k)
        return [self._annotate(i, s, s, 0.0, "semantic") for i, s in results]

    def _search_sparse_only(self, query: str, final_k: int) -> list[dict]:
        results = self._full_corpus_bm25_search(query, final_k)
        return [self._annotate(i, s, 0.0, s, "keyword") for i, s in results]

    def _search_mmr(self, qvec: np.ndarray, dense_k: int, final_k: int,
                    mmr_lambda: float = 0.5) -> list[dict]:
        """Maximal Marginal Relevance (Carbonell & Goldstein, 1998): starting
        from the dense candidate pool, iteratively picks the item maximizing
        `mmr_lambda * relevance_to_query - (1 - mmr_lambda) * max_similarity
        _to_already_selected`, trading relevance off against redundancy.
        mmr_lambda=1.0 reduces to plain dense ranking; mmr_lambda=0.0
        maximizes diversity, ignoring relevance after the first pick."""
        candidates = self._dense_search(qvec, dense_k)
        if not candidates:
            return []
        relevance = {i: s for i, s in candidates}
        remaining = [i for i, _ in candidates]

        selected: list[int] = []
        while remaining and len(selected) < final_k:
            best_idx, best_score = None, float("-inf")
            for i in remaining:
                if selected:
                    sim_to_selected = max(float(self.vectors[i] @ self.vectors[j]) for j in selected)
                else:
                    sim_to_selected = 0.0
                mmr_score = mmr_lambda * relevance[i] - (1 - mmr_lambda) * sim_to_selected
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [self._annotate(i, relevance[i], relevance[i], 0.0, "semantic+mmr") for i in selected]

    def _search_hybrid_rrf(self, query: str, qvec: np.ndarray, dense_k: int, sparse_k: int,
                           final_k: int, rrf_k: int, dense_weight: float,
                           sparse_weight: float) -> list[dict]:
        dense_raw = self._dense_search(qvec, dense_k)
        dense_idx = [i for i, _ in dense_raw]
        dense_scores = {i: s for i, s in dense_raw}

        sparse_raw = self._bm25_over_candidates(query, dense_idx, sparse_k)
        sparse_idx = [i for i, _ in sparse_raw]
        sparse_scores = {i: s for i, s in sparse_raw}

        fused = HybridRetriever._rrf(
            [dense_idx, sparse_idx], [dense_weight, sparse_weight], rrf_k
        )
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:final_k]

        results: list[dict] = []
        for idx, fused_score in ranked:
            matched_by = ("both" if idx in dense_scores and idx in sparse_scores
                         else "semantic" if idx in dense_scores else "keyword")
            results.append(self._annotate(idx, fused_score, dense_scores.get(idx, 0.0),
                                          sparse_scores.get(idx, 0.0), matched_by))
        return results

    def _search_hybrid_weighted(self, query: str, qvec: np.ndarray, dense_k: int, sparse_k: int,
                                final_k: int, dense_weight: float,
                                sparse_weight: float) -> list[dict]:
        dense_raw = self._dense_search(qvec, dense_k)
        dense_idx = [i for i, _ in dense_raw]
        dense_scores = {i: s for i, s in dense_raw}

        sparse_raw = self._bm25_over_candidates(query, dense_idx, sparse_k)
        sparse_scores = {i: s for i, s in sparse_raw}

        candidate_idx = list(dict.fromkeys(dense_idx + list(sparse_scores.keys())))
        dense_norm = dict(zip(candidate_idx,
                              _minmax([dense_scores.get(i, 0.0) for i in candidate_idx])))
        sparse_norm = dict(zip(candidate_idx,
                               _minmax([sparse_scores.get(i, 0.0) for i in candidate_idx])))

        combined = {
            i: dense_weight * dense_norm.get(i, 0.0) + sparse_weight * sparse_norm.get(i, 0.0)
            for i in candidate_idx
        }
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:final_k]

        results: list[dict] = []
        for idx, fused_score in ranked:
            matched_by = ("both" if idx in dense_scores and idx in sparse_scores
                         else "semantic" if idx in dense_scores else "keyword")
            results.append(self._annotate(idx, fused_score, dense_scores.get(idx, 0.0),
                                          sparse_scores.get(idx, 0.0), matched_by))
        return results

    # ------------------------------------------------------------------
    def inject_noise_into(self, results: list[dict], noise_pct: float,
                          rng: random.Random | None = None) -> list[dict]:
        """Replace the tail (lowest-ranked) slots of an already-ranked
        result list with random unrelated chunks from the corpus. Models
        the "some retrieved evidence is irrelevant/noisy" condition studied
        in the input-quality literature, without needing a separately
        curated noisy corpus. `results` may come from fusion alone or from
        post-rerank output; either way this is a pure post-processing step."""
        if noise_pct <= 0 or not results or len(self.chunks) <= len(results):
            return results
        rng = rng or random
        n_swap = max(1, round(len(results) * noise_pct))
        n_swap = min(n_swap, len(results))

        used_idx = {self._id_to_idx[c["chunk_id"]] for c in results if c["chunk_id"] in self._id_to_idx}
        pool = [i for i in range(len(self.chunks)) if i not in used_idx]
        if not pool:
            return results
        rng.shuffle(pool)
        swap_targets = pool[:n_swap]

        # Replace the LOWEST-ranked n_swap slots (tail of the list), which is
        # the realistic failure mode: noise crowds out marginal hits, not the
        # single best match.
        out = list(results)
        for offset, new_idx in enumerate(swap_targets):
            pos = len(out) - 1 - offset
            if pos < 0:
                break
            c = dict(self.chunks[new_idx])
            c["fused_score"] = 0.0
            c["dense_score"] = 0.0
            c["bm25_score"] = 0.0
            c["matched_by"] = "noise"
            c["injected_noise"] = True
            out[pos] = c
        return out

    # ------------------------------------------------------------------
    def all_texts(self) -> list[str]:
        return [c["text"] for c in self.chunks]
