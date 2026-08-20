"""Local cross-encoder reranking for the Evaluation Lab.

Runs entirely on-device via `sentence_transformers.CrossEncoder` (already a
transitive dependency of sentence-transformers, so no new package is
required). This is the RQ2 "reranking" axis: on/off, and which cross-encoder,
each measured against retrieval-layer and latency metrics.
"""
from __future__ import annotations

import threading

_models: dict[str, object] = {}
_lock = threading.Lock()


def _get(model_name: str):
    if model_name not in _models:
        with _lock:
            if model_name not in _models:
                from sentence_transformers import CrossEncoder

                _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


def rerank(model_name: str, query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Score each chunk's `text` against the query and return the top_n,
    each annotated with `rerank_score`. `chunks` order/content otherwise
    unchanged. If `model_name` is "none" or empty, this is a no-op."""
    if not model_name or model_name == "none" or not chunks:
        return chunks[:top_n]
    model = _get(model_name)
    pairs = [(query, c.get("text", "")) for c in chunks]
    scores = model.predict(pairs)
    scored = sorted(
        zip(chunks, scores), key=lambda cs: -float(cs[1])
    )
    out = []
    for c, s in scored[:top_n]:
        c2 = dict(c)
        c2["rerank_score"] = round(float(s), 5)
        out.append(c2)
    return out


def preload(model_name: str) -> None:
    if model_name and model_name != "none":
        _get(model_name)
