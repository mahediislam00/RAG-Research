"""Local semantic embeddings.

Runs entirely on your machine via sentence-transformers, so it never consumes
HuggingFace inference credits — only the chat LLM does. This keeps semantic
search free and always available even when every chat model is on cooldown.
"""
from __future__ import annotations

import threading

import numpy as np

from .config import EMBED_MODEL, EMBED_QUERY_PREFIX

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_passages(texts: list[str], batch_size: int = 64,
                   progress_cb=None) -> np.ndarray:
    """Embed document chunks. Returns L2-normalized float32 vectors.

    If ``progress_cb`` is given it is called as ``progress_cb(done, total)``
    after each batch so callers (e.g. the upload endpoint) can report indexing
    progress to the UI.
    """
    if not texts:
        return np.zeros((0, dim()), dtype=np.float32)
    model = _get_model()
    total = len(texts)
    if progress_cb is None:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    # Manual batching so we can emit progress between batches.
    out: list[np.ndarray] = []
    done = 0
    progress_cb(0, total)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out.append(vecs.astype(np.float32))
        done += len(batch)
        progress_cb(done, total)
    return np.vstack(out)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query (bge models want an instruction prefix)."""
    model = _get_model()
    vec = model.encode(
        [EMBED_QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec[0].astype(np.float32)


def dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
