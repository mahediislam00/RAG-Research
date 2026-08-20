"""Multi-model embedding cache for the Evaluation Lab.

The production `app/embeddings.py` caches exactly one model (whatever
`config.EMBED_MODEL` is). Sweeping the embedding axis means loading several
different models in the same process, so this module keeps its own
name -> SentenceTransformer cache, entirely separate from the production
embedder. Nothing here is imported by the production chat/upload path.
"""
from __future__ import annotations

import threading

import numpy as np

_models: dict[str, object] = {}
_lock = threading.Lock()

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# e5-family models expect "query: " / "passage: " prefixes rather than the
# bge-style instruction sentence; applying the wrong one just costs a little
# retrieval quality, so this is best-effort, not load-bearing.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _get(model_name: str):
    if model_name not in _models:
        with _lock:
            if model_name not in _models:
                from sentence_transformers import SentenceTransformer

                _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def _is_e5(model_name: str) -> bool:
    return "e5-" in model_name.lower()


def embed_passages(model_name: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim(model_name)), dtype=np.float32)
    model = _get(model_name)
    prefixed = [(_E5_PASSAGE_PREFIX + t) if _is_e5(model_name) else t for t in texts]
    vecs = model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_query(model_name: str, text: str) -> np.ndarray:
    model = _get(model_name)
    prefix = _E5_QUERY_PREFIX if _is_e5(model_name) else QUERY_PREFIX
    vec = model.encode(
        [prefix + text], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return vec[0].astype(np.float32)


def dim(model_name: str) -> int:
    return _get(model_name).get_sentence_embedding_dimension()


def preload(model_name: str) -> None:
    """Force-load a model (used to report a clean error before a run starts,
    rather than failing mid-sweep)."""
    _get(model_name)


def loaded_models() -> list[str]:
    return sorted(_models.keys())
