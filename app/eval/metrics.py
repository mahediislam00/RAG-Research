"""Retrieval-layer, generation-layer, and latency metrics.

Design notes (read before trusting a number blindly):

* Recall@k / Precision@k / NDCG@k / MRR need a "relevant set" per question.
  If the eval dataset supplied gold `relevant_sections`, that is used
  (`relevance_source = "gold"`). Otherwise a silver proxy is built by
  embedding the full corpus with the SAME embedding model used for
  retrieval and taking the top `silver_relevance_k` chunks by cosine
  similarity to the question (`relevance_source = "silver"`). Silver labels
  are a proxy, not ground truth, and every run result is tagged with which
  one was used so this is never silently conflated with a gold benchmark.

* Faithfulness and Answer Relevance are local, free re-implementations of
  the RAGAS ideas (not the `ragas` package, and not identical to it):
    - faithfulness (cosine mode): per-sentence max cosine similarity to any
      retrieved chunk, averaged.
    - faithfulness (nli mode): per-sentence entailment probability against
      its best-matching retrieved chunk, via a local cross-encoder NLI
      model, averaged.
    - answer relevance: cosine similarity between the question and the
      generated answer (RAGAS's own method instead generates reverse
      questions from the answer and averages their similarity to the
      original question; this is a simplified, single-pass proxy for that).
  Both are reported for transparency about being proxies, consistent with
  the literature's own caution that LLM-/embedding-based RAG metrics do not
  always track human judgment.
"""
from __future__ import annotations

import math
import re
import statistics
import threading
from collections import Counter

from . import embed_cache

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    pieces = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


def _cos(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b))  # both already L2-normalized


# ==========================================================================
# Relevant-set construction
# ==========================================================================

def gold_relevant_ids(eval_question, chunks: list[dict]) -> set[str] | None:
    """Match `relevant_sections` substrings against each chunk's `section`
    or `text`. Returns None (not an empty set) when no gold labels were
    supplied, so callers can distinguish "no gold" from "gold says nothing
    is relevant"."""
    hints = getattr(eval_question, "relevant_sections", None) or (
        eval_question.get("relevant_sections") if isinstance(eval_question, dict) else None
    )
    if not hints:
        return None
    hints = [h.strip().lower() for h in hints if h and h.strip()]
    if not hints:
        return None
    out: set[str] = set()
    for c in chunks:
        hay = f"{c.get('section', '')} {c.get('text', '')}".lower()
        if any(h in hay for h in hints):
            out.add(c["chunk_id"])
    return out


def silver_relevant_ids(question: str, chunks: list[dict], embed_model: str,
                        top_k: int) -> set[str]:
    """Embedding-similarity proxy relevant set, built against the FULL
    corpus (not just what was retrieved), so it is an independent yardstick
    for judging the retriever rather than something the retriever itself
    produced."""
    if not chunks:
        return set()
    qvec = embed_cache.embed_query(embed_model, question)
    vecs = embed_cache.embed_passages(embed_model, [c["text"] for c in chunks])
    sims = vecs @ qvec
    k = min(top_k, len(chunks))
    top_idx = sorted(range(len(chunks)), key=lambda i: -sims[i])[:k]
    return {chunks[i]["chunk_id"] for i in top_idx}


# ==========================================================================
# Retrieval-layer metrics
# ==========================================================================

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return float("nan")
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return float("nan")
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(retrieved_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids or not retrieved_ids:
        return float("nan")
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_hits = min(len(relevant_ids), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else float("nan")


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return float("nan")
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """RAGAS-style order-aware precision: average of precision@i taken at
    each rank where a relevant item occurs, normalized by the number of
    relevant items found (0 if none found). Rewards relevant passages
    appearing earlier, unlike plain Precision@k."""
    if not relevant_ids or not retrieved_ids:
        return float("nan")
    hits = 0
    precisions = []
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / hits


def context_recall_vs_answer(expected_answer: str, retrieved_texts: list[str],
                             embed_model: str) -> float:
    """When a gold `expected_answer` is supplied: fraction of the expected
    answer's sentences that are attributable (by cosine similarity) to some
    retrieved chunk. Falls back to NaN when there is no expected answer;
    callers should use Recall@k against the relevant set instead in that
    case."""
    sentences = split_sentences(expected_answer)
    if not sentences or not retrieved_texts:
        return float("nan")
    import numpy as np

    sent_vecs = embed_cache.embed_passages(embed_model, sentences)
    ctx_vecs = embed_cache.embed_passages(embed_model, retrieved_texts)
    supported = 0
    for sv in sent_vecs:
        sims = ctx_vecs @ sv
        if float(np.max(sims)) >= 0.55:
            supported += 1
    return supported / len(sentences)


# ==========================================================================
# Generation-layer metrics
# ==========================================================================

_nli_models: dict[str, object] = {}
_nli_lock = threading.Lock()


def _get_nli(model_name: str = "cross-encoder/nli-deberta-v3-small"):
    if model_name not in _nli_models:
        with _nli_lock:
            if model_name not in _nli_models:
                from sentence_transformers import CrossEncoder

                _nli_models[model_name] = CrossEncoder(model_name)
    return _nli_models[model_name]


def _entailment_index(model) -> int:
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)
    return 1  # sentence-transformers' documented convention for this model family


def faithfulness(answer: str, retrieved_texts: list[str], embed_model: str,
                 mode: str = "cosine") -> tuple[float, float]:
    """Returns (score, supported_fraction). `score` is the mean per-sentence
    support signal (cosine similarity, or NLI entailment probability);
    `supported_fraction` is the share of sentences clearing a 0.5 threshold,
    a more interpretable secondary stat."""
    sentences = split_sentences(answer)
    if not sentences or not retrieved_texts:
        return float("nan"), float("nan")

    import numpy as np

    if mode == "nli":
        nli = _get_nli()
        ent_idx = _entailment_index(nli)
        scores = []
        for s in sentences:
            pairs = [(ctx, s) for ctx in retrieved_texts]
            probs = nli.predict(pairs, apply_softmax=True)
            best = max(float(p[ent_idx]) for p in probs)
            scores.append(best)
    else:
        ctx_vecs = embed_cache.embed_passages(embed_model, retrieved_texts)
        sent_vecs = embed_cache.embed_passages(embed_model, sentences)
        scores = []
        for sv in sent_vecs:
            sims = ctx_vecs @ sv
            # rescale cosine [-1,1] -> [0,1] for consistency with other metrics
            scores.append((float(np.max(sims)) + 1) / 2)

    mean_score = sum(scores) / len(scores)
    supported_frac = sum(1 for s in scores if s >= 0.5) / len(scores)
    return mean_score, supported_frac


def answer_relevance(question: str, answer: str, embed_model: str) -> float:
    if not answer.strip():
        return float("nan")
    qvec = embed_cache.embed_query(embed_model, question)
    avec = embed_cache.embed_passages(embed_model, [answer])[0]
    cos = _cos(qvec, avec)
    return (cos + 1) / 2  # rescale to [0,1]


def diversity(retrieved_texts: list[str], embed_model: str) -> float:
    """1 - average pairwise cosine similarity among the retrieved chunks'
    embeddings. Higher means the retrieved set covers more distinct
    content; lower means the top-k are largely redundant with each other
    (e.g. near-duplicate chunks from overlapping windows, or a retrieval
    mode/config that keeps surfacing the same passage restated). Pairs
    naturally with the "mmr" retrieval mode, which explicitly optimizes
    for this."""
    if len(retrieved_texts) < 2:
        return float("nan")
    vecs = embed_cache.embed_passages(embed_model, retrieved_texts)
    n = len(vecs)
    sims = [float(vecs[i] @ vecs[j]) for i in range(n) for j in range(i + 1, n)]
    if not sims:
        return float("nan")
    avg_sim = sum(sims) / len(sims)
    return 1 - avg_sim


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def keyword_f1(answer: str, expected_answer: str) -> float:
    """SQuAD-style token F1 between the generated answer and a gold
    reference. Only meaningful when `expected_answer` was supplied."""
    if not expected_answer or not expected_answer.strip():
        return float("nan")
    pred = _normalize_tokens(answer)
    gold = _normalize_tokens(expected_answer)
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


# ==========================================================================
# Aggregation
# ==========================================================================

_METRIC_KEYS = [
    "recall_at_k", "precision_at_k", "ndcg_at_k", "mrr", "context_precision",
    "context_recall", "diversity", "faithfulness", "faithfulness_supported_frac",
    "answer_relevance", "keyword_f1", "composite_score",
    "retrieval_ms", "rerank_ms", "generation_ms", "total_ms",
]


def _safe_stats(values: list[float]) -> dict:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.fmean(clean), 4),
        "median": round(statistics.median(clean), 4),
        "std": round(statistics.pstdev(clean), 4) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def composite_score(metrics: dict) -> float:
    """A convenience aggregate (NOT the official RAGAS score): mean of
    context precision, recall@k, faithfulness, and answer relevance, when
    each is available."""
    parts = [metrics.get(k) for k in
             ("context_precision", "recall_at_k", "faithfulness", "answer_relevance")]
    parts = [p for p in parts if p is not None and not (isinstance(p, float) and math.isnan(p))]
    if not parts:
        return float("nan")
    return sum(parts) / len(parts)


def aggregate_run(per_question: list[dict]) -> dict:
    """per_question: list of per-question metric dicts (as produced by the
    pipeline). Returns {metric_name: {mean, median, std, n}} plus an
    `errors` count for questions that failed generation/retrieval."""
    out = {}
    for key in _METRIC_KEYS:
        out[key] = _safe_stats([q.get(key) for q in per_question])
    out["errors"] = sum(1 for q in per_question if q.get("error"))
    out["n_questions"] = len(per_question)
    return out
