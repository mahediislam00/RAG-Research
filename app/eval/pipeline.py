"""Orchestrates one experiment run: rebuild the pipeline under a RunConfig,
answer every eval question through it, compute metrics per question, persist
everything, and return a summary. `run_sweep()` just calls `run_experiment()`
once per value of the swept axis and tags the results with a shared
`sweep_id` so the UI can plot metric-vs-configuration-value.
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path

from .. import config as app_config
from .. import db
from ..ingestion import extract_pages
from ..config import ChunkConfig
from . import chunkers, embed_cache, generation, metrics, reranker, store
from .hybrid_index import EphemeralIndex
from .schema import RunConfig

log = logging.getLogger("raglab.eval")


class PipelineError(RuntimeError):
    pass


# ==========================================================================
# Corpus construction (reuses each document's already-stored file, so no
# separate "eval corpus" upload path is needed)
# ==========================================================================

def _load_corpus_chunks(user_id: str, doc_ids: list[str], run_config: RunConfig) -> list[dict]:
    chunk_cfg = ChunkConfig(
        target_tokens=run_config.chunk_target_tokens,
        overlap_tokens=run_config.chunk_overlap_tokens,
        min_tokens=run_config.chunk_min_tokens,
    )
    all_chunks: list[dict] = []
    for doc_id in doc_ids:
        row = db.get_document(doc_id, user_id)
        if not row:
            continue
        path = Path(row["stored_path"])
        if not path.exists():
            log.warning("stored file missing for doc %s (%s)", doc_id, path)
            continue
        try:
            pages = [p for p in extract_pages(path) if p.text and p.text.strip()]
            if not pages:
                continue
            chunks = chunkers.chunk_document(
                pages, filename=row["filename"], doc_id=doc_id, chunk_config=chunk_cfg,
                strategy=run_config.chunking_strategy,
                embed_model=run_config.embed_model if run_config.chunking_strategy == "semantic" else None,
            )
        except Exception as e:
            log.warning("failed to chunk doc %s: %s", doc_id, e)
            continue
        for c in chunks:
            d = c.to_dict()
            d["chunk_id"] = f"{doc_id}::{c.chunk_index}"
            all_chunks.append(d)
    return all_chunks


def build_index(user_id: str, doc_ids: list[str], run_config: RunConfig) -> EphemeralIndex:
    chunks = _load_corpus_chunks(user_id, doc_ids, run_config)
    if not chunks:
        raise PipelineError(
            "No usable text found across the selected documents for this "
            "chunk configuration. Pick different documents or a smaller "
            "min_tokens value."
        )
    texts = [c["text"] for c in chunks]
    try:
        vectors = embed_cache.embed_passages(run_config.embed_model, texts)
    except ModuleNotFoundError as e:
        raise PipelineError(
            f"The embedding backend is not installed ({e}). Run "
            "`pip install sentence-transformers` in this environment."
        )
    except Exception as e:
        raise PipelineError(
            f"Could not load or run embedding model '{run_config.embed_model}': {e}. "
            "Check the model id and that this machine has internet access to "
            "download it on first use."
        )
    return EphemeralIndex(chunks, vectors)


# ==========================================================================
# Per-question execution
# ==========================================================================

def _llm_judge_relevant_ids(question: str, passages: list[dict], model_id: str) -> set[str] | None:
    """Label each RETRIEVED passage (not the whole corpus -- that would be
    one inference call per chunk per question, too costly) as relevant or
    not via the generator LLM. Returns None if every judgment call failed,
    so the caller falls back to the silver proxy rather than reporting an
    empty relevant-set as if it were meaningful."""
    judged: set[str] = set()
    any_success = False
    for p in passages:
        verdict = generation.judge_relevance(question, p.get("text", ""), model_id)
        if verdict is None:
            continue
        any_success = True
        if verdict:
            judged.add(p["chunk_id"])
    return judged if any_success else None


def _answer_question(index: EphemeralIndex, question_text: str,
                     run_config: RunConfig, rng: random.Random) -> dict:
    """Retrieval -> (optional) rerank -> noise injection -> generation ->
    metric computation, for one question. Never raises: failures land in the
    returned dict's `error` field."""
    t_retr0 = time.perf_counter()
    try:
        qvec = embed_cache.embed_query(run_config.embed_model, question_text)
    except Exception as e:
        return {"question": question_text, "error": f"embedding failed: {e}"}

    use_reranker = run_config.reranker_model and run_config.reranker_model != "none"
    candidate_pool = max(run_config.rerank_pool, run_config.final_k) if use_reranker else run_config.final_k

    passages = index.search(
        question_text, qvec,
        dense_k=run_config.dense_k, sparse_k=run_config.sparse_k,
        final_k=candidate_pool, rrf_k=run_config.rrf_k,
        dense_weight=run_config.dense_weight, sparse_weight=run_config.sparse_weight,
        retrieval_mode=run_config.retrieval_mode, mmr_lambda=run_config.mmr_lambda,
    )
    retrieval_ms = (time.perf_counter() - t_retr0) * 1000

    rerank_ms = 0.0
    if use_reranker and passages:
        t_rr0 = time.perf_counter()
        try:
            passages = reranker.rerank(run_config.reranker_model, question_text,
                                       passages, top_n=run_config.final_k)
        except Exception as e:
            return {"question": question_text, "error": f"reranking failed: {e}"}
        rerank_ms = (time.perf_counter() - t_rr0) * 1000
    else:
        passages = passages[:run_config.final_k]

    passages = index.inject_noise_into(passages, run_config.noise_pct, rng)

    if not passages:
        return {"question": question_text, "error": "no passages retrieved",
               "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms}

    messages = _build_eval_messages(question_text, passages)
    gen = generation.complete(messages, run_config.llm_model_id,
                              temperature=run_config.temperature,
                              max_tokens=run_config.max_tokens)

    result = {
        "question": question_text,
        "answer": gen.text,
        "passages": passages,
        "retrieval_ms": round(retrieval_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "generation_ms": round(gen.latency_ms, 2),
        "total_ms": round(retrieval_ms + rerank_ms + gen.latency_ms, 2),
        "model_id": gen.model_id,
    }
    if gen.error:
        result["error"] = gen.error
    return result


def _build_eval_messages(question: str, passages: list[dict]) -> list[dict]:
    """A lean, mode-agnostic Q&A prompt for evaluation (mirrors the app's QA
    prompt's grounding rule without the government-contracting-specific
    drafting instructions, since eval questions may not be procurement
    questions)."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        loc = f"p.{p.get('page_start')}-{p.get('page_end')}"
        blocks.append(f"[{i}] {p.get('filename', '')} {loc}\n{p.get('text', '')}")
    context = "\n\n---\n\n".join(blocks)
    system = (
        "Answer strictly using the SOURCE PASSAGES below. Cite passages as "
        "[filename p.X]. If the passages do not contain the answer, say so "
        "plainly rather than inventing one."
    )
    user = f"SOURCE PASSAGES:\n\n{context}\n\n-----\n\nQUESTION:\n{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ==========================================================================
# Metric computation for one answered question
# ==========================================================================

def _compute_metrics(question_obj: dict, result: dict, index: EphemeralIndex,
                     run_config: RunConfig) -> dict:
    if result.get("error") and "answer" not in result:
        return {"error": result["error"],
               "retrieval_ms": result.get("retrieval_ms"),
               "rerank_ms": result.get("rerank_ms")}

    q_text = question_obj["question"]
    passages = result.get("passages", [])
    retrieved_ids = [p["chunk_id"] for p in passages]
    retrieved_texts = [p["text"] for p in passages]
    answer = result.get("answer", "") or ""

    gold_ids = metrics.gold_relevant_ids(question_obj, index.chunks)
    relevance_source = "gold"
    relevant_ids = gold_ids
    if relevant_ids is None and run_config.use_llm_judge and not result.get("error"):
        judged = _llm_judge_relevant_ids(q_text, passages, run_config.llm_model_id)
        if judged is not None:
            relevant_ids = judged
            relevance_source = "llm_judge"
    if relevant_ids is None:
        relevant_ids = metrics.silver_relevant_ids(
            q_text, index.chunks, run_config.embed_model, run_config.silver_relevance_k
        )
        relevance_source = "silver"

    m: dict = {
        "relevance_source": relevance_source,
        "recall_at_k": metrics.recall_at_k(retrieved_ids, relevant_ids),
        "precision_at_k": metrics.precision_at_k(retrieved_ids, relevant_ids),
        "ndcg_at_k": metrics.ndcg_at_k(retrieved_ids, relevant_ids),
        "mrr": metrics.mrr(retrieved_ids, relevant_ids),
        "context_precision": metrics.context_precision(retrieved_ids, relevant_ids),
        "diversity": metrics.diversity(retrieved_texts, run_config.embed_model) if retrieved_texts else float("nan"),
    }

    expected_answer = question_obj.get("expected_answer")
    if expected_answer:
        m["context_recall"] = metrics.context_recall_vs_answer(
            expected_answer, retrieved_texts, run_config.embed_model
        )
        m["keyword_f1"] = metrics.keyword_f1(answer, expected_answer)
    else:
        m["context_recall"] = m["recall_at_k"]  # documented fallback
        m["keyword_f1"] = float("nan")

    if answer.strip() and retrieved_texts:
        faith, supported = metrics.faithfulness(
            answer, retrieved_texts, run_config.embed_model, mode=run_config.faithfulness_mode
        )
        m["faithfulness"] = faith
        m["faithfulness_supported_frac"] = supported
        m["answer_relevance"] = metrics.answer_relevance(q_text, answer, run_config.embed_model)
    else:
        m["faithfulness"] = float("nan")
        m["faithfulness_supported_frac"] = float("nan")
        m["answer_relevance"] = float("nan")

    m["composite_score"] = metrics.composite_score(m)
    m["model_id"] = result.get("model_id")
    m["retrieval_ms"] = result.get("retrieval_ms")
    m["rerank_ms"] = result.get("rerank_ms")
    m["generation_ms"] = result.get("generation_ms")
    m["total_ms"] = result.get("total_ms")
    if result.get("error"):
        m["generation_error"] = result["error"]
    return m


# ==========================================================================
# Public entry points
# ==========================================================================

def run_experiment(user_id: str, run_config: RunConfig, doc_ids: list[str],
                   questions: list[dict], dataset_id: str | None = None,
                   seed: int = 0) -> dict:
    """Run one configuration against one question set. Persists the run and
    every per-question result, returns the run summary dict."""
    store.init_schema()
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    label = run_config.label or run_config.auto_label()
    store.create_run(run_id, user_id, run_config.sweep_id, run_config.sweep_axis,
                     label, run_config.to_dict(), doc_ids, dataset_id)

    rng = random.Random(seed)
    try:
        index = build_index(user_id, doc_ids, run_config)
    except PipelineError as e:
        store.finish_run(run_id, user_id, {}, status="error", error=str(e))
        raise

    per_question: list[dict] = []
    for q in questions:
        q_obj = q if isinstance(q, dict) else q.to_dict()
        result = _answer_question(index, q_obj["question"], run_config, rng)
        m = _compute_metrics(q_obj, result, index, run_config)
        per_question.append(m)
        store.add_run_item(
            f"item_{uuid.uuid4().hex[:12]}", run_id, user_id, q_obj["question"],
            q_obj.get("expected_answer"), result.get("answer"), m,
            result.get("passages", []), result.get("error"),
        )

    summary = metrics.aggregate_run(per_question)
    summary["corpus_size"] = len(index)
    store.finish_run(run_id, user_id, summary, status="done")
    return {"run_id": run_id, "label": label, "summary": summary,
           "config": run_config.to_dict()}


def run_sweep(user_id: str, base_config: RunConfig, axis: str, values: list,
             doc_ids: list[str], questions: list[dict],
             dataset_id: str | None = None) -> dict:
    """Run one experiment per value in `values`, varying only `axis` and
    holding every other field of `base_config` fixed. Returns the shared
    sweep_id plus each run's summary, in the order the values were given."""
    if axis == "none" or not values:
        cfg = _clone_with(base_config, {})
        cfg.sweep_id = f"sweep_{uuid.uuid4().hex[:10]}"
        cfg.sweep_axis = "none"
        result = run_experiment(user_id, cfg, doc_ids, questions, dataset_id)
        return {"sweep_id": cfg.sweep_id, "axis": "none", "runs": [result]}

    sweep_id = f"sweep_{uuid.uuid4().hex[:10]}"
    runs = []
    for v in values:
        cfg = _clone_with(base_config, {axis: v})
        cfg.sweep_id = sweep_id
        cfg.sweep_axis = axis
        cfg.label = f"{axis}={v}"
        try:
            result = run_experiment(user_id, cfg, doc_ids, questions, dataset_id)
            result["sweep_value"] = v
            runs.append(result)
        except PipelineError as e:
            runs.append({"error": str(e), "sweep_value": v, "config": cfg.to_dict()})
    return {"sweep_id": sweep_id, "axis": axis, "runs": runs}


def _clone_with(base: RunConfig, overrides: dict) -> RunConfig:
    d = base.to_dict()
    d.update(overrides)
    d.pop("sweep_id", None)
    d.pop("sweep_axis", None)
    d.pop("label", None)
    from .schema import RunConfig as _RC
    return _RC(**d)
