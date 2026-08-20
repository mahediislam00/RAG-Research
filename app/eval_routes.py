"""REST API for the Evaluation Lab.

Mounted into `app/main.py` via `app.include_router(eval_routes.router)`.
Every route depends on `current_user`, exactly like the rest of the app, and
every store/pipeline call is scoped to that user's own documents and runs —
the Evaluation Lab reuses the same tenancy discipline as chat and documents,
it does not introduce a parallel authorization path.
"""
from __future__ import annotations

import csv
import io
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config as app_config, db
from .auth import Principal, current_user
from .eval import store as eval_store
from .eval.schema import (
    CHUNKING_PRESETS, EMBEDDING_PRESETS, FAITHFULNESS_MODES, OPENROUTER_MODEL_PRESETS,
    RERANKER_PRESETS, SWEEPABLE_AXES, TOPK_PRESETS, parse_run_config,
)
from .eval.chunkers import CHUNKING_STRATEGIES
from .eval.hybrid_index import RETRIEVAL_MODES
from .eval.pipeline import PipelineError, run_experiment, run_sweep

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

# Idempotent (CREATE TABLE IF NOT EXISTS) and cheap; safe to run at import
# time regardless of whether app/main.py's own startup handler has run yet,
# since db.connect() lazily creates the shared SQLite connection on first use.
eval_store.init_schema()


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None.

    Metrics that could not be computed for a given question (e.g. keyword_f1
    with no gold answer) are stored as float('nan') so aggregation can filter
    them out cleanly. Starlette's JSONResponse enforces strict JSON (no NaN
    literal), so anything containing a raw NaN has to be sanitized before it
    is returned over HTTP -- this walks dicts/lists/tuples and does that."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --- options for populating the UI ---------------------------------------
@router.get("/options")
def options(user: Principal = Depends(current_user)):
    hf_models = [
        {"id": f"hf:{m.id}", "label": m.label, "provider": "hf", "paid": False}
        for m in app_config.MODEL_CHAIN
    ]
    openrouter_models = [
        {"id": p["id"], "label": p["label"], "provider": "openrouter", "paid": True}
        for p in OPENROUTER_MODEL_PRESETS
    ]
    return {
        "embedding_presets": EMBEDDING_PRESETS,
        "reranker_presets": RERANKER_PRESETS,
        "chunking_presets": CHUNKING_PRESETS,
        "chunking_strategies": CHUNKING_STRATEGIES,
        "retrieval_modes": RETRIEVAL_MODES,
        "topk_presets": TOPK_PRESETS,
        "faithfulness_modes": FAITHFULNESS_MODES,
        "sweepable_axes": SWEEPABLE_AXES,
        "llm_models": [{"id": "auto", "label": "Auto (production HF failover chain)",
                        "provider": "hf", "paid": False}] + hf_models,
        "openrouter_models": openrouter_models,
        "openrouter_configured": bool(app_config.OPENROUTER_API_KEY),
        "documents": db.list_documents(user.user_id),
    }


# --- eval question datasets ------------------------------------------------
class QuestionIn(BaseModel):
    question: str
    expected_answer: str | None = None
    relevant_sections: list[str] | None = None


class SaveDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    questions: list[QuestionIn]


@router.post("/datasets")
def save_dataset(req: SaveDatasetRequest, user: Principal = Depends(current_user)):
    if not req.questions:
        raise HTTPException(400, "At least one question is required.")
    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    eval_store.save_dataset(dataset_id, user.user_id, req.name,
                            [q.model_dump() for q in req.questions])
    return {"dataset_id": dataset_id, "n_questions": len(req.questions)}


@router.post("/datasets/upload")
async def upload_dataset(name: str, file: UploadFile = File(...),
                         user: Principal = Depends(current_user)):
    """Accepts CSV (columns: question, expected_answer, relevant_sections
    [pipe-separated]) or JSON (a list of {question, expected_answer,
    relevant_sections})."""
    raw = (await file.read()).decode("utf-8", errors="ignore")
    filename = (file.filename or "").lower()
    questions: list[dict] = []
    try:
        if filename.endswith(".json"):
            data = json.loads(raw)
            for row in data:
                if isinstance(row, str):
                    questions.append({"question": row})
                else:
                    questions.append({
                        "question": row["question"],
                        "expected_answer": row.get("expected_answer"),
                        "relevant_sections": row.get("relevant_sections"),
                    })
        else:  # csv, or plain text (one question per line)
            first_line = raw.splitlines()[0] if raw.splitlines() else ""
            looks_like_csv = "," in first_line and "question" in first_line.lower()
            if looks_like_csv:
                reader = csv.DictReader(io.StringIO(raw))
                for row in reader:
                    sections = row.get("relevant_sections") or ""
                    questions.append({
                        "question": row.get("question", "").strip(),
                        "expected_answer": (row.get("expected_answer") or "").strip() or None,
                        "relevant_sections": [s.strip() for s in sections.split("|") if s.strip()] or None,
                    })
            else:
                questions = [{"question": line.strip()} for line in raw.splitlines() if line.strip()]
    except Exception as e:
        raise HTTPException(400, f"Could not parse dataset file: {e}")

    questions = [q for q in questions if q.get("question")]
    if not questions:
        raise HTTPException(400, "No questions found in the uploaded file.")

    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    eval_store.save_dataset(dataset_id, user.user_id, name or file.filename, questions)
    return {"dataset_id": dataset_id, "n_questions": len(questions)}


@router.get("/datasets")
def list_datasets(user: Principal = Depends(current_user)):
    return {"datasets": eval_store.list_datasets(user.user_id)}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, user: Principal = Depends(current_user)):
    d = eval_store.get_dataset(dataset_id, user.user_id)
    if not d:
        raise HTTPException(404, "Dataset not found")
    return d


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user: Principal = Depends(current_user)):
    eval_store.delete_dataset(dataset_id, user.user_id)
    return {"deleted": dataset_id}


# --- running experiments --------------------------------------------------
class RunRequest(BaseModel):
    doc_ids: list[str]
    config: dict
    dataset_id: str | None = None
    questions: list[QuestionIn] | None = None   # inline questions, alternative to dataset_id
    sweep_axis: str = "none"
    sweep_values: list | None = None


def _resolve_questions(user_id: str, req: RunRequest) -> list[dict]:
    if req.dataset_id:
        d = eval_store.get_dataset(req.dataset_id, user_id)
        if not d:
            raise HTTPException(404, "Dataset not found")
        return d["questions"]
    if req.questions:
        return [q.model_dump() for q in req.questions]
    raise HTTPException(400, "Provide either dataset_id or an inline questions list.")


@router.post("/run")
def run(req: RunRequest, user: Principal = Depends(current_user)):
    if not req.doc_ids:
        raise HTTPException(400, "Select at least one document to build the eval corpus from.")
    owned = db.owned_document_ids(user.user_id, req.doc_ids)
    if not owned:
        raise HTTPException(400, "None of the selected documents belong to you.")

    questions = _resolve_questions(user.user_id, req)
    base_config = parse_run_config(req.config)

    try:
        if req.sweep_axis and req.sweep_axis != "none" and req.sweep_values:
            result = run_sweep(user.user_id, base_config, req.sweep_axis,
                               req.sweep_values, owned, questions, req.dataset_id)
            return _json_safe(result)
        base_config.sweep_id = f"sweep_{uuid.uuid4().hex[:10]}"
        result = run_experiment(user.user_id, base_config, owned, questions, req.dataset_id)
        return _json_safe({"sweep_id": base_config.sweep_id, "axis": "none", "runs": [result]})
    except PipelineError as e:
        raise HTTPException(400, str(e))


# --- browsing results ------------------------------------------------------
@router.get("/runs")
def list_runs(sweep_id: str | None = None, user: Principal = Depends(current_user)):
    return _json_safe({"runs": eval_store.list_runs(user.user_id, sweep_id)})


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: Principal = Depends(current_user)):
    r = eval_store.get_run(run_id, user.user_id)
    if not r:
        raise HTTPException(404, "Run not found")
    r["items"] = eval_store.list_run_items(run_id, user.user_id)
    return _json_safe(r)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, user: Principal = Depends(current_user)):
    eval_store.delete_run(run_id, user.user_id)
    return {"deleted": run_id}


@router.get("/runs/{run_id}/export.csv")
def export_run_csv(run_id: str, user: Principal = Depends(current_user)):
    r = eval_store.get_run(run_id, user.user_id)
    if not r:
        raise HTTPException(404, "Run not found")
    items = eval_store.list_run_items(run_id, user.user_id)

    buf = io.StringIO()
    fields = ["question", "answer", "expected_answer", "error",
             "recall_at_k", "precision_at_k", "ndcg_at_k", "mrr",
             "context_precision", "context_recall", "faithfulness",
             "answer_relevance", "keyword_f1", "composite_score",
             "retrieval_ms", "rerank_ms", "generation_ms", "total_ms"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for it in items:
        row = {"question": it["question"], "answer": it.get("answer"),
              "expected_answer": it.get("expected_answer"), "error": it.get("error")}
        for k, v in it.get("metrics", {}).items():
            row[k] = "" if isinstance(v, float) and v != v else v
        writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )
