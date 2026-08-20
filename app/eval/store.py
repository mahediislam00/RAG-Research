"""Persistence for the Evaluation Lab: saved question sets, runs, and the
per-question detail behind each run.

Shares the same SQLite file as the rest of the app (`app/db.py`'s
connection) but owns its own tables, created idempotently on import via
`init_schema()`. Every row is scoped by `user_id`, following the same
tenancy rule the rest of the app follows: no query here ever reads or
deletes another user's rows.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from .. import db

_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_datasets (
    dataset_id     TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    name           TEXT NOT NULL,
    questions_json TEXT NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_datasets_user ON eval_datasets(user_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    sweep_id     TEXT,
    sweep_axis   TEXT,
    label        TEXT,
    config_json  TEXT NOT NULL,
    doc_ids_json TEXT NOT NULL,
    dataset_id   TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    summary_json TEXT,
    error        TEXT,
    created_at   REAL NOT NULL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_user ON eval_runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_sweep ON eval_runs(sweep_id);

CREATE TABLE IF NOT EXISTS eval_run_items (
    item_id          TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    question         TEXT NOT NULL,
    expected_answer  TEXT,
    answer           TEXT,
    metrics_json     TEXT,
    passages_json    TEXT,
    error            TEXT,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_items_run ON eval_run_items(run_id);
"""


def init_schema() -> None:
    conn = db.connect()
    with _lock:
        conn.executescript(_SCHEMA)
        conn.commit()


def _exec(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    conn = db.connect()
    with _lock:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = db.connect()
    with _lock:
        return conn.execute(sql, params).fetchall()


# --- datasets -------------------------------------------------------------
def save_dataset(dataset_id: str, user_id: str, name: str, questions: list[dict]) -> None:
    _exec(
        "INSERT OR REPLACE INTO eval_datasets (dataset_id, user_id, name, questions_json, created_at) "
        "VALUES (?,?,?,?,?)",
        (dataset_id, user_id, name, json.dumps(questions), time.time()),
    )


def list_datasets(user_id: str) -> list[dict]:
    rows = _query(
        "SELECT dataset_id, name, questions_json, created_at FROM eval_datasets "
        "WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    out = []
    for r in rows:
        d = dict(r)
        qs = json.loads(d.pop("questions_json"))
        d["n_questions"] = len(qs)
        out.append(d)
    return out


def get_dataset(dataset_id: str, user_id: str) -> dict | None:
    rows = _query(
        "SELECT * FROM eval_datasets WHERE dataset_id=? AND user_id=?",
        (dataset_id, user_id),
    )
    if not rows:
        return None
    d = dict(rows[0])
    d["questions"] = json.loads(d.pop("questions_json"))
    return d


def delete_dataset(dataset_id: str, user_id: str) -> bool:
    cur = _exec("DELETE FROM eval_datasets WHERE dataset_id=? AND user_id=?",
               (dataset_id, user_id))
    return cur.rowcount > 0


# --- runs -------------------------------------------------------------
def create_run(run_id: str, user_id: str, sweep_id: str, sweep_axis: str, label: str,
               config: dict, doc_ids: list[str], dataset_id: str | None) -> None:
    _exec(
        "INSERT INTO eval_runs (run_id, user_id, sweep_id, sweep_axis, label, config_json, "
        "doc_ids_json, dataset_id, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (run_id, user_id, sweep_id, sweep_axis, label, json.dumps(config),
         json.dumps(doc_ids), dataset_id, "running", time.time()),
    )


def finish_run(run_id: str, user_id: str, summary: dict, status: str = "done",
              error: str | None = None) -> None:
    _exec(
        "UPDATE eval_runs SET status=?, summary_json=?, error=?, finished_at=? "
        "WHERE run_id=? AND user_id=?",
        (status, json.dumps(summary), error, time.time(), run_id, user_id),
    )


def list_runs(user_id: str, sweep_id: str | None = None) -> list[dict]:
    if sweep_id:
        rows = _query(
            "SELECT * FROM eval_runs WHERE user_id=? AND sweep_id=? ORDER BY created_at ASC",
            (user_id, sweep_id),
        )
    else:
        rows = _query(
            "SELECT * FROM eval_runs WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        )
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d.pop("config_json"))
        d["doc_ids"] = json.loads(d.pop("doc_ids_json"))
        d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else None
        d.pop("summary_json", None)
        out.append(d)
    return out


def get_run(run_id: str, user_id: str) -> dict | None:
    rows = _query("SELECT * FROM eval_runs WHERE run_id=? AND user_id=?", (run_id, user_id))
    if not rows:
        return None
    d = dict(rows[0])
    d["config"] = json.loads(d.pop("config_json"))
    d["doc_ids"] = json.loads(d.pop("doc_ids_json"))
    d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else None
    d.pop("summary_json", None)
    return d


def delete_run(run_id: str, user_id: str) -> bool:
    _exec("DELETE FROM eval_run_items WHERE run_id=? AND user_id=?", (run_id, user_id))
    cur = _exec("DELETE FROM eval_runs WHERE run_id=? AND user_id=?", (run_id, user_id))
    return cur.rowcount > 0


# --- run items --------------------------------------------------------
def add_run_item(item_id: str, run_id: str, user_id: str, question: str,
                 expected_answer: str | None, answer: str | None, metrics: dict,
                 passages: list[dict], error: str | None) -> None:
    _exec(
        "INSERT INTO eval_run_items (item_id, run_id, user_id, question, expected_answer, "
        "answer, metrics_json, passages_json, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (item_id, run_id, user_id, question, expected_answer, answer,
         json.dumps(metrics), json.dumps(passages), error, time.time()),
    )


def list_run_items(run_id: str, user_id: str) -> list[dict]:
    rows = _query(
        "SELECT * FROM eval_run_items WHERE run_id=? AND user_id=? ORDER BY created_at ASC",
        (run_id, user_id),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else {}
        d["passages"] = json.loads(d.pop("passages_json")) if d.get("passages_json") else []
        out.append(d)
    return out
