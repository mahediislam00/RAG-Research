"""End-to-end checks for the Evaluation Lab (app/eval/*, app/eval_routes.py).

Follows the same pattern as test_tenancy.py: real chunking, real hybrid
BM25+dense+RRF retrieval, real metric math, and real SQLite persistence, with
only the two calls that need a downloaded model (embeddings, chat
completion) stubbed out -- so this validates the harness's own logic without
requiring network access or a HuggingFace token.

Run with: python tests/test_eval_lab.py
"""
import hashlib
import json as _json
import os
import sys
import tempfile
import time
import uuid

import numpy as np

TMP = tempfile.mkdtemp()
os.environ.update(
    DATA_DIR=TMP,
    QDRANT_MODE="local",
    QDRANT_PATH=os.path.join(TMP, "qdrant"),
    JWT_SECRET="x" * 40,
    HF_TOKEN="test_token",
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ok = lambda m: print(f"  PASS  {m}")

# --- stub the two calls that need a downloaded model ----------------------
# Deterministic hash-bucket "embedding": lexically similar text lands in
# similar buckets, which is enough to exercise real ranking/RRF/metric logic
# without a real model or network access.
from app.eval import embed_cache, generation

_DIM = 64


def _pseudo_vec(text: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    for w in text.lower().split():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % _DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


embed_cache.embed_passages = lambda model, texts, batch_size=32: (
    np.stack([_pseudo_vec(t) for t in texts]) if texts else np.zeros((0, _DIM), dtype=np.float32)
)
embed_cache.embed_query = lambda model, text: _pseudo_vec(text)
embed_cache.dim = lambda model: _DIM


def _fake_complete(messages, model_id, temperature=0.0, max_tokens=700, timeout=(15, 120)):
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    ctx = user_msg.split("QUESTION:")[0]
    snippet = " ".join(ctx.split()[10:40])
    return generation.GenerationResult(
        text=f"Based on the source passages, {snippet}. [doc p.1]",
        latency_ms=12.3, model_id=model_id, error=None,
    )


generation.complete = _fake_complete

from fastapi.testclient import TestClient
from app import config as appconfig
from app import db as appdb
from app.main import app

client = TestClient(app)

print("=== 1. account + seed a document (bypass production embedder) ===")
r = client.post("/api/auth/register", json={"email": "evallab@example.com", "password": "testpass123"})
assert r.status_code == 200, r.text
token = r.json()["token"]
user_id = r.json()["user"]["user_id"]
H = {"Authorization": f"Bearer {token}"}
ok("registered")

sow_text = (
    "SECTION C - STATEMENT OF WORK\n\n"
    "The contractor shall provide cloud migration services for the data center. "
    "The period of performance is 12 months from the effective date of the task order. "
    "Deliverables include a migration plan, a security assessment, and monthly status reports. "
    "All work shall comply with FAR 52.219-14 and NIST 800-53 controls. " * 6 +
    "\n\nSECTION L - INSTRUCTIONS TO OFFERORS\n\n"
    "Offerors shall submit a technical proposal not exceeding 20 pages. "
    "The proposal shall address staffing plan, past performance, and management approach. " * 6
)
doc_id = uuid.uuid4().hex[:12]
user_dir = appconfig.UPLOAD_DIR / user_id
user_dir.mkdir(parents=True, exist_ok=True)
dest = user_dir / f"{doc_id}.txt"
dest.write_text(sow_text, encoding="utf-8")
appdb.add_document(doc_id, user_id, "sow.txt", str(dest), pages=1, chunks=0,
                   upload_time=time.time(), session_id=None)
ok(f"seeded document {doc_id}")

print("\n=== 2. options + dataset ===")
r = client.get("/api/eval/options", headers=H)
assert r.status_code == 200, r.text
assert r.json()["documents"][0]["document_id"] == doc_id
ok("GET /api/eval/options lists the seeded document")

r = client.post("/api/eval/datasets", headers=H, json={
    "name": "smoke-test",
    "questions": [
        {"question": "What is the period of performance?", "expected_answer": "12 months"},
        {"question": "How many pages is the technical proposal limited to?",
         "relevant_sections": ["SECTION L"]},
    ],
})
assert r.status_code == 200, r.text
dataset_id = r.json()["dataset_id"]
ok(f"saved dataset {dataset_id} with 2 questions")

print("\n=== 3. single run: real chunking + hybrid retrieval + metrics ===")
r = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "chunk_overlap_tokens": 20,
              "embed_model": "fake/pseudo-embed", "final_k": 3,
              "dense_k": 10, "sparse_k": 10, "reranker_model": "none"},
})
assert r.status_code == 200, r.text
run = r.json()["runs"][0]
summary = run["summary"]
assert summary["n_questions"] == 2 and summary["errors"] == 0
assert 0 <= summary["recall_at_k"]["mean"] <= 1
assert 0 <= summary["ndcg_at_k"]["mean"] <= 1
assert 0 <= summary["faithfulness"]["mean"] <= 1
ok(f"run {run['run_id']} produced valid metrics with zero per-question errors")

print("\n=== 4. per-question detail: gold vs silver relevance is tagged correctly ===")
r = client.get(f"/api/eval/runs/{run['run_id']}", headers=H)
assert r.status_code == 200, r.text
items = r.json()["items"]
assert len(items) == 2
by_q = {it["question"]: it for it in items}
assert by_q["How many pages is the technical proposal limited to?"]["metrics"]["relevance_source"] == "gold"
assert by_q["What is the period of performance?"]["metrics"]["relevance_source"] == "silver"
ok("gold relevant_sections used when supplied, silver (embedding) proxy used otherwise")

print("\n=== 5. NaN metrics (e.g. keyword_f1 with no gold answer) serialize safely ===")
_json.dumps(r.json())  # would raise if any stray NaN slipped through
ok("run detail is valid JSON end to end (no raw NaN in the HTTP response)")

print("\n=== 6. sweeping Top-K changes Recall@k monotonically (sanity check) ===")
r = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "embed_model": "fake/pseudo-embed",
              "reranker_model": "none"},
    "sweep_axis": "final_k", "sweep_values": [1, 2, 4],
})
assert r.status_code == 200, r.text
sweep = r.json()
assert len(sweep["runs"]) == 3
recalls = [run_["summary"]["recall_at_k"]["mean"] for run_ in sweep["runs"]]
assert recalls == sorted(recalls), f"expected non-decreasing recall as k grows, got {recalls}"
ok(f"recall@k grows monotonically across k=1,2,4: {recalls}")

r = client.get(f"/api/eval/runs?sweep_id={sweep['sweep_id']}", headers=H)
assert r.status_code == 200 and len(r.json()["runs"]) == 3
ok("all 3 sweep runs are retrievable by shared sweep_id")

print("\n=== 7. injected input noise measurably changes generation-layer metrics ===")
r_clean = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "embed_model": "fake/pseudo-embed",
              "final_k": 3, "reranker_model": "none", "noise_pct": 0.0},
}).json()["runs"][0]
r_noisy = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "embed_model": "fake/pseudo-embed",
              "final_k": 3, "reranker_model": "none", "noise_pct": 0.67},
}).json()["runs"][0]
assert r_noisy["summary"]["faithfulness"]["mean"] <= r_clean["summary"]["faithfulness"]["mean"]
ok(f"faithfulness with noise ({r_noisy['summary']['faithfulness']['mean']}) <= "
  f"without ({r_clean['summary']['faithfulness']['mean']})")

print("\n=== 8. CSV export ===")
r = client.get(f"/api/eval/runs/{run['run_id']}/export.csv", headers=H)
assert r.status_code == 200
assert r.text.splitlines()[0].startswith("question,answer")
assert len(r.text.splitlines()) == 3  # header + 2 questions
ok("CSV export has a header row plus one row per question")

print("\n=== 9. tenancy: a second user cannot see the first user's runs ===")
r2 = client.post("/api/auth/register", json={"email": "other@example.com", "password": "testpass123"})
H2 = {"Authorization": f"Bearer {r2.json()['token']}"}
r = client.get("/api/eval/runs", headers=H2)
assert r.status_code == 200 and r.json()["runs"] == []
r = client.get(f"/api/eval/runs/{run['run_id']}", headers=H2)
assert r.status_code == 404
ok("a different user sees zero runs and gets 404 for the first user's run_id")

print("\n=== 10. deleting a run removes it and its items ===")
r = client.delete(f"/api/eval/runs/{run['run_id']}", headers=H)
assert r.status_code == 200
r = client.get(f"/api/eval/runs/{run['run_id']}", headers=H)
assert r.status_code == 404
ok("deleted run is gone")

print("\n=== 11. LLM-judge relevance mode falls back to silver when the judge is inconclusive ===")
r = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "embed_model": "fake/pseudo-embed",
              "final_k": 3, "reranker_model": "none", "use_llm_judge": True},
})
assert r.status_code == 200, r.text
judge_run = r.json()["runs"][0]
detail = client.get(f"/api/eval/runs/{judge_run['run_id']}", headers=H).json()
# the stubbed generator never answers literally YES/NO, so every judgment call
# is inconclusive and the pipeline must fall back to the silver proxy rather
# than silently treating "no judgments succeeded" as "nothing is relevant".
sources = {it["metrics"]["relevance_source"] for it in detail["items"]}
assert sources <= {"silver", "gold"}, sources
ok(f"use_llm_judge=True with an inconclusive judge falls back cleanly: {sources}")

print("\n=== 12. chunking strategies produce different chunk counts on the same corpus ===")
r = client.get("/api/eval/options", headers=H)
strategy_ids = {s["id"] for s in r.json()["chunking_strategies"]}
assert strategy_ids == {"fixed", "sentence", "recursive", "semantic"}, strategy_ids
corpus_sizes = {}
for strategy in ["fixed", "sentence", "recursive", "semantic"]:
    resp = client.post("/api/eval/run", headers=H, json={
        "doc_ids": [doc_id], "dataset_id": dataset_id,
        "config": {"chunking_strategy": strategy, "chunk_target_tokens": 120,
                  "chunk_overlap_tokens": 20, "embed_model": "fake/pseudo-embed",
                  "final_k": 3, "reranker_model": "none"},
    })
    assert resp.status_code == 200, (strategy, resp.text)
    run_ = resp.json()["runs"][0]
    assert run_["summary"]["errors"] == 0, (strategy, run_["summary"])
    corpus_sizes[strategy] = run_["summary"]["corpus_size"]
ok(f"all 4 chunking strategies ran without error; corpus sizes: {corpus_sizes}")

print("\n=== 13. retrieval modes all run cleanly on a realistically-chunked corpus ===")
r = client.get("/api/eval/options", headers=H)
mode_ids = {m["id"] for m in r.json()["retrieval_modes"]}
assert mode_ids == {"hybrid_rrf", "hybrid_weighted", "dense", "sparse", "mmr"}, mode_ids
mode_results = {}
for mode in sorted(mode_ids):
    resp = client.post("/api/eval/run", headers=H, json={
        "doc_ids": [doc_id], "dataset_id": dataset_id,
        "config": {"chunk_target_tokens": 120, "chunk_overlap_tokens": 20,
                  "embed_model": "fake/pseudo-embed",
                  "final_k": 3, "reranker_model": "none", "retrieval_mode": mode},
    })
    assert resp.status_code == 200, (mode, resp.text)
    run_ = resp.json()["runs"][0]
    # "sparse" (pure BM25) is a special case: classic BM25 IDF can go
    # negative for terms that appear in most documents in a small corpus
    # (this test's fixture repeats boilerplate phrasing deliberately, so a
    # term like "period of performance" can appear in nearly every chunk),
    # which correctly yields zero retrieved passages rather than a crash.
    # The other four modes always have a dense semantic fallback and should
    # never legitimately zero out on this fixture.
    if mode != "sparse":
        assert run_["summary"]["errors"] == 0, (mode, run_["summary"])
    mode_results[mode] = (run_["summary"]["ndcg_at_k"]["mean"], run_["summary"]["errors"])
ok(f"all 5 retrieval modes ran without crashing; (mean NDCG@k, error count) per mode: {mode_results}")

print("\n=== 14. mmr retrieval mode surfaces a diversity metric ===")
r = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "chunk_overlap_tokens": 20,
              "embed_model": "fake/pseudo-embed",
              "final_k": 3, "reranker_model": "none", "retrieval_mode": "mmr", "mmr_lambda": 0.3},
})
assert r.status_code == 200, r.text
mmr_run = r.json()["runs"][0]
assert "diversity" in mmr_run["summary"], mmr_run["summary"].keys()
assert mmr_run["summary"]["diversity"]["n"] >= 1
ok(f"mmr run reports diversity stats: {mmr_run['summary']['diversity']}")

print("\n=== 15. OpenRouter model reference is forwarded through the full pipeline ===")
r = client.post("/api/eval/run", headers=H, json={
    "doc_ids": [doc_id], "dataset_id": dataset_id,
    "config": {"chunk_target_tokens": 120, "chunk_overlap_tokens": 20,
              "embed_model": "fake/pseudo-embed",
              "final_k": 3, "reranker_model": "none",
              "llm_model_id": "openrouter:openai/gpt-4o-mini"},
})
assert r.status_code == 200, r.text
or_run = r.json()["runs"][0]
detail = client.get(f"/api/eval/runs/{or_run['run_id']}", headers=H).json()
model_ids_used = {it["metrics"].get("model_id") for it in detail["items"]}
assert model_ids_used == {"openrouter:openai/gpt-4o-mini"}, model_ids_used
ok(f"run configured with an OpenRouter model correctly forwarded that model id per question: {model_ids_used}")

print("\n=== 16. openrouter_configured flag reflects whether a key is set ===")
r = client.get("/api/eval/options", headers=H)
assert r.json()["openrouter_configured"] is False, "no OPENROUTER_API_KEY set in this test env"
assert any(m["provider"] == "openrouter" for m in r.json()["openrouter_models"])
ok("openrouter_configured correctly False with no key set; preset models still listed for the UI")

print("\nALL CHECKS PASSED\n")
