"""Data shapes for the Evaluation Lab: the configuration one run is built
from, a single eval question, and the curated preset lists the UI's dropdowns
are populated from.

Keeping every tunable in one dataclass (`RunConfig`) is what makes a "sweep"
possible: a sweep is just N `RunConfig` instances that differ in exactly one
field, run back to back and tagged with a shared `sweep_id`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

from .chunkers import CHUNKING_STRATEGIES
from .hybrid_index import RETRIEVAL_MODES


# --------------------------------------------------------------------------
# Curated presets shown in the UI. Free-text override is always allowed too.
# --------------------------------------------------------------------------

EMBEDDING_PRESETS = [
    {"id": "BAAI/bge-small-en-v1.5", "label": "BGE Small (fast, 384d)"},
    {"id": "BAAI/bge-base-en-v1.5", "label": "BGE Base (768d)"},
    {"id": "BAAI/bge-large-en-v1.5", "label": "BGE Large (1024d, slow)"},
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "label": "MiniLM-L6 (fast, 384d)"},
    {"id": "sentence-transformers/all-mpnet-base-v2", "label": "MPNet Base (768d)"},
    {"id": "intfloat/e5-small-v2", "label": "E5 Small (384d)"},
    {"id": "intfloat/e5-base-v2", "label": "E5 Base (768d)"},
]

RERANKER_PRESETS = [
    {"id": "none", "label": "No reranker"},
    {"id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "label": "MiniLM-L6 cross-encoder (fast)"},
    {"id": "cross-encoder/ms-marco-MiniLM-L-12-v2", "label": "MiniLM-L12 cross-encoder"},
    {"id": "cross-encoder/ms-marco-TinyBERT-L-2-v2", "label": "TinyBERT cross-encoder (fastest)"},
]

# Curated common OpenRouter models, prefixed "openrouter:" so generation.py's
# parse_model_ref() routes them to the paid provider. Any other OpenRouter
# model id works too -- this list is just what the UI offers by default.
# See https://openrouter.ai/models for the full, current catalog and pricing.
OPENROUTER_MODEL_PRESETS = [
    {"id": "openrouter:openai/gpt-4o", "label": "GPT-4o (OpenAI, paid)"},
    {"id": "openrouter:openai/gpt-4o-mini", "label": "GPT-4o mini (OpenAI, paid)"},
    {"id": "openrouter:anthropic/claude-3.7-sonnet", "label": "Claude 3.7 Sonnet (Anthropic, paid)"},
    {"id": "openrouter:anthropic/claude-3.5-haiku", "label": "Claude 3.5 Haiku (Anthropic, paid)"},
    {"id": "openrouter:google/gemini-2.0-flash-001", "label": "Gemini 2.0 Flash (Google, paid)"},
    {"id": "openrouter:deepseek/deepseek-chat", "label": "DeepSeek Chat (paid)"},
    {"id": "openrouter:meta-llama/llama-3.1-405b-instruct", "label": "Llama 3.1 405B via OpenRouter (paid)"},
]

CHUNKING_PRESETS = [200, 400, 750, 1200, 1600, 2400, 3200]

TOPK_PRESETS = [1, 3, 5, 8, 10, 15, 20]

FAITHFULNESS_MODES = [
    {"id": "cosine", "label": "Fast (embedding cosine)"},
    {"id": "nli", "label": "Accurate (local NLI cross-encoder, slower)"},
]

SWEEPABLE_AXES = [
    {"id": "chunking_strategy", "label": "Chunking strategy"},
    {"id": "chunk_target_tokens", "label": "Chunk size (tokens)"},
    {"id": "chunk_overlap_tokens", "label": "Chunk overlap (tokens)"},
    {"id": "embed_model", "label": "Embedding model"},
    {"id": "retrieval_mode", "label": "Retrieval mode"},
    {"id": "reranker_model", "label": "Reranker"},
    {"id": "final_k", "label": "Top-K (retrieval depth)"},
    {"id": "dense_weight", "label": "Dense fusion weight"},
    {"id": "sparse_weight", "label": "Sparse (BM25) fusion weight"},
    {"id": "llm_model_id", "label": "Generator LLM"},
    {"id": "noise_pct", "label": "Injected input noise (%)"},
    {"id": "none", "label": "No sweep (single run)"},
]


@dataclass
class RunConfig:
    # --- chunking ---------------------------------------------------
    chunking_strategy: str = "fixed"   # fixed | sentence | recursive | semantic
    chunk_target_tokens: int = 750
    chunk_overlap_tokens: int = 150
    chunk_min_tokens: int = 80

    # --- embedding ----------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- retrieval / fusion --------------------------------------------
    retrieval_mode: str = "hybrid_rrf"  # dense | sparse | hybrid_rrf | hybrid_weighted | mmr
    dense_k: int = 25
    sparse_k: int = 25
    final_k: int = 8
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    mmr_lambda: float = 0.5   # only used when retrieval_mode == "mmr"; 1.0 = pure relevance, 0.0 = pure diversity

    # --- reranking -------------------------------------------------------
    reranker_model: str = "none"       # "none" disables reranking
    rerank_pool: int = 20              # candidates handed to the reranker before truncating to final_k

    # --- generation --------------------------------------------------
    llm_model_id: str = "auto"         # "auto" = existing failover chain; else a specific MODEL_CHAIN id
    temperature: float = 0.0
    max_tokens: int = 700

    # --- input type / quality -------------------------------------------
    noise_pct: float = 0.0             # 0..1 fraction of final_k slots replaced with unrelated chunks

    # --- metric computation ------------------------------------------
    faithfulness_mode: str = "cosine"  # "cosine" | "nli"
    silver_relevance_k: int = 5        # size of the embedding-derived proxy relevant-set per question
    use_llm_judge: bool = False        # optional, costs HF inference credits

    # --- bookkeeping -----------------------------------------------------
    label: str = ""
    sweep_id: str = ""
    sweep_axis: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the fields that actually change pipeline behaviour
        (excludes bookkeeping fields), used to detect/skip duplicate configs
        within a sweep and as a short run label suffix."""
        d = self.to_dict()
        for k in ("label", "sweep_id", "sweep_axis"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def auto_label(self) -> str:
        rr = "none" if self.reranker_model == "none" else self.reranker_model.split("/")[-1]
        return (
            f"{self.chunking_strategy}{self.chunk_target_tokens}"
            f"_emb-{self.embed_model.split('/')[-1]}"
            f"_{self.retrieval_mode}"
            f"_rr-{rr}"
            f"_k{self.final_k}"
            f"_llm-{self.llm_model_id.split(':')[-1]}"
            f"_noise{int(self.noise_pct * 100)}"
        )


@dataclass
class EvalQuestion:
    """One evaluation item. `expected_answer` and `relevant_sections` are
    optional gold labels; when absent, metrics fall back to embedding-derived
    proxies (clearly flagged as such in results)."""
    question: str
    expected_answer: str | None = None
    relevant_sections: list[str] | None = None   # substrings matched against chunk `section`/`text`

    def to_dict(self) -> dict:
        return asdict(self)


def parse_run_config(payload: dict) -> RunConfig:
    """Build a RunConfig from a JSON payload, ignoring unknown keys and
    filling in defaults for missing ones."""
    valid = {f for f in RunConfig.__dataclass_fields__.keys()}
    clean = {k: v for k, v in (payload or {}).items() if k in valid}
    return RunConfig(**clean)
