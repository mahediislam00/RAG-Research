"""Central configuration.

Everything tunable lives here or in environment variables (.env). The model
fallback chain is the heart of the "switch model when free credits run out"
behaviour: models are tried top-to-bottom; one that returns a credit/rate-limit
error is put on cooldown and the next one is used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Paths --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
for _d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- HuggingFace Inference Providers (OpenAI-compatible router) ----------
# Docs: https://huggingface.co/docs/inference-providers
HF_TOKEN = _env("HF_TOKEN", "")
HF_ROUTER_URL = _env("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")


@dataclass
class ModelSpec:
    """One entry in the fallback chain."""
    id: str                 # HF model id, optionally with a :provider suffix
    label: str              # friendly name shown in the UI
    max_tokens: int = 1024  # max completion tokens to request
    context_tokens: int = 32000  # rough context budget for trimming the prompt


# Ordered best -> lightest. The chain mixes Qwen / Llama / Mistral so that if a
# model's free monthly allowance is spent (HTTP 402) or it is briefly rate
# limited (HTTP 429), the app keeps working on the next one. The small 7B/8B
# models at the bottom are the most reliably available on the free tier and act
# as a safety net. Update these ids freely — `huggingface.co/models?inference=warm`
# lists what is currently servable.
MODEL_CHAIN: list[ModelSpec] = [
    ModelSpec("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B", max_tokens=1536, context_tokens=32000),
    ModelSpec("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B", max_tokens=1536, context_tokens=32000),
    ModelSpec("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral Small 24B", max_tokens=1536, context_tokens=32000),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B", max_tokens=1024, context_tokens=32000),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B", max_tokens=1024, context_tokens=16000),
]

# Cooldown applied to a model after it signals exhaustion, in seconds.
COOLDOWN_CREDIT_EXHAUSTED = int(_env("COOLDOWN_CREDIT_EXHAUSTED", str(60 * 60)))  # 1h
COOLDOWN_RATE_LIMITED = int(_env("COOLDOWN_RATE_LIMITED", "45"))
COOLDOWN_PROVIDER_ERROR = int(_env("COOLDOWN_PROVIDER_ERROR", "30"))


# --- OpenRouter (optional, paid models) -----------------------------------
# Only used by the Evaluation Lab, to benchmark paid frontier models (GPT-4o,
# Claude, Gemini, etc.) alongside the free HuggingFace chain above — chat
# never calls OpenRouter. Get a key at https://openrouter.ai/keys and leave
# blank to disable; the Eval Lab clearly marks OpenRouter models as
# unavailable until a key is set, rather than failing silently mid-run.
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = _env("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
# Sent as optional attribution headers OpenRouter uses for their public
# leaderboard; cosmetic only, safe to leave as-is or override.
OPENROUTER_SITE_URL = _env("OPENROUTER_SITE_URL", "http://localhost:8000")
OPENROUTER_SITE_NAME = _env("OPENROUTER_SITE_NAME", "RAG Lab")


# --- Embeddings (local, free, no API) -----------------------------------
# bge-small-en-v1.5 is a strong, small retrieval model. It runs locally via
# sentence-transformers; no credits involved. Swap for bge-base / bge-large
# for higher quality at the cost of speed/RAM.
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models retrieve better when the *query* carries this instruction. Passages
# are embedded without it.
EMBED_QUERY_PREFIX = _env(
    "EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)


# --- Chunking -----------------------------------------------------------
@dataclass
class ChunkConfig:
    target_tokens: int = 750     # ~ a long paragraph; good recall for dense + BM25
    overlap_tokens: int = 150    # carry context across boundaries
    min_tokens: int = 80         # discard tiny fragments


CHUNK = ChunkConfig()


# --- Retrieval ----------------------------------------------------------
@dataclass
class RetrievalConfig:
    dense_k: int = 25            # candidates from semantic search
    sparse_k: int = 25           # candidates from BM25
    final_k: int = 8             # passages handed to the LLM
    rrf_k: int = 60              # Reciprocal Rank Fusion constant
    dense_weight: float = 1.0    # weight of semantic ranking in fusion
    sparse_weight: float = 1.0   # weight of keyword ranking in fusion


RETRIEVAL = RetrievalConfig()


# --- Qdrant vector store ------------------------------------------------
# QDRANT_MODE controls where vectors are stored:
#   memory  – in-process (default; no server; data lost on restart)
#   local   – on-disk via qdrant-client embedded mode (set QDRANT_PATH)
#   remote  – external Qdrant server or Qdrant Cloud (set QDRANT_URL / QDRANT_API_KEY)
QDRANT_MODE       = _env("QDRANT_MODE",       "memory")
QDRANT_PATH       = _env("QDRANT_PATH",       str(DATA_DIR / "qdrant"))
QDRANT_URL        = _env("QDRANT_URL",        "http://localhost:6333")
QDRANT_API_KEY    = _env("QDRANT_API_KEY",    "")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "raglab_chunks")


# --- Auth & tenancy -----------------------------------------------------
# Users, sessions, and the document ownership registry live in SQLite. This is
# the authorization source of truth; Qdrant payloads mirror it.
DB_PATH = _env("DB_PATH", str(DATA_DIR / "raglab.db"))

JWT_ISSUER = _env("JWT_ISSUER", "raglab")
JWT_TTL_SECONDS = int(_env("JWT_TTL_SECONDS", str(12 * 60 * 60)))  # 12h

# HMAC key for session tokens. Set JWT_SECRET in .env for anything beyond a
# single-machine dev run: without it we generate one and persist it to
# data/.jwt_secret (mode 0600) so restarts don't silently sign out everyone —
# but that file is per-machine, so multiple app servers MUST share an explicit
# JWT_SECRET or tokens minted by one will be rejected by the next.
_SECRET_FILE = DATA_DIR / ".jwt_secret"


def _load_or_create_secret() -> str:
    env_secret = os.environ.get("JWT_SECRET", "").strip()
    if env_secret:
        if len(env_secret) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 characters.")
        return env_secret
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text(encoding="utf-8").strip()
    import secrets as _secrets

    generated = _secrets.token_urlsafe(48)
    _SECRET_FILE.write_text(generated, encoding="utf-8")
    try:
        os.chmod(_SECRET_FILE, 0o600)
    except OSError:
        pass
    return generated


JWT_SECRET = _load_or_create_secret()

# Brute-force throttle on /api/auth/login (per email, per process).
LOGIN_MAX_ATTEMPTS = int(_env("LOGIN_MAX_ATTEMPTS", "8"))
LOGIN_WINDOW_SECONDS = int(_env("LOGIN_WINDOW_SECONDS", "60"))

# Reject oversized uploads before reading them into memory.
MAX_UPLOAD_BYTES = int(_env("MAX_UPLOAD_MB", "50")) * 1024 * 1024

# Browser origins allowed to call the API. Same-origin only by default.
CORS_ORIGINS = [o for o in _env("CORS_ORIGINS", "").split(",") if o.strip()]
