"""Synchronous, single-shot chat completion for the Evaluation Lab, across
two providers:

- **hf**          — HuggingFace Inference Providers (the same free-tier
                    router `app/llm.py` uses for production chat).
- **openrouter**   — OpenRouter, an OpenAI-compatible gateway to paid
                    frontier models (GPT-4o, Claude, Gemini, etc.), so a
                    sweep can compare free HF models against paid ones on
                    the same questions and metrics.

`app/llm.py`'s `ChatRouter` is built for the product's automatic multi-model
failover during a *streamed* chat turn. Evaluation runs want the opposite: a
specific model pinned per run (so "generator LLM" is a real independent
variable), a blocking call (so latency is measured cleanly and the full text
is available for metric computation), and a failure that is recorded on the
one question it affected rather than one that cascades through a whole model
chain. This module is intentionally separate from `ChatRouter` so nothing
here can change production chat behaviour.

Model references use a provider prefix: `openrouter:openai/gpt-4o-mini` or
`hf:meta-llama/Llama-3.1-8B-Instruct`. A bare id with no recognized prefix
(or `"auto"`) is treated as `hf`, matching the production model ids in
`config.MODEL_CHAIN`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .. import config


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    model_id: str
    error: str | None = None


def parse_model_ref(model_id: str) -> tuple[str, str]:
    """Split a "provider:model" reference into (provider, model_name).
    Bare ids and "auto" default to "hf" for backward compatibility with
    plain HuggingFace model ids, which never contain a provider prefix."""
    if model_id.startswith("openrouter:"):
        return "openrouter", model_id.split(":", 1)[1]
    if model_id.startswith("hf:"):
        return "hf", model_id.split(":", 1)[1]
    return "hf", model_id


def resolve_model_id(llm_model_id: str) -> str:
    """"auto" means "the first model in the production HF fallback chain",
    used so a run can approximate current production behaviour without the
    user having to name a model explicitly. Only applies to the hf provider
    -- OpenRouter references are always explicit."""
    if llm_model_id and llm_model_id != "auto":
        return llm_model_id
    if config.MODEL_CHAIN:
        return config.MODEL_CHAIN[0].id
    raise RuntimeError("No models configured in MODEL_CHAIN.")


def _max_tokens_for_hf(model_id: str, fallback: int) -> int:
    for m in config.MODEL_CHAIN:
        if m.id == model_id:
            return m.max_tokens
    return fallback


def _post_chat_completion(url: str, headers: dict, payload: dict,
                          timeout: tuple[int, int]) -> tuple[str, float, str | None]:
    """Shared request/response handling for any OpenAI-compatible chat
    completions endpoint. Never raises: returns (text, latency_ms, error)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return "", (time.perf_counter() - t0) * 1000, f"network error: {e}"
    latency_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        body = resp.text[:300] if resp.text else ""
        return "", latency_ms, f"HTTP {resp.status_code}: {body}"
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return "", latency_ms, f"unparsable response: {e}"
    return text.strip(), latency_ms, None


def _complete_hf(model_name: str, messages: list[dict], temperature: float,
                 max_tokens: int, timeout: tuple[int, int]) -> GenerationResult:
    if not config.HF_TOKEN:
        return GenerationResult("", 0.0, f"hf:{model_name}",
                                error="HF_TOKEN is not set; cannot call HuggingFace Inference Providers.")
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": min(max_tokens, _max_tokens_for_hf(model_name, max_tokens)),
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {config.HF_TOKEN}", "Content-Type": "application/json"}
    text, latency_ms, error = _post_chat_completion(config.HF_ROUTER_URL, headers, payload, timeout)
    return GenerationResult(text, latency_ms, f"hf:{model_name}", error)


def _complete_openrouter(model_name: str, messages: list[dict], temperature: float,
                         max_tokens: int, timeout: tuple[int, int]) -> GenerationResult:
    if not config.OPENROUTER_API_KEY:
        return GenerationResult("", 0.0, f"openrouter:{model_name}",
                                error="OPENROUTER_API_KEY is not set. Add one at "
                                      "https://openrouter.ai/keys to benchmark paid models.")
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.OPENROUTER_SITE_URL,
        "X-Title": config.OPENROUTER_SITE_NAME,
    }
    text, latency_ms, error = _post_chat_completion(config.OPENROUTER_API_URL, headers, payload, timeout)
    return GenerationResult(text, latency_ms, f"openrouter:{model_name}", error)


def complete(messages: list[dict], model_id: str, temperature: float = 0.0,
            max_tokens: int = 700, timeout: tuple[int, int] = (15, 120)) -> GenerationResult:
    """One non-streaming completion, routed to whichever provider the
    model reference specifies. Never raises: failures come back as a
    GenerationResult with `.error` set, so a sweep can keep going and the
    failure shows up per-question in the results table instead of aborting
    the whole run."""
    provider, name = parse_model_ref(model_id)
    if provider == "hf":
        name = resolve_model_id(name if name != "auto" else "auto")
        return _complete_hf(name, messages, temperature, max_tokens, timeout)
    if provider == "openrouter":
        return _complete_openrouter(name, messages, temperature, max_tokens, timeout)
    return GenerationResult("", 0.0, model_id, error=f"Unknown provider prefix in model id {model_id!r}.")


def judge_relevance(question: str, chunk_text: str, model_id: str) -> bool | None:
    """Optional LLM-as-judge relevance label for one (question, chunk) pair.
    Only invoked when `use_llm_judge` is enabled on the run config -- it costs
    one inference call per retrieved passage per question (real money, for
    OpenRouter models), so it is opt-in and should be used on small question
    sets. Returns None on any failure (caller falls back to the embedding-
    based silver label)."""
    messages = [
        {"role": "system", "content": (
            "You judge whether a passage is relevant to a question. "
            "Reply with exactly one word: YES or NO."
        )},
        {"role": "user", "content": f"Question: {question}\n\nPassage:\n{chunk_text[:1500]}\n\nRelevant?"},
    ]
    result = complete(messages, model_id, temperature=0.0, max_tokens=5, timeout=(10, 30))
    if result.error:
        return None
    ans = result.text.strip().upper()
    if ans.startswith("Y"):
        return True
    if ans.startswith("N"):
        return False
    return None
