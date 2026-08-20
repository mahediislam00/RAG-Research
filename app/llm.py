"""Chat client for the HuggingFace Inference Providers router with automatic
model fallback.

The "switch model when free credits run out" requirement is handled here:

  * Models are tried in the order defined by config.MODEL_CHAIN.
  * HTTP 402 / "exceeded ... included credits" -> the model is put on a long
    cooldown (default 1 hour) and we move to the next model.
  * HTTP 429 (rate limit) -> short cooldown, move on.
  * 5xx / network / provider errors -> short cooldown, move on.
  * Cooldowns are tracked per model so an exhausted model isn't retried until
    its window expires; meanwhile other models keep serving.

Streaming: we open the request and only commit to streaming once the router
returns HTTP 200. Credit/rate errors surface as the initial status code, so we
can fall back cleanly before any tokens reach the user.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator

import requests

from . import config


class AllModelsUnavailable(RuntimeError):
    pass


@dataclass
class _Cooldown:
    until: float = 0.0
    reason: str = ""
    total: float = 0.0      # full cooldown duration, for progress bars
    credit: bool = False    # True when triggered by free-credit exhaustion


class ChatRouter:
    def __init__(self) -> None:
        self._cooldowns: dict[str, _Cooldown] = {}
        self._served: dict[str, int] = {}   # successful completions this session

    # --- cooldown bookkeeping ------------------------------------------
    def _available_models(self) -> list[config.ModelSpec]:
        now = time.time()
        return [m for m in config.MODEL_CHAIN
                if self._cooldowns.get(m.id, _Cooldown()).until <= now]

    def _cooldown(self, model_id: str, seconds: int, reason: str,
                  credit: bool = False) -> None:
        self._cooldowns[model_id] = _Cooldown(
            until=time.time() + seconds, reason=reason,
            total=float(seconds), credit=credit,
        )

    def status(self) -> list[dict]:
        now = time.time()
        out = []
        for m in config.MODEL_CHAIN:
            cd = self._cooldowns.get(m.id, _Cooldown())
            limited = cd.until > now
            remaining = max(0, int(cd.until - now))
            if not limited:
                state = "ready"
            elif cd.credit:
                state = "exhausted"
            else:
                state = "cooldown"
            out.append({
                "id": m.id,
                "label": m.label,
                "available": not limited,
                "state": state,
                "cooldown_remaining": remaining,
                "cooldown_total": int(cd.total) if limited else 0,
                "reason": cd.reason if limited else "",
                "served": self._served.get(m.id, 0),
            })
        return out

    # --- error classification ------------------------------------------
    @staticmethod
    def _is_credit_error(status: int, body: str) -> bool:
        b = body.lower()
        return status == 402 or "included credits" in b or "monthly" in b and "credit" in b

    # --- main entry -----------------------------------------------------
    def stream_chat(self, messages: list[dict], temperature: float = 0.2
                    ) -> Iterator[dict]:
        """Yield events: {"type": "model", "label": ...} once a model is
        chosen, then {"type": "token", "text": ...} repeatedly, then
        {"type": "done"}. Raises AllModelsUnavailable if the whole chain is
        exhausted."""
        if not config.HF_TOKEN:
            raise AllModelsUnavailable(
                "HF_TOKEN is not set. Create a token at "
                "https://huggingface.co/settings/tokens (with 'Make calls to "
                "Inference Providers' permission) and put it in .env."
            )

        tried: list[str] = []
        for model in self._available_models():
            tried.append(model.label)
            try:
                yield from self._stream_one(model, messages, temperature)
                self._served[model.id] = self._served.get(model.id, 0) + 1
                return  # success
            except _ModelFailover as e:
                self._cooldown(model.id, e.cooldown, e.reason, credit=e.credit)
                continue

        # Nothing available. Report the soonest model to recover.
        cds = [(m.label, self._cooldowns.get(m.id, _Cooldown())) for m in config.MODEL_CHAIN]
        cds = [(lbl, cd) for lbl, cd in cds if cd.until > time.time()]
        if cds:
            lbl, cd = min(cds, key=lambda x: x[1].until)
            wait = int(cd.until - time.time())
            raise AllModelsUnavailable(
                f"All models are temporarily unavailable. Soonest to recover: "
                f"{lbl} in ~{wait}s ({cd.reason}). Tried this turn: {', '.join(tried) or 'none'}."
            )
        raise AllModelsUnavailable(
            "No models responded. Check HF_TOKEN and model ids in config.py. "
            f"Tried: {', '.join(tried) or 'none'}."
        )

    def _stream_one(self, model: config.ModelSpec, messages: list[dict],
                    temperature: float) -> Iterator[dict]:
        payload = {
            "model": model.id,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {config.HF_TOKEN}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                config.HF_ROUTER_URL, headers=headers, json=payload,
                stream=True, timeout=(15, 180),
            )
        except requests.RequestException as e:
            raise _ModelFailover(config.COOLDOWN_PROVIDER_ERROR,
                                 f"network error: {e}")

        if resp.status_code != 200:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            resp.close()
            if self._is_credit_error(resp.status_code, body):
                raise _ModelFailover(config.COOLDOWN_CREDIT_EXHAUSTED,
                                     "free credits exhausted", credit=True)
            if resp.status_code == 429:
                raise _ModelFailover(config.COOLDOWN_RATE_LIMITED, "rate limited")
            raise _ModelFailover(config.COOLDOWN_PROVIDER_ERROR,
                                 f"HTTP {resp.status_code}")

        # Committed: this model is serving. Tell the UI which one.
        yield {"type": "model", "id": model.id, "label": model.label}

        # The router returns text/event-stream without a charset, so requests
        # would otherwise default to ISO-8859-1 and mangle multibyte UTF-8 (a
        # right single quote U+2019 -> "â" + invisible control chars, i.e.
        # "Respondent’s" shown as "Respondentâs"). Force UTF-8 decoding.
        resp.encoding = "utf-8"

        got_any = False
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        got_any = True
                        yield {"type": "token", "text": piece}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        finally:
            resp.close()

        if not got_any:
            # Empty stream usually means the provider dropped us; try the next.
            raise _ModelFailover(config.COOLDOWN_PROVIDER_ERROR, "empty response")
        yield {"type": "done", "label": model.label}


class _ModelFailover(Exception):
    def __init__(self, cooldown: int, reason: str, credit: bool = False):
        super().__init__(reason)
        self.cooldown = cooldown
        self.reason = reason
        self.credit = credit
