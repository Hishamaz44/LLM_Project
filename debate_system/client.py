"""Thin wrapper around the OpenRouter API (OpenAI-compatible) with disk caching.

Caching rule: only a response that passes the caller's `validate` check is ever written
to disk. Bad/empty responses are retried (never cached), so a re-run replays only good
data and there is never a stale bad entry to work around.
"""

import os
import threading
from collections.abc import Callable

from openai import OpenAI

from .cache import Cache, make_key

_cache = Cache()
_client: OpenAI | None = None
_client_lock = threading.Lock()


# Lazily creates (and caches) the singleton OpenAI client, pointed at OpenRouter. Double-checked
# locking so concurrent workers (run_experiment --jobs) share one client instead of racing to
# build several. The underlying openai/httpx client is itself safe for concurrent requests.
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is not set (check your .env file)")
                _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


# Makes a single API call and returns non-empty text, or raises if the model gave us
# nothing usable (no choices / None content / blank) — the caller's retry loop handles it.
def _raw_call(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not response.choices:
        raise RuntimeError(f"{model} returned no choices")
    content = response.choices[0].message.content
    if content is None or not content.strip():
        finish = response.choices[0].finish_reason
        raise RuntimeError(f"{model} returned empty content (finish_reason={finish})")
    return content


# Calls an LLM with a single-turn prompt, serving from cache when possible.
def call_model(
    model: str,
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    slot: int = 0,
    validate: Callable[[str], bool] | None = None,
    max_attempts: int = 3,
) -> str:
    """Call `model` with `prompt` as a single user message, cached by (model, prompt, params, slot).

    `slot` exists so that repeated calls to the *same* model for different judge-panel
    seats produce independent (and independently cached) samples instead of one call
    being reused three times.

    Only a response that passes `validate` (if given) is cached and returned. Failed API
    calls and responses that don't validate are retried up to `max_attempts` times — with
    temperature > 0 each retry is a fresh sample — and are never cached.
    """
    key = make_key(model, prompt, max_tokens, temperature, slot)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            text = _raw_call(model, prompt, max_tokens, temperature)
            if validate is None or validate(text):
                _cache.set(key, text)
                return text
            last_error = ValueError(f"Response failed validation: {text!r}")
        except Exception as exc:  # transient API error, or empty content from _raw_call
            last_error = exc
    raise RuntimeError(
        f"call_model failed for {model} after {max_attempts} attempts"
    ) from last_error
