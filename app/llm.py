"""One Gemini client, one cache, one trace line per call.

Extraction, adjudication, and document metadata all go through
``generate_structured``. That is the only place a network call is made for text
generation, so the cache, the retry policy, and the trace are guaranteed to
cover every call rather than most of them.

The cache key deliberately hashes the *content* being reasoned about, not the
rendered prompt. The extraction prompt carries a vocabulary hint drawn from the
predicate registry, which grows as documents are ingested. Hashing the rendered
prompt would make that hint invalidate the cache, and re-ingesting an unchanged
PDF would spend tokens again. Hashing the content keeps the re-ingest promise
literally true: zero calls.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

import config
from app import db
from app.tracing import trace_llm_call

T = TypeVar("T", bound=BaseModel)

_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()


class LLMUnavailable(RuntimeError):
    """Raised when a call is attempted with the model layer switched off."""


def content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def llm_available() -> bool:
    return bool(config.LLM_ENABLED and config.GEMINI_API_KEY)


def get_client() -> Any:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            if not config.GEMINI_API_KEY:
                raise LLMUnavailable("GEMINI_API_KEY is not set")
            from google import genai

            _CLIENT = genai.Client(api_key=config.GEMINI_API_KEY)
        return _CLIENT


@dataclass(slots=True)
class LLMResult:
    parsed: Any | None
    raw_text: str
    cached: bool
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.error is None


class RateLimiter:
    """A token bucket, one per model, so bursts never exceed the per minute cap.

    Retrying into a rate limit is not the same as respecting it: the retry
    still consumes a request slot and the call still fails. Pacing the calls up
    front is what actually makes a long run finish.
    """

    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait > 0:
            time.sleep(wait)

    def back_off_until(self, seconds: float) -> None:
        """Push every waiting caller past a server stated retry delay."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def limiter_for(model: str) -> RateLimiter:
    with _LIMITER_LOCK:
        if model not in _LIMITERS:
            per_minute = (
                config.EMBEDDING_REQUESTS_PER_MINUTE
                if "embedding" in model
                else config.LLM_REQUESTS_PER_MINUTE
            )
            _LIMITERS[model] = RateLimiter(per_minute)
        return _LIMITERS[model]


def retry_delay_seconds(exc: Exception) -> float | None:
    """Read the retry delay the server asked for, when it stated one."""
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", str(exc))
    return float(match.group(1)) if match else None


# A permanent quota failure trips this breaker once, and every later call
# fails immediately instead of spending its retries on a request that cannot
# succeed. Without it an exhausted balance turns a 1,400 chunk ingest into
# hours of guaranteed 429s, because the error is retryable by shape and
# permanent in fact.
_BREAKER_LOCK = threading.Lock()
_BREAKER_REASON: str | None = None


def is_daily_quota_error(exc: Exception) -> bool:
    """Distinguish quota failures that waiting cannot fix from the per minute cap.

    A per day cap, an exhausted prepaid balance and a disabled billing account
    all arrive as 429 RESOURCE_EXHAUSTED, the same status as the per minute
    cap. Only the per minute cap clears by sleeping. Retrying the others
    consumes the remaining attempts and still fails, so they have to be told
    apart by message rather than by status.
    """
    text = str(exc).lower()
    permanent = (
        "perday",
        "per day",
        "prepayment credits",
        "credits are depleted",
        "billing",
        "free_tier",
        "exceeded your current quota",
    )
    return any(marker in text for marker in permanent)


def trip_breaker(reason: str) -> None:
    """Record a permanent quota failure so later calls stop early."""
    global _BREAKER_REASON
    with _BREAKER_LOCK:
        if _BREAKER_REASON is None:
            _BREAKER_REASON = reason


def breaker_reason() -> str | None:
    with _BREAKER_LOCK:
        return _BREAKER_REASON


def reset_breaker() -> None:
    """Clear the breaker. Called by the tests and after a key is replaced."""
    global _BREAKER_REASON
    with _BREAKER_LOCK:
        _BREAKER_REASON = None


def thinking_config(model: str, level: str) -> Any:
    """Build a thinking configuration that the given model actually accepts.

    Gemini 3 and later take a ``thinking_level``. Gemini 2.5 takes a
    ``thinking_budget`` in tokens and rejects the level. Callers ask for "low"
    or "high" and this translates.
    """
    from google.genai import types

    level = (level or "low").strip().lower()
    if re.match(r"^gemini-([3-9]|\d{2,})", model):
        return types.ThinkingConfig(thinking_level=level)
    return types.ThinkingConfig(thinking_budget=0 if level == "low" else 2048)


def _retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "resource_exhausted", "503", "unavailable", "500", "internal",
                       "deadline", "timeout", "connection", "temporarily")
    )


def generate_structured(
    *,
    stage: str,
    prompt: str,
    schema: Type[T],
    model: str,
    prompt_version: str,
    cache_content_hash: str,
    thinking: str = "low",
    temperature: float = 0.0,
    recorder: Any | None = None,
) -> LLMResult:
    """Structured generation with caching, retries, and a trace line.

    Returns an ``LLMResult`` rather than raising, because a single failed chunk
    must not abort an ingest. The failure is recorded and the pipeline moves on.
    """
    key = db.cache_key(cache_content_hash, prompt_version, model, config.SCHEMA_VERSION)
    cached_response = db.get_llm_cache(key)
    if cached_response is not None:
        try:
            parsed = schema.model_validate_json(cached_response)
            if recorder is not None:
                recorder.count_llm(cached=True)
            trace_llm_call(
                stage=stage, model=model, prompt_version=prompt_version,
                content_hash=cache_content_hash, cached=True, latency_ms=0.0,
            )
            return LLMResult(parsed=parsed, raw_text=cached_response, cached=True)
        except ValidationError:
            pass  # A stale cache entry is simply refetched.

    if not llm_available():
        message = "LLM disabled or GEMINI_API_KEY missing"
        trace_llm_call(
            stage=stage, model=model, prompt_version=prompt_version,
            content_hash=cache_content_hash, cached=False, latency_ms=0.0, error=message,
        )
        return LLMResult(parsed=None, raw_text="", cached=False, error=message)

    tripped = breaker_reason()
    if tripped is not None:
        # The quota is gone for the whole run, not just this call. Failing here
        # keeps the cache and the trace honest without spending a request.
        message = f"quota exhausted, calls stopped: {tripped}"
        trace_llm_call(
            stage=stage, model=model, prompt_version=prompt_version,
            content_hash=cache_content_hash, cached=False, latency_ms=0.0, error=message,
        )
        if recorder is not None:
            recorder.add_error("llm_quota_exhausted", message, stage=stage)
        return LLMResult(parsed=None, raw_text="", cached=False, error=message)

    from google.genai import types

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        thinking_config=thinking_config(model, thinking),
        http_options=types.HttpOptions(timeout=int(config.LLM_TIMEOUT_SECONDS * 1000)),
    )

    last_error: str | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        limiter_for(model).acquire()
        started = time.perf_counter()
        try:
            response = get_client().models.generate_content(
                model=model, contents=prompt, config=generation_config
            )
            latency_ms = (time.perf_counter() - started) * 1000
            raw_text = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
            tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
            tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0)
            parsed = schema.model_validate_json(raw_text) if raw_text else None
            if parsed is None:
                raise ValueError("empty response body")
            db.put_llm_cache(
                key, cache_content_hash, prompt_version, model, config.SCHEMA_VERSION,
                raw_text, tokens_in, tokens_out,
            )
            if recorder is not None:
                recorder.count_llm(cached=False, tokens_in=tokens_in, tokens_out=tokens_out)
            trace_llm_call(
                stage=stage, model=model, prompt_version=prompt_version,
                content_hash=cache_content_hash, cached=False, latency_ms=latency_ms,
                tokens_in=tokens_in, tokens_out=tokens_out,
                extra={"attempt": attempt + 1, "prompt_chars": len(prompt)},
            )
            return LLMResult(parsed, raw_text, False, tokens_in, tokens_out, None, latency_ms)
        except Exception as exc:  # noqa: BLE001 - the failure is reported, not swallowed
            last_error = f"{type(exc).__name__}: {exc}"[:400]
            latency_ms = (time.perf_counter() - started) * 1000
            trace_llm_call(
                stage=stage, model=model, prompt_version=prompt_version,
                content_hash=cache_content_hash, cached=False, latency_ms=latency_ms,
                error=last_error, extra={"attempt": attempt + 1},
            )
            if is_daily_quota_error(exc):
                # Does not clear by waiting, so stop rather than burn the
                # remaining retries, and stop every later call in this run too.
                trip_breaker(last_error)
                break
            if attempt + 1 >= config.LLM_MAX_RETRIES or not _retryable(exc):
                break
            stated = retry_delay_seconds(exc)
            delay = stated if stated is not None else config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
            limiter_for(model).back_off_until(delay)
            time.sleep(delay + random.uniform(0, 0.5))

    if recorder is not None:
        recorder.add_error("llm_call_failed", last_error or "unknown", stage=stage)
    return LLMResult(parsed=None, raw_text="", cached=False, error=last_error or "unknown")


def dump_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
