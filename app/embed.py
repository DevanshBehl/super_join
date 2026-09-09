"""Embeddings for the candidate funnel.

One vector per fact, built from a canonical string rather than the raw
sentence, so that two facts phrased differently in two documents land close
together. Vectors are cached by the hash of the string, which means a re-ingest
and any repeated subject wording cost nothing.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Iterable, Sequence

import config
from app import db, llm
from app.tracing import trace_llm_call


def canonical_string(fact: dict[str, Any]) -> str:
    """The string that gets embedded.

    Subject, predicate, period, and scope. Deliberately not the value: two
    facts about the same quantity must be neighbours whether or not they agree,
    otherwise a contradiction could never be discovered.
    """
    scope = fact.get("scope_tags") or []
    if isinstance(scope, str):
        scope = db.loads(scope, [])
    return " | ".join(
        [
            str(fact.get("subject_canonical") or fact.get("subject_raw") or ""),
            str(fact.get("predicate_canonical") or fact.get("predicate") or ""),
            str(fact.get("period_label_raw") or ""),
            ",".join(sorted(scope)),
        ]
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit_norm(vector: Sequence[float]) -> list[float]:
    """Normalise to unit length so cosine similarity is a plain dot product.

    Google recommends renormalising when a truncated output dimensionality is
    requested, and it also keeps DuckDB's cosine similarity numerically tidy.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0] * len(vector)
    return [float(v) / norm for v in vector]


def _call_embedding_api(batch: list[str]) -> list[list[float]]:
    from google.genai import types

    client = llm.get_client()
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=batch,
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIM,
            task_type="SEMANTIC_SIMILARITY",
            # Without an explicit timeout a stalled connection blocks the whole
            # ingest indefinitely, which is exactly what happened once.
            http_options=types.HttpOptions(timeout=int(config.LLM_TIMEOUT_SECONDS * 1000)),
        ),
    )
    return [_unit_norm(item.values) for item in response.embeddings]


def embed_texts(texts: Iterable[str], *, recorder: Any | None = None) -> dict[str, list[float]]:
    """Embed a set of strings, using the cache and batching the remainder."""
    unique = sorted({t for t in texts if t and t.strip()})
    if not unique:
        return {}

    hashes = {text: text_hash(text) for text in unique}
    cached = db.get_cached_embeddings(list(hashes.values()), config.EMBEDDING_MODEL)
    out: dict[str, list[float]] = {t: cached[h] for t, h in hashes.items() if h in cached}
    if recorder is not None:
        for _ in range(len(out)):
            recorder.count_llm(cached=True)

    pending = [t for t in unique if t not in out]
    for start in range(0, len(pending), config.EMBEDDING_BATCH_SIZE):
        batch = pending[start : start + config.EMBEDDING_BATCH_SIZE]
        if not llm.llm_available():
            trace_llm_call(
                stage="embed", model=config.EMBEDDING_MODEL, prompt_version="embed-1",
                content_hash=text_hash("|".join(batch)), cached=False, latency_ms=0.0,
                error="LLM disabled or GEMINI_API_KEY missing",
            )
            if recorder is not None:
                recorder.add_error("embedding_unavailable", "no API key", n_texts=len(batch))
            continue
        vectors = _embed_batch_with_retries(batch, recorder=recorder)
        if vectors is None:
            continue
        db.put_cached_embeddings(
            [(hashes[text], config.EMBEDDING_MODEL, vector) for text, vector in zip(batch, vectors)]
        )
        out.update(dict(zip(batch, vectors)))
    return out


def _embed_batch_with_retries(batch: list[str], *, recorder: Any | None = None) -> list[list[float]] | None:
    last_error = None
    for attempt in range(config.LLM_MAX_RETRIES):
        llm.limiter_for(config.EMBEDDING_MODEL).acquire()
        started = time.perf_counter()
        try:
            vectors = _call_embedding_api(batch)
            latency_ms = (time.perf_counter() - started) * 1000
            if recorder is not None:
                recorder.count_llm(cached=False)
            trace_llm_call(
                stage="embed", model=config.EMBEDDING_MODEL, prompt_version="embed-1",
                content_hash=text_hash("|".join(batch)), cached=False, latency_ms=latency_ms,
                n_results=len(vectors), extra={"batch_size": len(batch), "attempt": attempt + 1},
            )
            return vectors
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            trace_llm_call(
                stage="embed", model=config.EMBEDDING_MODEL, prompt_version="embed-1",
                content_hash=text_hash("|".join(batch)), cached=False,
                latency_ms=(time.perf_counter() - started) * 1000, error=last_error,
                extra={"attempt": attempt + 1},
            )
            if llm.is_daily_quota_error(exc) or attempt + 1 >= config.LLM_MAX_RETRIES:
                break
            stated = llm.retry_delay_seconds(exc)
            delay = stated if stated is not None else config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
            llm.limiter_for(config.EMBEDDING_MODEL).back_off_until(delay)
            time.sleep(delay)
    if recorder is not None:
        recorder.add_error("embedding_failed", last_error or "unknown", n_texts=len(batch))
    return None


def embed_facts(facts: Sequence[dict[str, Any]], *, recorder: Any | None = None) -> int:
    """Embed and store one vector per fact. Returns how many were written."""
    pairs = [(fact["fact_id"], canonical_string(fact)) for fact in facts]
    vectors = embed_texts([text for _, text in pairs], recorder=recorder)
    rows = [(fact_id, text, vectors[text]) for fact_id, text in pairs if text in vectors]
    return db.upsert_fact_embeddings(rows)


def embed_one(text: str) -> list[float] | None:
    result = embed_texts([text])
    return result.get(text)


def make_entity_embedder() -> Any:
    """An embed function for the entity resolver, or None when unavailable."""
    if not llm.llm_available():
        return None

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = embed_texts(texts)
        return [vectors.get(t, [0.0] * config.EMBEDDING_DIM) for t in texts]

    return embed
