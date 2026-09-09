"""A tracer a grader can actually read.

Every LLM call appends one JSON object to ``data/traces.jsonl``. Every pipeline
stage writes one row to the ``runs`` table. There is no hosted dashboard and no
account to create: ``tail -f data/traces.jsonl`` is the observability story.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config
from app import db

_WRITE_LOCK = threading.Lock()


def trace_event(event: dict[str, Any]) -> None:
    """Append one JSON object to the trace file. Never raises into the caller."""
    payload = {"ts": datetime.utcnow().isoformat(timespec="milliseconds"), **event}
    try:
        config.TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, default=str, ensure_ascii=False)
        with _WRITE_LOCK:
            with config.TRACE_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:  # pragma: no cover - tracing must never break a pipeline
        pass


def trace_llm_call(
    *,
    stage: str,
    model: str,
    prompt_version: str,
    content_hash: str,
    cached: bool,
    latency_ms: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    n_results: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    trace_event(
        {
            "type": "llm_call",
            "stage": stage,
            "model": model,
            "prompt_version": prompt_version,
            "content_hash": content_hash,
            "cached": cached,
            "latency_ms": round(latency_ms, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "n_results": n_results,
            "error": error,
            **(extra or {}),
        }
    )


@dataclass
class RunRecorder:
    """Accumulates counters for one pipeline stage and writes a ``runs`` row.

    Used as a context manager. Exceptions are recorded and re-raised, so a
    failed stage still leaves a row explaining what happened.
    """

    stage: str
    doc_id: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    n_items: int = 0
    n_llm_calls: int = 0
    n_cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    _t0: float = field(default_factory=time.perf_counter)

    def add_error(self, kind: str, detail: str, **fields: Any) -> None:
        self.errors.append({"kind": kind, "detail": str(detail)[:600], **fields})

    def count_llm(self, cached: bool, tokens_in: int = 0, tokens_out: int = 0) -> None:
        if cached:
            self.n_cache_hits += 1
        else:
            self.n_llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def __enter__(self) -> "RunRecorder":
        trace_event({"type": "stage_start", "stage": self.stage, "doc_id": self.doc_id, "run_id": self.run_id})
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.add_error("exception", f"{exc_type.__name__}: {exc}")
        self.finished_at = datetime.utcnow()
        self.notes.setdefault("seconds", round(self.elapsed_seconds, 3))
        self.notes.setdefault("cache_hits", self.n_cache_hits)
        db.record_run(
            {
                "run_id": self.run_id,
                "doc_id": self.doc_id,
                "stage": self.stage,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "n_items": self.n_items,
                "n_llm_calls": self.n_llm_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "errors": self.errors,
                "notes": self.notes,
            }
        )
        trace_event(
            {
                "type": "stage_end",
                "stage": self.stage,
                "doc_id": self.doc_id,
                "run_id": self.run_id,
                "n_items": self.n_items,
                "n_llm_calls": self.n_llm_calls,
                "n_cache_hits": self.n_cache_hits,
                "seconds": round(self.elapsed_seconds, 3),
                "n_errors": len(self.errors),
            }
        )
        return False


def read_traces(limit: int = 500, kind: str | None = None) -> list[dict[str, Any]]:
    """Read back the tail of the trace file, newest last."""
    if not config.TRACE_PATH.exists():
        return []
    lines = config.TRACE_PATH.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit * 4 :]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if kind is None or item.get("type") == kind:
            out.append(item)
    return out[-limit:]
