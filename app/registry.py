"""The dynamically evolving predicate schema.

Extraction is free to coin a new predicate whenever an existing one does not
fit. Left alone that produces a thousand near synonyms. The registry closes the
loop: every predicate ever used is recorded with its frequency, and the most
frequent ones are fed back into later extraction prompts as a vocabulary hint.
The schema therefore converges on the vocabulary the corpus actually needs,
without anyone having to write that vocabulary down in advance.
"""

from __future__ import annotations

from typing import Any, Iterable

import config
from app import db
from app.normalize import canonical_predicate


def vocabulary_hint(limit: int | None = None) -> list[dict[str, Any]]:
    """The top predicates by frequency, for injection into an extraction prompt."""
    return db.top_predicates(limit or config.PREDICATE_HINT_TOP_N)


def render_vocabulary_hint(limit: int | None = None) -> str:
    """Render the hint as compact lines. Empty on a cold database, which is fine."""
    entries = vocabulary_hint(limit)
    if not entries:
        return "(the registry is empty, so coin predicates as needed)"
    lines = []
    for entry in entries:
        dimension = entry.get("expected_unit_dimension") or "any"
        lines.append(
            f"- {entry['predicate_canonical']}"
            f" (type: {entry.get('expected_value_type') or 'any'}, dimension: {dimension},"
            f" used {entry.get('n_facts', 0)} times)"
        )
    return "\n".join(lines)


def register_facts(facts: Iterable[dict[str, Any]]) -> int:
    """Append every new predicate and increase counts for the ones already known."""
    seen: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for fact in facts:
        predicate = fact.get("predicate_canonical") or canonical_predicate(fact.get("predicate", ""))
        if not predicate:
            continue
        counts[predicate] = counts.get(predicate, 0) + 1
        if predicate not in seen:
            seen[predicate] = {
                "predicate_canonical": predicate,
                "description": _describe(fact),
                "expected_value_type": fact.get("value_type", "numeric"),
                "expected_unit_dimension": fact.get("unit_dimension"),
                "example_fact_id": fact.get("fact_id"),
                "first_seen_doc_id": fact.get("doc_id"),
            }
    for predicate, entry in seen.items():
        db.bump_predicate(entry, increment=counts[predicate])
    return len(seen)


def _describe(fact: dict[str, Any]) -> str:
    """A one line description built from the first fact that used the predicate."""
    subject = fact.get("subject_canonical") or fact.get("subject_raw") or "a subject"
    value = fact.get("value_raw") or ""
    unit = fact.get("unit_canonical") or ""
    tail = f" {unit}".rstrip()
    return f"first seen as: {subject} -> {value}{tail}"[:300]


def snapshot() -> list[dict[str, Any]]:
    """The registry as it currently stands, most used first."""
    return db.query(
        "SELECT predicate_canonical, description, expected_value_type, expected_unit_dimension,"
        " example_fact_id, first_seen_doc_id, n_facts, created_at"
        " FROM predicate_registry ORDER BY n_facts DESC, predicate_canonical"
    )
