"""Subject canonicalization and alias resolution.

The ladder is cheapest first and most expensive last: exact canonical match,
then alias match, then fuzzy string match, then embedding similarity, and only
if all four fail is a new entity created. Every raw form that resolves onto an
entity is kept in ``aliases``, so a merge never destroys the wording the
document actually used, and a close call can be reviewed later.

Addresses get their own path. The brief calls them out, and they behave
differently from names: the postal code is a near unique key that survives
every formatting difference, so postal code plus a fuzzy street match is
treated as a strong signal.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from rapidfuzz import fuzz

import config
from app import db
from app.normalize import (
    LEGAL_SUFFIXES,
    canonical_subject,
    clean_text,
    normalize_address,
)

EmbedFn = Callable[[list[str]], list[list[float]]]

_PERSON_HONORIFICS = ("mr", "mrs", "ms", "dr", "prof", "shri", "smt", "sh")
_ADDRESS_HINTS = re.compile(
    r"\b(road|street|lane|avenue|marg|nagar|sector|phase|block|plot|floor|building|tower|"
    r"complex|estate|village|district|po box|p\.o\.|opposite|near)\b",
    re.IGNORECASE,
)


def make_entity_id(canonical_name: str, entity_type: str) -> str:
    digest = hashlib.sha256(f"{entity_type}\x00{canonical_name}".encode("utf-8"))
    return digest.hexdigest()[:24]


def guess_entity_type(subject_raw: str) -> str:
    """A light classifier. Wrong guesses cost nothing beyond a separate entity."""
    text = clean_text(subject_raw)
    lowered = text.lower()
    address = normalize_address(text)
    if address.postal_code or _ADDRESS_HINTS.search(text):
        return "address"
    if any(re.search(rf"(?:^|\W){re.escape(s)}\.?(?:$|\W)", lowered) for s in LEGAL_SUFFIXES):
        return "organisation"
    if any(lowered.startswith(f"{h} ") or lowered.startswith(f"{h}. ") for h in _PERSON_HONORIFICS):
        return "person"
    words = text.split()
    if 1 < len(words) <= 4 and all(w[:1].isupper() for w in words if w[:1].isalpha()) and not any(
        ch.isdigit() for ch in text
    ):
        return "person_or_organisation"
    return "concept"


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass(slots=True)
class EntityMatch:
    entity_id: str
    canonical_name: str
    method: str  # exact | alias | fuzzy | embedding | address | created
    score: float
    created: bool = False
    close_call: bool = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EntityResolver:
    """Resolves raw subjects onto entities, holding the table in memory.

    The whole entity set is small enough to keep in memory during an ingest,
    which turns the fuzzy stage into a fast scan and avoids a query per fact.
    """

    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self.embed_fn = embed_fn
        self.entities: dict[str, EntityRecord] = {}
        self.by_canonical: dict[str, str] = {}
        self.by_alias: dict[str, str] = {}
        self.close_calls: list[dict[str, Any]] = []
        self._dirty: set[str] = set()

    # -- loading and saving -------------------------------------------------

    def load(self) -> "EntityResolver":
        for row in db.query("SELECT entity_id, canonical_name, entity_type, aliases, embedding FROM entities"):
            record = EntityRecord(
                entity_id=row["entity_id"],
                canonical_name=row["canonical_name"],
                entity_type=row["entity_type"],
                aliases=db.loads(row["aliases"], []),
                embedding=list(row["embedding"]) if row["embedding"] is not None else None,
            )
            self._index(record)
        return self

    def _index(self, record: EntityRecord) -> None:
        self.entities[record.entity_id] = record
        self.by_canonical[record.canonical_name] = record.entity_id
        for alias in record.aliases:
            self.by_alias.setdefault(canonical_subject(alias), record.entity_id)

    def flush(self) -> int:
        """Write back only the entities that changed."""
        for entity_id in self._dirty:
            record = self.entities[entity_id]
            db.upsert_entity(
                {
                    "entity_id": record.entity_id,
                    "canonical_name": record.canonical_name,
                    "entity_type": record.entity_type,
                    "aliases": record.aliases,
                    "embedding": record.embedding,
                }
            )
        written = len(self._dirty)
        self._dirty.clear()
        return written

    # -- resolution ---------------------------------------------------------

    def resolve(self, subject_raw: str, *, entity_type: str | None = None) -> EntityMatch | None:
        """Resolve one raw subject, creating an entity only as a last resort."""
        raw = clean_text(subject_raw)
        canonical = canonical_subject(raw)
        if not canonical:
            return None
        kind = entity_type or guess_entity_type(raw)

        entity_id = self.by_canonical.get(canonical)
        if entity_id:
            return self._attach(entity_id, raw, "exact", 100.0)

        entity_id = self.by_alias.get(canonical)
        if entity_id:
            return self._attach(entity_id, raw, "alias", 100.0)

        if kind == "address":
            match = self._match_address(raw)
            if match is not None:
                return match

        match = self._match_fuzzy(canonical, kind)
        if match is not None:
            return match

        match = self._match_embedding(canonical, kind)
        if match is not None:
            return match

        return self._create(canonical, raw, kind)

    def _attach(self, entity_id: str, raw: str, method: str, score: float, close_call: bool = False) -> EntityMatch:
        record = self.entities[entity_id]
        if raw and raw not in record.aliases and canonical_subject(raw) != record.canonical_name:
            record.aliases.append(raw)
            self.by_alias.setdefault(canonical_subject(raw), entity_id)
            self._dirty.add(entity_id)
        return EntityMatch(entity_id, record.canonical_name, method, score, False, close_call)

    def _match_fuzzy(self, canonical: str, kind: str) -> EntityMatch | None:
        best_id, best_score = None, 0.0
        for record in self.entities.values():
            if not _types_compatible(record.entity_type, kind):
                continue
            score = fuzz.token_sort_ratio(canonical, record.canonical_name)
            if score > best_score:
                best_id, best_score = record.entity_id, score
        if best_id is None or best_score < config.ENTITY_FUZZY_THRESHOLD:
            if best_id is not None and best_score >= config.ENTITY_FUZZY_REVIEW_THRESHOLD:
                self.close_calls.append(
                    {
                        "subject": canonical,
                        "candidate": self.entities[best_id].canonical_name,
                        "score": round(best_score, 1),
                        "decision": "kept_separate",
                    }
                )
            return None
        close = best_score < config.ENTITY_FUZZY_THRESHOLD + 3
        if close:
            self.close_calls.append(
                {
                    "subject": canonical,
                    "candidate": self.entities[best_id].canonical_name,
                    "score": round(best_score, 1),
                    "decision": "merged",
                }
            )
        return self._attach(best_id, canonical, "fuzzy", float(best_score), close_call=close)

    def _match_embedding(self, canonical: str, kind: str) -> EntityMatch | None:
        if self.embed_fn is None or not self.entities:
            return None
        candidates = [r for r in self.entities.values() if r.embedding and _types_compatible(r.entity_type, kind)]
        if not candidates:
            return None
        try:
            vector = self.embed_fn([canonical])[0]
        except Exception:  # pragma: no cover - an embedding outage must not stop ingest
            return None
        best_record, best_score = None, 0.0
        for record in candidates:
            score = _cosine(vector, record.embedding or [])
            if score > best_score:
                best_record, best_score = record, score
        if best_record is None or best_score < config.ENTITY_EMBED_THRESHOLD:
            return None
        return self._attach(best_record.entity_id, canonical, "embedding", float(best_score))

    def _match_address(self, raw: str) -> EntityMatch | None:
        target = normalize_address(raw)
        if not target.postal_code:
            return None
        for record in self.entities.values():
            if record.entity_type != "address":
                continue
            for candidate_text in [record.canonical_name, *record.aliases]:
                candidate = normalize_address(candidate_text)
                if candidate.postal_code != target.postal_code:
                    continue
                score = fuzz.token_set_ratio(candidate.normalized, target.normalized)
                if score >= config.ADDRESS_FUZZY_THRESHOLD:
                    return self._attach(record.entity_id, raw, "address", float(score))
        return None

    def _create(self, canonical: str, raw: str, kind: str) -> EntityMatch:
        entity_id = make_entity_id(canonical, kind)
        record = EntityRecord(
            entity_id=entity_id,
            canonical_name=canonical,
            entity_type=kind,
            aliases=[raw] if raw and canonical_subject(raw) != canonical else [],
        )
        if self.embed_fn is not None:
            try:
                record.embedding = self.embed_fn([canonical])[0]
            except Exception:  # pragma: no cover - optional enrichment
                record.embedding = None
        self._index(record)
        self._dirty.add(entity_id)
        return EntityMatch(entity_id, canonical, "created", 100.0, created=True)

    # -- bulk ---------------------------------------------------------------

    def prewarm(self, subjects: Iterable[str]) -> None:
        """Embed every distinct subject in one batched pass before resolving.

        Without this the resolver makes one network call per new entity, which
        is both slow and fragile. After this the per entity calls are cache
        hits, so resolution runs at memory speed.
        """
        if self.embed_fn is None:
            return
        distinct = sorted({canonical_subject(s) for s in subjects if s})
        if not distinct:
            return
        try:
            self.embed_fn(distinct)
        except Exception:  # pragma: no cover - the resolver degrades gracefully
            pass

    def assign(self, facts: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Attach ``entity_id`` to each fact in place and report the method mix."""
        facts = list(facts)
        self.prewarm(f.get("subject_raw", "") for f in facts)
        counts: dict[str, int] = {}
        for fact in facts:
            match = self.resolve(fact.get("subject_raw", ""))
            if match is None:
                continue
            fact["entity_id"] = match.entity_id
            counts[match.method] = counts.get(match.method, 0) + 1
        return counts


def _types_compatible(a: str, b: str) -> bool:
    """A guessed type never blocks a merge with an unresolved or generic type."""
    loose = {"unknown", "concept", "person_or_organisation"}
    return a == b or a in loose or b in loose
