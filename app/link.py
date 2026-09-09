"""Candidate generation funnel and the deterministic rule engine.

Two ideas do the work here.

The funnel replaces the naive all pairs comparison. Facts are blocked by
collection, then by entity or predicate, then by value type and unit dimension.
Only pairs that survive blocking have their cosine similarity computed, and
only the nearest few of those survive retrieval. The count at each stage is
recorded, because the shape of that funnel is the engineering result.

The rule engine then decides as much as it can without a model. Each check
appends an entry to ``rule_trace``, and the trace is the explanation: a reader
can see which check fired, on what inputs, and what it concluded. Only pairs
that no rule can settle are sent to the adjudicator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from rapidfuzz import fuzz

import config
from app import db
from app.normalize import (
    Period,
    UnitInfo,
    clean_text,
    convert,
    normalize_address,
    period_contains,
    periods_equal,
    periods_overlap,
    relative_delta,
    relative_tolerance,
    scope_shift,
)

FX_KEYS = ("exchange_rate", "fx_rate", "conversion_rate", "usd_inr", "rate_used")


@dataclass(slots=True)
class PairDecision:
    """The outcome of running the chain over one candidate pair.

    ``relation is None`` means no rule could settle it, which is the only way a
    pair reaches the adjudicator.
    """

    relation: str | None
    confidence: float = 0.5
    trace: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    numeric_delta: float | None = None
    relative_delta: float | None = None

    @property
    def needs_llm(self) -> bool:
        return self.relation is None


def make_link_id(fact_a: str, fact_b: str) -> str:
    first, second = sorted([fact_a, fact_b])
    digest = hashlib.sha256(f"{first}\x00{second}\x00{config.LINKER_VERSION}".encode("utf-8"))
    return digest.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Candidate funnel
# ---------------------------------------------------------------------------

_SCOPED_FACTS = """
    SELECT f.fact_id, f.doc_id, f.chunk_id, f.entity_id, f.predicate_canonical,
           f.value_type, f.unit_dimension, d.collection, fe.embedding
    FROM facts f
    JOIN evidence ev ON ev.evidence_id = f.evidence_id AND ev.verified
    JOIN documents d ON d.doc_id = f.doc_id
    JOIN fact_embeddings fe ON fe.fact_id = f.fact_id
"""

_BLOCK_PREDICATE = """
        a.fact_id < b.fact_id
    AND a.chunk_id <> b.chunk_id
    AND a.value_type = b.value_type
    AND (a.unit_dimension IS NOT DISTINCT FROM b.unit_dimension
         OR a.unit_dimension IS NULL OR b.unit_dimension IS NULL)
    AND (a.entity_id = b.entity_id OR a.predicate_canonical = b.predicate_canonical)
"""


def _collection_clause() -> str:
    return "" if config.CROSS_COLLECTION_LINKING else "    AND a.collection = b.collection\n"


def count_blocked_pairs(doc_id: str | None = None) -> int:
    """How many pairs survive blocking, before any vector is touched."""
    params: list[Any] = []
    doc_filter = ""
    if doc_id:
        doc_filter = "    AND (a.doc_id = ? OR b.doc_id = ?)\n"
        params = [doc_id, doc_id]
    sql = (
        f"WITH scoped AS ({_SCOPED_FACTS})\n"
        "SELECT COUNT(*) FROM scoped a JOIN scoped b ON "
        + _BLOCK_PREDICATE
        + _collection_clause()
        + doc_filter
    )
    return int(db.scalar(sql, params, 0))


def generate_candidates(
    doc_id: str | None = None,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Return deduplicated candidate pairs ordered deterministically.

    Passing ``doc_id`` restricts the funnel to pairs touching that document,
    which is what makes incremental ingest cheap: an existing document is never
    re-compared with another existing document.
    """
    top_k = config.NEIGHBOUR_TOP_K if top_k is None else top_k
    threshold = config.NEIGHBOUR_COSINE_THRESHOLD if threshold is None else threshold

    params: list[Any] = []
    doc_filter = ""
    if doc_id:
        doc_filter = "    AND (a.doc_id = ? OR b.doc_id = ?)\n"
        params = [doc_id, doc_id]

    existing_filter = (
        "    AND NOT EXISTS (SELECT 1 FROM links l WHERE l.fact_a = a.fact_id AND l.fact_b = b.fact_id)\n"
        if skip_existing
        else ""
    )

    sql = (
        f"WITH scoped AS ({_SCOPED_FACTS}),\n"
        "blocked AS (\n"
        "  SELECT a.fact_id AS fact_a, b.fact_id AS fact_b,\n"
        "         array_cosine_similarity(a.embedding, b.embedding) AS similarity\n"
        "  FROM scoped a JOIN scoped b ON "
        + _BLOCK_PREDICATE
        + _collection_clause()
        + doc_filter
        + existing_filter
        + ")\n"
        "SELECT fact_a, fact_b, similarity FROM blocked\n"
        "WHERE similarity >= ?\n"
        "QUALIFY row_number() OVER (PARTITION BY fact_a ORDER BY similarity DESC) <= ?\n"
        "     OR row_number() OVER (PARTITION BY fact_b ORDER BY similarity DESC) <= ?\n"
        "ORDER BY similarity DESC, fact_a, fact_b"
    )
    return db.query(sql, [*params, threshold, top_k, top_k])


def load_facts(fact_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load full fact rows with their evidence and document context."""
    if not fact_ids:
        return {}
    unique = sorted(set(fact_ids))
    placeholders = ", ".join(["?"] * len(unique))
    rows = db.query(
        f"""
        SELECT f.*, d.collection, d.title AS document_title, d.filename, d.as_of_date,
               d.published_date, ev.quote, ev.page_index, ev.page_label
        FROM facts f
        JOIN documents d ON d.doc_id = f.doc_id
        LEFT JOIN evidence ev ON ev.evidence_id = f.evidence_id
        WHERE f.fact_id IN ({placeholders})
        """,
        unique,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["scope_tags"] = db.loads(row.get("scope_tags"), [])
        row["qualifiers"] = db.loads(row.get("qualifiers"), {})
        out[row["fact_id"]] = row
    return out


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


def _period_of(fact: dict[str, Any]) -> Period:
    return Period(
        start=_as_date(fact.get("period_start")),
        end=_as_date(fact.get("period_end")),
        basis=fact.get("period_basis") or "unknown",
        label=fact.get("period_label_raw") or "",
    )


def _as_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _unit_info(fact: dict[str, Any]) -> UnitInfo:
    from app.normalize import canonical_unit

    canonical = fact.get("unit_canonical")
    dimension = fact.get("unit_dimension")
    if canonical is None:
        return UnitInfo(None, dimension, 1.0)
    reference = canonical_unit(canonical)
    to_base = reference.to_base if reference.dimension == dimension else 1.0
    return UnitInfo(canonical, dimension, to_base)


def _stated_fx_rate(*facts: dict[str, Any]) -> str | None:
    for fact in facts:
        qualifiers = fact.get("qualifiers") or {}
        for key, value in qualifiers.items():
            if any(marker in str(key).lower() for marker in FX_KEYS):
                return f"{key}={value}"
    return None


def _check(trace: list[dict[str, Any]], name: str, inputs: dict[str, Any], outcome: str, detail: str = "") -> None:
    trace.append({"name": name, "inputs": inputs, "outcome": outcome, "detail": detail})


def _label(fact: dict[str, Any]) -> str:
    return f"{fact.get('value_raw', '')}".strip() or str(fact.get("value_text") or "")


def evaluate_pair(a: dict[str, Any], b: dict[str, Any]) -> PairDecision:
    """Run the ordered chain over one pair, recording every check.

    Returns a decision with ``relation`` set when a rule settled the pair, or
    ``relation is None`` when the pair must go to the adjudicator.
    """
    trace: list[dict[str, Any]] = []

    # 1. Comparability -----------------------------------------------------
    if a.get("value_type") != b.get("value_type"):
        _check(trace, "value_type", {"a": a.get("value_type"), "b": b.get("value_type")}, "different")
        return PairDecision("unrelated", 0.95, trace, "The two facts carry different value types, so they are not comparable.")
    _check(trace, "value_type", {"a": a.get("value_type"), "b": b.get("value_type")}, "same")

    dim_a, dim_b = a.get("unit_dimension"), b.get("unit_dimension")
    if dim_a and dim_b and dim_a != dim_b:
        _check(trace, "unit_dimension", {"a": dim_a, "b": dim_b}, "different")
        return PairDecision(
            "unrelated", 0.95, trace,
            f"One value is measured in {dim_a} and the other in {dim_b}. No conversion exists between those dimensions.",
        )

    # A missing dimension used to be treated as a wildcard that matched
    # anything. That is right when both facts assert the same predicate and one
    # document simply omitted the unit, and wrong the rest of the time: it let
    # "$1 billion in revenues" corroborate "740 Mn express parcel shipments",
    # because blocking had already paired them on the shared entity and the
    # null dimension waved the mismatch through. When the predicates disagree,
    # the two facts have to share a stated dimension to be comparable at all.
    pred_a = a.get("predicate_canonical") or a.get("predicate")
    pred_b = b.get("predicate_canonical") or b.get("predicate")
    if pred_a != pred_b and not (dim_a and dim_b):
        _check(
            trace, "unit_dimension",
            {"a": dim_a, "b": dim_b, "predicate_a": pred_a, "predicate_b": pred_b},
            "incomparable_without_shared_dimension",
        )
        stated = dim_a or dim_b
        detail = (
            f"one side is measured in {stated} and the other states no unit"
            if stated
            else "neither side states a unit"
        )
        return PairDecision(
            "unrelated", 0.9, trace,
            f"The two facts assert different predicates ({pred_a} against {pred_b}) and {detail}, "
            "so there is nothing that makes their values comparable.",
        )

    _check(trace, "unit_dimension", {"a": dim_a, "b": dim_b}, "compatible")

    # 2. Currency ----------------------------------------------------------
    cur_a, cur_b = a.get("currency"), b.get("currency")
    if cur_a and cur_b and cur_a != cur_b:
        rate = _stated_fx_rate(a, b)
        if rate is None:
            _check(trace, "currency", {"a": cur_a, "b": cur_b, "stated_rate": None}, "different_no_rate")
            return PairDecision(
                "unrelated", 0.9, trace,
                f"The figures are in {cur_a} and {cur_b} and neither document states an exchange rate, so they cannot be compared.",
            )
        _check(trace, "currency", {"a": cur_a, "b": cur_b, "stated_rate": rate}, "different_with_rate",
               "a stated rate is present, so the pair is escalated for adjudication")
        return PairDecision(None, 0.5, trace, "")
    _check(trace, "currency", {"a": cur_a, "b": cur_b}, "same" if cur_a else "not_applicable")

    if a.get("value_type") != "numeric":
        return _evaluate_non_numeric(a, b, trace)

    # 3. Unit conversion ---------------------------------------------------
    value_a, value_b = a.get("value_num"), b.get("value_num")
    if value_a is None or value_b is None:
        _check(trace, "numeric_values", {"a": value_a, "b": value_b}, "missing")
        return PairDecision(None, 0.4, trace, "")

    unit_a, unit_b = _unit_info(a), _unit_info(b)
    conversion_detail = ""
    converted_b = value_b
    if unit_a.canonical and unit_b.canonical and unit_a.canonical != unit_b.canonical:
        candidate = convert(value_b, unit_b, unit_a)
        if candidate is None:
            _check(trace, "unit_conversion", {"from": unit_b.canonical, "to": unit_a.canonical}, "not_convertible")
            return PairDecision("unrelated", 0.9, trace, "The units cannot be converted into one another.")
        converted_b = candidate
        conversion_detail = f"{_label(b)} converted from {unit_b.canonical} to {unit_a.canonical} gives {converted_b:,.4g}"
        _check(trace, "unit_conversion", {"from": unit_b.canonical, "to": unit_a.canonical,
                                          "value_before": value_b, "value_after": converted_b},
               "converted", conversion_detail)
    else:
        _check(trace, "unit_conversion", {"unit": unit_a.canonical or unit_b.canonical}, "not_needed")

    delta = converted_b - value_a
    rel = relative_delta(value_a, converted_b)
    tolerance = relative_tolerance(
        str(a.get("value_raw") or ""), str(b.get("value_raw") or ""), value_a, converted_b,
        floor=config.RELATIVE_TOLERANCE_FLOOR,
    )
    agrees = rel <= tolerance
    _check(trace, "numeric_comparison",
           {"a": value_a, "b": converted_b, "absolute_delta": delta, "relative_delta": round(rel, 6),
            "tolerance": round(tolerance, 6)},
           "agrees_within_tolerance" if agrees else "differs",
           f"tolerance comes from the precision the raw strings claim: {a.get('value_raw')} and {b.get('value_raw')}")

    if conversion_detail and agrees:
        return PairDecision(
            "reconciled_by_unit", 0.9, trace,
            f"The same quantity is stated in two different units. {conversion_detail}, which matches "
            f"{_label(a)} within the precision either side claims.",
            delta, rel,
        )

    # 4. Periods -----------------------------------------------------------
    period_a, period_b = _period_of(a), _period_of(b)
    period_inputs = {
        "a": {"label": period_a.label, "start": str(period_a.start), "end": str(period_a.end), "basis": period_a.basis},
        "b": {"label": period_b.label, "start": str(period_b.start), "end": str(period_b.end), "basis": period_b.basis},
    }
    if period_a.start and period_b.start:
        if periods_equal(period_a, period_b):
            _check(trace, "period_comparison", period_inputs, "equal")
        elif period_contains(period_a, period_b) or period_contains(period_b, period_a):
            outer, inner = (period_a, period_b) if period_contains(period_a, period_b) else (period_b, period_a)
            _check(trace, "period_comparison", period_inputs, "nested",
                   f"{inner.label or 'one period'} falls inside {outer.label or 'the other period'}")
            return PairDecision(
                "refines", 0.9, trace,
                f"These are not in conflict. {inner.label or 'the shorter period'} sits inside "
                f"{outer.label or 'the longer period'}, so the smaller figure is a part of the larger one.",
                delta, rel,
            )
        elif not periods_overlap(period_a, period_b):
            _check(trace, "period_comparison", period_inputs, "disjoint")
            return PairDecision(
                "reconciled_by_period", 0.92, trace,
                f"The figures cover different periods, {period_a.label or period_a.start} against "
                f"{period_b.label or period_b.start}, so a difference between them is expected rather than contradictory.",
                delta, rel,
            )
        else:
            _check(trace, "period_comparison", period_inputs, "overlapping_not_equal",
                   "periods overlap without being equal, which is a contradiction candidate")
    else:
        _check(trace, "period_comparison", period_inputs, "unknown_on_one_side")

    # 5. Scope -------------------------------------------------------------
    tags_a = a.get("scope_tags") or []
    tags_b = b.get("scope_tags") or []
    shifted, reasons = scope_shift(tags_a, tags_b)
    if shifted and not agrees:
        _check(trace, "scope_comparison", {"a": tags_a, "b": tags_b}, "scope_shift", "; ".join(reasons))
        return PairDecision(
            "reconciled_by_scope", 0.85, trace,
            f"The two figures are measured on a different basis ({'; '.join(reasons)}), which accounts for the gap "
            f"between {_label(a)} and {_label(b)}.",
            delta, rel,
        )
    _check(trace, "scope_comparison", {"a": tags_a, "b": tags_b}, "scope_shift" if shifted else "comparable")

    # 6. Vintage -----------------------------------------------------------
    as_of_a, as_of_b = _as_date(a.get("as_of_date")), _as_date(b.get("as_of_date"))
    same_claim = (
        a.get("predicate_canonical") == b.get("predicate_canonical")
        and a.get("entity_id") == b.get("entity_id")
        and periods_equal(period_a, period_b)
    )
    if same_claim and as_of_a and as_of_b and as_of_a != as_of_b and not agrees:
        later, earlier = (a, b) if as_of_a > as_of_b else (b, a)
        _check(trace, "document_vintage",
               {"a": {"doc": a.get("filename"), "as_of": str(as_of_a)},
                "b": {"doc": b.get("filename"), "as_of": str(as_of_b)}},
               "different_vintage")
        return PairDecision(
            "reconciled_by_vintage", 0.85, trace,
            f"Both documents report the same measure for the same period, but "
            f"{later.get('filename')} speaks as of {max(as_of_a, as_of_b)} while "
            f"{earlier.get('filename')} speaks as of {min(as_of_a, as_of_b)}. The later figure is a revision "
            f"of the earlier one rather than a contradiction of it.",
            delta, rel,
        )
    _check(trace, "document_vintage",
           {"as_of_a": str(as_of_a), "as_of_b": str(as_of_b), "same_claim": same_claim},
           "same_or_not_applicable")

    # 7. Numeric agreement -------------------------------------------------
    if agrees:
        rounding = rel > 0 and rel <= tolerance
        explanation = (
            f"{_label(a)} and {_label(b)} are the same quantity stated differently; they agree within the "
            f"precision either side claims."
        )
        if rounding and abs(delta) > 0:
            explanation += (
                f" The residual difference of {abs(delta):,.4g} is a rounding artefact: one side is written to a "
                f"precision the other does not claim."
            )
        _check(trace, "agreement", {"relative_delta": round(rel, 6), "tolerance": round(tolerance, 6)}, "corroborates")
        return PairDecision("corroborates", 0.9, trace, explanation, delta, rel)

    # 8. Residue -----------------------------------------------------------
    if rel < config.MATERIAL_GAP_RELATIVE:
        _check(trace, "agreement", {"relative_delta": round(rel, 6), "tolerance": round(tolerance, 6)},
               "minor_gap", "the gap is below the material threshold but above the claimed precision")
        return PairDecision(
            "corroborates", 0.6, trace,
            f"{_label(a)} and {_label(b)} differ by {rel * 100:.2f} per cent, which is below the materiality "
            f"threshold but larger than the precision either side claims.",
            delta, rel,
        )
    _check(trace, "agreement", {"relative_delta": round(rel, 6), "tolerance": round(tolerance, 6)},
           "material_gap", "no rule settles this pair, so it is sent to the adjudicator")
    return PairDecision(None, 0.5, trace, "", delta, rel)


def _evaluate_non_numeric(a: dict[str, Any], b: dict[str, Any], trace: list[dict[str, Any]]) -> PairDecision:
    """The chain for string, boolean, and date facts."""
    text_a = clean_text(str(a.get("value_text") or a.get("value_raw") or ""))
    text_b = clean_text(str(b.get("value_text") or b.get("value_raw") or ""))

    address_a, address_b = normalize_address(text_a), normalize_address(text_b)
    if address_a.postal_code and address_a.postal_code == address_b.postal_code:
        score = fuzz.token_set_ratio(address_a.normalized, address_b.normalized)
        if score >= config.ADDRESS_FUZZY_THRESHOLD:
            _check(trace, "address_comparison",
                   {"postal_code": address_a.postal_code, "similarity": round(score, 1)}, "same_address")
            return PairDecision(
                "corroborates", 0.9, trace,
                "The two addresses share a postal code and normalize to the same street address, so they refer to "
                "the same place written two different ways.",
            )

    similarity = fuzz.token_sort_ratio(text_a.lower(), text_b.lower())
    _check(trace, "text_comparison", {"a": text_a[:120], "b": text_b[:120], "similarity": round(similarity, 1)},
           "same" if similarity >= 95 else "different")
    if similarity >= 95:
        return PairDecision("corroborates", 0.85, trace,
                            "Both documents state the same value for this subject, in near identical wording.")

    period_a, period_b = _period_of(a), _period_of(b)
    if period_a.start and period_b.start and not periods_overlap(period_a, period_b):
        _check(trace, "period_comparison",
               {"a": period_a.label, "b": period_b.label}, "disjoint")
        return PairDecision(
            "reconciled_by_period", 0.8, trace,
            f"The two statements describe different periods, {period_a.label} against {period_b.label}. A change "
            f"over time supersedes the earlier statement rather than contradicting it.",
        )

    shifted, reasons = scope_shift(a.get("scope_tags") or [], b.get("scope_tags") or [])
    if shifted:
        _check(trace, "scope_comparison", {"a": a.get("scope_tags"), "b": b.get("scope_tags")}, "scope_shift",
               "; ".join(reasons))
        return PairDecision("reconciled_by_scope", 0.75, trace,
                            f"The statements are qualified differently ({'; '.join(reasons)}).")

    _check(trace, "agreement", {"similarity": round(similarity, 1)}, "unsettled",
           "differing text with no period or scope explanation, so the adjudicator decides")
    return PairDecision(None, 0.5, trace, "")


def decide_pairs(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the rule engine over candidate pairs.

    Returns the links the rules settled and the residue for adjudication.
    Rule decided ``unrelated`` pairs are counted but not stored: they are the
    bulk of the funnel and keeping them would bury the interesting links.
    """
    facts = load_facts([p["fact_a"] for p in pairs] + [p["fact_b"] for p in pairs])
    settled: list[dict[str, Any]] = []
    residue: list[dict[str, Any]] = []
    for pair in pairs:
        a, b = facts.get(pair["fact_a"]), facts.get(pair["fact_b"])
        if a is None or b is None:
            continue
        decision = evaluate_pair(a, b)
        record = {
            "link_id": make_link_id(a["fact_id"], b["fact_id"]),
            "fact_a": a["fact_id"],
            "fact_b": b["fact_id"],
            "relation": decision.relation,
            "confidence": decision.confidence,
            "decided_by": "rule",
            "rule_trace": decision.trace,
            "explanation": decision.explanation,
            "numeric_delta": decision.numeric_delta,
            "relative_delta": decision.relative_delta,
            "similarity": pair.get("similarity"),
        }
        if decision.needs_llm:
            residue.append(record)
        elif decision.relation != "unrelated":
            settled.append(record)
        else:
            settled.append({**record, "_skip_store": True})
    return settled, residue
