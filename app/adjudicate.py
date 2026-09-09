"""Gemini adjudication for the residue the rules cannot settle.

Every pair that reaches this module has already been through the ordered chain
in ``link.py`` and survived it, so the model is only ever asked the questions
that are genuinely questions of judgement. The rule trace is handed to it, both
verbatim quotes are handed to it, and it is told to prefer ``unrelated`` over
forcing a verdict.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import config
from app import db, llm
from app.models import RawAdjudication

ADJUDICATION_INSTRUCTIONS = """\
You are settling whether two extracted facts agree, disagree, or only appear to
disagree. Deterministic rules have already run and could not settle this pair.
Their trace is given below so that you do not repeat work they already did.

Choose exactly one relation:
- corroborates: the two facts assert the same thing, even if worded or
  measured differently.
- contradicts: the two facts assert incompatible things about the same subject,
  the same measure, the same period, and the same basis. Only use this when a
  reader would have to conclude one of the documents is wrong.
- reconciled_by_period: the apparent conflict disappears once the periods are
  taken into account.
- reconciled_by_scope: the apparent conflict disappears once the basis of
  preparation is taken into account, for example adjusted against reported,
  standalone against consolidated, or one segment against a total.
- reconciled_by_unit: the apparent conflict disappears once the units or
  magnitudes are converted.
- reconciled_by_vintage: both documents report the same measure for the same
  period but at different points in time, so the later figure is a revision.
- refines: one fact is a component or sub period of the other.
- unrelated: the two facts are not about the same thing, or there is not
  enough information to say. Prefer this over forcing a verdict.

Rules for your answer:
- Your explanation must be two to four sentences and must refer to the
  evidence quotes below. Quote the specific words that decide it.
- If you spot a contextual explanation the deterministic rules missed, say so
  and choose the matching reconciled_by_* relation.
- If one document states an exchange rate, a restatement, or a definition that
  resolves the difference, cite that stated text in your explanation.
- Do not invent facts that are not in the quotes or the records.

Fact A
  document: {doc_a_title} ({doc_a_filename}), as of {doc_a_as_of}
  page: PDF index {page_a}, printed label {label_a}
  record: {record_a}
  evidence quote: "{quote_a}"

Fact B
  document: {doc_b_title} ({doc_b_filename}), as of {doc_b_as_of}
  page: PDF index {page_b}, printed label {label_b}
  record: {record_b}
  evidence quote: "{quote_b}"

Deterministic checks already run, in order:
{rule_trace}
"""

_RECORD_FIELDS = (
    "subject_canonical",
    "predicate_canonical",
    "value_raw",
    "value_num",
    "value_text",
    "unit_canonical",
    "unit_dimension",
    "magnitude_label",
    "currency",
    "period_label_raw",
    "period_start",
    "period_end",
    "period_basis",
    "scope_tags",
    "qualifiers",
    "confidence",
)


def _record(fact: dict[str, Any]) -> str:
    payload = {key: fact.get(key) for key in _RECORD_FIELDS if fact.get(key) not in (None, "", [], {})}
    return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)


def _render_trace(trace: Sequence[dict[str, Any]]) -> str:
    if not trace:
        return "(no checks recorded)"
    lines = []
    for index, check in enumerate(trace, start=1):
        detail = f" ({check['detail']})" if check.get("detail") else ""
        lines.append(f"{index}. {check.get('name')}: {check.get('outcome')}{detail}")
        lines.append(f"   inputs: {json.dumps(check.get('inputs', {}), default=str, ensure_ascii=False)}")
    return "\n".join(lines)


def build_prompt(a: dict[str, Any], b: dict[str, Any], trace: Sequence[dict[str, Any]]) -> str:
    return ADJUDICATION_INSTRUCTIONS.format(
        doc_a_title=a.get("document_title") or a.get("filename") or "unknown",
        doc_a_filename=a.get("filename") or "",
        doc_a_as_of=a.get("as_of_date") or "unstated",
        page_a=a.get("page_index"),
        label_a=a.get("page_label") or "none",
        record_a=_record(a),
        quote_a=(a.get("quote") or "").strip(),
        doc_b_title=b.get("document_title") or b.get("filename") or "unknown",
        doc_b_filename=b.get("filename") or "",
        doc_b_as_of=b.get("as_of_date") or "unstated",
        page_b=b.get("page_index"),
        label_b=b.get("page_label") or "none",
        record_b=_record(b),
        quote_b=(b.get("quote") or "").strip(),
        rule_trace=_render_trace(trace),
    )


def adjudicate_pair(
    a: dict[str, Any], b: dict[str, Any], record: dict[str, Any], *, recorder: Any | None = None
) -> dict[str, Any] | None:
    """Ask the model about one pair. Returns a link row, or None on failure."""
    trace = list(record.get("rule_trace") or [])
    result = llm.generate_structured(
        stage="adjudicate",
        prompt=build_prompt(a, b, trace),
        schema=RawAdjudication,
        model=config.ADJUDICATION_MODEL,
        prompt_version=config.ADJUDICATION_PROMPT_VERSION,
        cache_content_hash=llm.content_hash(record["fact_a"], record["fact_b"]),
        thinking=config.ADJUDICATION_THINKING,
        recorder=recorder,
    )
    if not result.ok:
        return None
    verdict = result.parsed
    trace.append(
        {
            "name": "llm_adjudication",
            "inputs": {"model": config.ADJUDICATION_MODEL, "prompt_version": config.ADJUDICATION_PROMPT_VERSION},
            "outcome": verdict.relation,
            "detail": "sent to the adjudicator because no deterministic rule settled the pair",
        }
    )
    return {
        **record,
        "relation": verdict.relation,
        "confidence": float(verdict.confidence),
        "decided_by": "llm",
        "rule_trace": trace,
        "explanation": verdict.explanation.strip(),
    }


def adjudicate(residue: list[dict[str, Any]], *, recorder: Any | None = None) -> list[dict[str, Any]]:
    """Adjudicate the residue, most similar pairs first, within the call budget."""
    if not residue:
        return []
    from app.link import load_facts

    ordered = sorted(residue, key=lambda r: -(r.get("similarity") or 0.0))[: config.ADJUDICATION_MAX_PAIRS]
    if len(residue) > config.ADJUDICATION_MAX_PAIRS and recorder is not None:
        recorder.add_error(
            "adjudication_budget",
            f"{len(residue) - config.ADJUDICATION_MAX_PAIRS} residual pairs were left undecided by the per run cap",
        )
    facts = load_facts([r["fact_a"] for r in ordered] + [r["fact_b"] for r in ordered])
    out: list[dict[str, Any]] = []
    for record in ordered:
        a, b = facts.get(record["fact_a"]), facts.get(record["fact_b"])
        if a is None or b is None:
            continue
        decided = adjudicate_pair(a, b, record, recorder=recorder)
        if decided is not None:
            out.append(decided)
    return out
