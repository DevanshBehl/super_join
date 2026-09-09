#!/usr/bin/env python3
"""Find and print the four required cases by querying the database.

Nothing here is hardcoded to the starter PDFs. Each case is selected by a rule
over whatever the system actually found, so the script produces a different but
equally valid answer on a different corpus. When a case genuinely is not
present, the script says so instead of inventing one.

Usage:
    python scripts/demo_cases.py                  all four cases
    python scripts/demo_cases.py --case 2         one case
    python scripts/demo_cases.py --collection X   restrict to one collection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402

RECONCILED = ("reconciled_by_period", "reconciled_by_scope", "reconciled_by_unit",
              "reconciled_by_vintage", "refines")

RULE = "-" * 78


def _collection_clause(collection: str | None, alias: str = "da") -> tuple[str, list[Any]]:
    if not collection:
        return "", []
    return f" AND {alias}.collection = ?", [collection]


def _link_query(where: str, params: list[Any], order: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = db.query(
        f"""
        SELECT l.*, da.collection
        FROM links l
        JOIN facts fa ON fa.fact_id = l.fact_a
        JOIN facts fb ON fb.fact_id = l.fact_b
        JOIN documents da ON da.doc_id = fa.doc_id
        JOIN documents dbb ON dbb.doc_id = fb.doc_id
        WHERE {where}
        ORDER BY {order}
        LIMIT {int(limit)}
        """,
        params,
    )
    for row in rows:
        row["rule_trace"] = db.loads(row.get("rule_trace"), [])
    return rows


def _print_fact(fact: dict[str, Any] | None, side: str) -> None:
    if fact is None:
        print(f"  Fact {side}: missing")
        return
    normalized = (
        f"{fact['value_num']:,.6g} {fact.get('unit_canonical') or ''}".strip()
        if fact.get("value_num") is not None
        else (fact.get("value_text") or "")
    )
    print(f"  Fact {side}")
    print(f"    subject      : {fact.get('subject_raw')}  [canonical: {fact.get('subject_canonical')}]")
    print(f"    predicate    : {fact.get('predicate_canonical')}")
    print(f"    raw value    : {fact.get('value_raw')}")
    print(f"    normalized   : {normalized}"
          + (f"   (magnitude {fact.get('magnitude_label')})" if fact.get("magnitude_label") else ""))
    print(f"    period       : {fact.get('period_label_raw') or 'unstated'} "
          f"[{fact.get('period_start')} to {fact.get('period_end')}, basis {fact.get('period_basis')}]")
    print(f"    scope        : {', '.join(fact.get('scope_tags') or []) or 'none'}")
    print(f"    source       : {fact.get('filename')}, PDF page index {fact.get('page_index')}"
          + (f", printed page {fact.get('page_label')}" if fact.get("page_label") else ""))
    print(f"    evidence     : \"{(fact.get('quote') or '').strip()}\"")
    print(f"    offsets      : {fact.get('evidence_char_start')} to {fact.get('evidence_char_end')} "
          f"({fact.get('match_method')} match, score {fact.get('match_score')})")


def _print_link(link: dict[str, Any], heading: str, why_selected: str) -> None:
    facts = db.facts_by_ids([link["fact_a"], link["fact_b"]])
    print(RULE)
    print(heading)
    print(RULE)
    print(f"  selected because : {why_selected}")
    print(f"  relation         : {link['relation']}  (confidence {link['confidence']:.2f}, "
          f"decided by {link['decided_by']})")
    if link.get("relative_delta") is not None:
        print(f"  numeric gap      : absolute {link.get('numeric_delta'):,.6g}, "
              f"relative {link['relative_delta'] * 100:.3f} per cent")
    print(f"  link id          : {link['link_id']}")
    print()
    _print_fact(facts.get(link["fact_a"]), "A")
    print()
    _print_fact(facts.get(link["fact_b"]), "B")
    print()
    print("  Rule trace")
    for index, check in enumerate(link["rule_trace"], start=1):
        detail = f"  ({check.get('detail')})" if check.get("detail") else ""
        print(f"    {index}. {check.get('name')}: {check.get('outcome')}{detail}")
        print(f"       inputs: {check.get('inputs')}")
    print()
    print("  Explanation")
    for line in _wrap(link.get("explanation") or "(none recorded)"):
        print(f"    {line}")
    print()
    print(f"  Reproduce: curl -s localhost:8000/api/links/{link['link_id']} | python -m json.tool")
    print()


def _wrap(text: str, width: int = 74) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


# ---------------------------------------------------------------------------
# Case 1: corroboration across documents, expressed differently
# ---------------------------------------------------------------------------


def case_corroboration(collection: str | None) -> None:
    extra, params = _collection_clause(collection)
    rows = _link_query(
        "l.relation = 'corroborates' AND fa.doc_id <> fb.doc_id"
        " AND lower(fa.value_raw) <> lower(fb.value_raw)" + extra,
        params,
        # Prefer pairs whose raw wording differs most, then the confident ones.
        "CASE WHEN COALESCE(fa.magnitude_label,'') <> COALESCE(fb.magnitude_label,'') THEN 0 ELSE 1 END,"
        " CASE WHEN COALESCE(fa.unit_canonical,'') <> COALESCE(fb.unit_canonical,'') THEN 0 ELSE 1 END,"
        " l.confidence DESC, l.relative_delta DESC",
        limit=1,
    )
    if not rows:
        print(RULE)
        print("CASE 1: a fact corroborated across documents though expressed differently")
        print(RULE)
        print("  The system found no cross document corroboration in this corpus.")
        print("  That is a real result, not a placeholder: with a single document, or with")
        print("  documents that share no measure, this case cannot exist.\n")
        return
    _print_link(
        rows[0],
        "CASE 1: a fact corroborated across documents though expressed differently",
        "highest confidence corroboration whose two sides come from different documents "
        "and are written with different wording, units, or magnitudes",
    )


# ---------------------------------------------------------------------------
# Case 2: a genuine or likely contradiction
# ---------------------------------------------------------------------------


def case_contradiction(collection: str | None) -> None:
    extra, params = _collection_clause(collection)
    rows = _link_query("l.relation = 'contradicts'" + extra, params,
                       "l.confidence DESC, COALESCE(l.relative_delta, 0) DESC", limit=1)
    if rows:
        _print_link(
            rows[0],
            "CASE 2: a genuine or likely contradiction",
            "the highest confidence link the adjudicator marked as a contradiction after the "
            "deterministic rules failed to reconcile it",
        )
        return

    print(RULE)
    print("CASE 2: a genuine or likely contradiction")
    print(RULE)
    print("  No link was classified as a contradiction. The nearest unresolved pairs, which")
    print("  the rules escalated but the adjudicator explained away, are listed below.\n")
    near = _link_query(
        "l.decided_by = 'llm' AND l.relation <> 'contradicts'" + extra, params,
        "COALESCE(l.relative_delta, 0) DESC", limit=3,
    )
    for row in near:
        facts = db.facts_by_ids([row["fact_a"], row["fact_b"]])
        a, b = facts.get(row["fact_a"]), facts.get(row["fact_b"])
        if not (a and b):
            continue
        print(f"  {a.get('predicate_canonical')}: {a.get('value_raw')} ({a.get('filename')}) against "
              f"{b.get('value_raw')} ({b.get('filename')}) -> {row['relation']}")
    print()


# ---------------------------------------------------------------------------
# Case 3: an apparent contradiction explained by context
# ---------------------------------------------------------------------------


def case_reconciled(collection: str | None) -> None:
    extra, params = _collection_clause(collection)
    placeholders = ", ".join(f"'{r}'" for r in RECONCILED)
    rows = _link_query(
        f"l.relation IN ({placeholders}) AND COALESCE(l.relative_delta, 0) > 0" + extra,
        params,
        # The most striking reconciliation is the one with the widest apparent gap.
        "COALESCE(l.relative_delta, 0) DESC, l.confidence DESC",
        limit=1,
    )
    if not rows:
        print(RULE)
        print("CASE 3: an apparent contradiction explained by context")
        print(RULE)
        print("  No reconciled link with a numeric gap was found in this corpus.\n")
        return
    relation = rows[0]["relation"]
    reason = {
        "reconciled_by_period": "the periods differ",
        "reconciled_by_scope": "the basis of preparation differs",
        "reconciled_by_unit": "the units or magnitudes differ",
        "reconciled_by_vintage": "the documents speak as of different dates",
        "refines": "one figure is a component of the other",
    }[relation]
    _print_link(
        rows[0],
        "CASE 3: an apparent contradiction explained by context",
        f"the widest numeric gap that a rule still explained away, here because {reason}",
    )


# ---------------------------------------------------------------------------
# Case 4: an extraction or reasoning failure
# ---------------------------------------------------------------------------


def case_failure(collection: str | None) -> None:
    print(RULE)
    print("CASE 4: an extraction or reasoning failure, how it was caught, and the fix")
    print(RULE)

    quarantine, total = db.quarantined_facts(limit=3)
    print(f"  4a. Evidence verification rejected {total} extracted facts.")
    if quarantine:
        print("      The closest misses are shown: the model produced a quote that does not")
        print("      appear in the chunk it was reading, so the fact never entered the")
        print("      active set. Caught by the evidence gate, automatically, at ingest time.\n")
        for fact in quarantine:
            print(f"      subject   : {fact.get('subject_raw')} / {fact.get('predicate_canonical')}")
            print(f"      value     : {fact.get('value_raw')}")
            print(f"      quote     : \"{(fact.get('quote') or '').strip()[:160]}\"")
            print(f"      source    : {fact.get('filename')}, PDF page index {fact.get('page_index')}")
            print(f"      best score: {fact.get('match_score'):.1f} by {fact.get('match_method')}, "
                  f"threshold is the configured fuzzy floor")
            print()
    else:
        print("      Nothing is currently quarantined.\n")

    warnings = db.query(
        "SELECT doc_id, stage, errors FROM runs WHERE stage = 'parse' AND errors <> '[]'"
    )
    gaps: list[tuple[str, Any]] = []
    for row in warnings:
        document = db.get_document(row["doc_id"]) or {}
        for error in db.loads(row["errors"], []):
            if error.get("kind") in {"image_only_page", "empty_page"}:
                gaps.append((document.get("filename", row["doc_id"]), error.get("page_index")))
    print(f"  4b. Parsing recorded {len(gaps)} pages that yielded no extractable text.")
    if gaps:
        print("      These are image only or blank pages. There is no OCR in the pipeline, so")
        print("      any fact printed only on those pages is invisible to the system. The gap")
        print("      is recorded rather than hidden, which is why it can be quoted here.\n")
        for filename, page_index in gaps[:6]:
            print(f"      {filename}, PDF page index {page_index}")
        print()

    close_calls = []
    for row in db.query("SELECT notes FROM runs WHERE stage = 'entities'"):
        close_calls.extend(db.loads(row["notes"], {}).get("close_calls", []))
    print(f"  4c. Entity resolution logged {len(close_calls)} borderline merge decisions.")
    if close_calls:
        print("      Each of these sat within a few points of the merge threshold, so the")
        print("      decision could have gone either way. They are recorded so a reviewer can")
        print("      check them rather than discovering a silent merge later.\n")
        for call in close_calls[:6]:
            print(f"      {call.get('subject')!r} against {call.get('candidate')!r} "
                  f"at {call.get('score')} -> {call.get('decision')}")
        print()

    _print_improvements(quarantine, total, gaps, close_calls)

    print("  Reproduce: curl -s localhost:8000/api/quarantine | python -m json.tool")
    print()




def _classify_failure(fact: dict[str, Any]) -> str:
    """Group a quarantined fact by what actually went wrong with its quote.

    The grouping is derived from the stored quote and its chunk, not asserted,
    so the diagnosis holds on any corpus.
    """
    quote = fact.get("quote") or ""
    if quote.count("\n") >= 1:
        return "multi_line_stitch"
    if "|" in quote:
        return "table_row"
    if len(quote.strip()) < 40:
        return "very_short_quote"
    return "other"


def _print_improvements(
    quarantine: list[dict[str, Any]],
    total: int,
    gaps: list[tuple[str, Any]],
    close_calls: list[dict[str, Any]],
) -> None:
    """Describe only the failure modes this corpus actually exhibits."""
    all_failed, _ = db.quarantined_facts(limit=1000)
    counts: dict[str, int] = {}
    for fact in all_failed:
        kind = _classify_failure(fact)
        counts[kind] = counts.get(kind, 0) + 1

    print("  Observed failure modes, counted from the quarantine table")
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {kind:20s} {count:4d} of {total}")
    print()

    print("  How these would be improved")
    remedies = {
        "multi_line_stitch": (
            "The model joined text from separate lines into one quote, which by "
            "definition cannot be a contiguous span. Asking for a line anchor plus a "
            "cell reference for chart and table figures, instead of a prose quote, "
            "would let these be verified without loosening the gate."
        ),
        "table_row": (
            "Quotes taken from a rendered pipe table carry markup the printed page "
            "does not have. The gate already matches through that markup; the residue "
            "wants a cell addressed view of the table so a fact can cite a cell."
        ),
        "very_short_quote": (
            "A quote too short to be distinctive matches nothing safely. Requiring the "
            "quote to carry both the subject and the value would remove this class."
        ),
        "other": (
            "The remainder are paraphrases. A second cheap pass that asks the model to "
            "copy the exact span, given its own earlier answer, would recover some."
        ),
    }
    for kind, _count in sorted(counts.items(), key=lambda kv: -kv[1]):
        for line in _wrap(remedies.get(kind, "")):
            print(f"    {line}")
        print()
    if gaps:
        for line in _wrap(
            f"Separately, {len(gaps)} pages yielded no extractable text. An OCR fallback "
            f"invoked only for pages under the sparse text threshold would close that gap "
            f"while leaving the rest of the pipeline CPU only."
        ):
            print(f"    {line}")
        print()
    if close_calls:
        for line in _wrap(
            f"{len(close_calls)} entity decisions sat within a few points of the merge "
            f"threshold. A blocking key stronger than a name, such as an identifier "
            f"printed in the document, would settle these before fuzzy matching fires."
        ):
            print(f"    {line}")
        print()


CASES = {1: case_corroboration, 2: case_contradiction, 3: case_reconciled, 4: case_failure}


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the four required cases from the database.")
    parser.add_argument("--case", type=int, choices=sorted(CASES), default=None)
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    if not db.query("SELECT 1 FROM documents LIMIT 1"):
        print("The database is empty. Run python scripts/ingest_all.py first.")
        return 1

    stats = db.stats()
    print("=" * 78)
    print("FACT KNOWLEDGE LAYER: the four required cases, selected by query")
    print("=" * 78)
    print(f"corpus: {stats['n_documents']} documents, {stats['n_verified_facts']} verified facts, "
          f"{stats['n_links']} links")
    print(f"links by relation: {stats['links_by_relation']}")
    if args.collection:
        print(f"restricted to collection: {args.collection}")
    print()

    for number in ([args.case] if args.case else sorted(CASES)):
        CASES[number](args.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
