"""Extraction tests for the deterministic half: the pre filter, the evidence
gate, and fact assembly. No test in this file touches the network."""

from __future__ import annotations

import pytest

from app.chunking import Chunk
from app.extract import build_fact, is_boilerplate, make_fact_id, verify_quote
from app.models import RawFact, RawQualifier

CHUNK_TEXT = (
    "Performance highlights\n\n"
    "FY24 revenue from services was Rs 8,142 Cr, up 12.7 per cent year on year.\n"
    "PTL freight tonnage in FY24 stood at 1.4 Mn Tons.\n"
)


def _chunk(text: str = CHUNK_TEXT, char_start: int = 1000) -> Chunk:
    return Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        page_index_start=4,
        page_index_end=4,
        text=text,
        char_start=char_start,
        char_end=char_start + len(text),
        kind="prose",
    )


PAGE_OFFSETS = [(0, 999, 3, "3"), (1000, 1000 + len(CHUNK_TEXT), 4, "5")]


# ---------------------------------------------------------------------------
# Pre filter
# ---------------------------------------------------------------------------


def test_short_chunk_is_skipped():
    assert is_boilerplate("Revenue rose.")[0] is True


def test_table_of_contents_is_skipped():
    toc = "\n".join(f"Chapter {i} ...................... {i * 7}" for i in range(1, 15))
    skip, reason = is_boilerplate(toc)
    assert skip and reason == "table_of_contents"


def test_disclaimer_without_numbers_is_skipped():
    text = (
        "Safe harbour and disclaimer. This presentation contains forward-looking statements "
        "which involve risks and uncertainties. No representation or warranty is made as to "
        "the accuracy or completeness of the information contained herein. " * 2
    )
    skip, reason = is_boilerplate(text)
    assert skip and reason == "boilerplate_notice"


def test_a_real_paragraph_is_not_skipped():
    assert is_boilerplate(CHUNK_TEXT)[0] is False


def test_a_dense_table_is_not_skipped_for_having_few_letters():
    table = "\n".join("| 2023-24 | 8,142 | 1,429 | 12.7 |" for _ in range(12))
    assert is_boilerplate(table, kind="table")[0] is False


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------


def test_exact_quote_matches_and_offsets_are_right():
    match = verify_quote("Rs 8,142 Cr", CHUNK_TEXT)
    assert match.method == "exact" and match.score == 100.0
    assert CHUNK_TEXT[match.start : match.end] == "Rs 8,142 Cr"


def test_whitespace_differences_still_match_exactly():
    match = verify_quote("FY24 revenue    from services\n was Rs 8,142 Cr", CHUNK_TEXT)
    assert match.method == "normalized"
    assert "8,142" in CHUNK_TEXT[match.start : match.end]


def test_case_differences_still_match():
    assert verify_quote("ptl freight tonnage in fy24", CHUNK_TEXT).found


def test_small_paraphrase_matches_fuzzily_above_threshold():
    match = verify_quote("PTL freight tonnage in FY24 stood at 1.42 Mn Tons", CHUNK_TEXT)
    assert match.found and match.method in {"fuzzy", "normalized"}
    assert match.score >= 90


def test_invented_quote_is_rejected():
    match = verify_quote("The company operates a fleet of submarines in Antarctica.", CHUNK_TEXT)
    assert not match.found and match.method == "fuzzy_failed"


def test_a_quote_that_is_too_short_is_rejected():
    assert verify_quote("Cr", CHUNK_TEXT).method == "too_short"


# ---------------------------------------------------------------------------
# Fact assembly
# ---------------------------------------------------------------------------


def _raw(**overrides) -> RawFact:
    base = dict(
        subject="Delhivery Limited",
        predicate="Revenue from services",
        value_raw="Rs 8,142 Cr",
        value_type="numeric",
        unit_raw="",
        period_label_raw="FY24",
        scope_tags=["Reported"],
        evidence_quote="FY24 revenue from services was Rs 8,142 Cr",
        confidence=0.9,
    )
    base.update(overrides)
    return RawFact(**base)


def test_verified_fact_carries_resolved_page_and_absolute_offsets():
    fact, evidence = build_fact(_raw(), _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert evidence["verified"] is True
    assert evidence["page_index"] == 4 and evidence["page_label"] == "5"
    assert evidence["char_start"] >= 1000
    assert fact["evidence_id"] == evidence["evidence_id"]


def test_a_fact_with_an_invented_quote_is_marked_unverified():
    raw = _raw(evidence_quote="Delhivery acquired a fleet of submarines during FY24.")
    fact, evidence = build_fact(raw, _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert evidence["verified"] is False
    assert evidence["char_start"] is None
    assert fact["fact_id"], "the fact is still stored so it can be quarantined"


def test_normalization_is_applied_during_assembly():
    fact, _ = build_fact(_raw(), _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert fact["value_num"] == pytest.approx(8.142e10)
    assert fact["currency"] == "INR" and fact["unit_dimension"] == "currency"
    assert fact["magnitude_label"] == "cr"
    assert fact["subject_canonical"] == "delhivery"
    assert fact["predicate_canonical"] == "revenue_from_services"
    assert fact["scope_tags"] == ["reported"]
    assert str(fact["period_start"]) == "2023-04-01" and str(fact["period_end"]) == "2024-03-31"
    assert fact["period_basis"] == "fiscal_in"


def test_extraction_note_and_qualifiers_are_preserved():
    raw = _raw(
        extraction_note="read from a chart data label",
        qualifiers=[RawQualifier(key="footnote", value="excludes traded goods")],
    )
    fact, _ = build_fact(raw, _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert fact["qualifiers"]["extraction_note"] == "read from a chart data label"
    assert fact["qualifiers"]["footnote"] == "excludes traded goods"


def test_fact_id_is_stable_and_content_addressed():
    first = make_fact_id("doc1", "chunk1", _raw())
    assert first == make_fact_id("doc1", "chunk1", _raw())
    assert first != make_fact_id("doc1", "chunk1", _raw(value_raw="Rs 7,225 Cr"))
    assert first != make_fact_id("doc2", "chunk1", _raw())
    # A different quote for the same claim does not create a different fact.
    assert first == make_fact_id("doc1", "chunk1", _raw(evidence_quote="Rs 8,142 Cr"))


def test_the_documents_own_period_label_wins_over_the_models_guess():
    raw = _raw(period_label_raw="Q4 FY24", period_start="2023-04-01", period_end="2024-03-31")
    fact, _ = build_fact(raw, _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert str(fact["period_start"]) == "2024-01-01"


def test_the_models_guess_is_used_only_when_the_label_cannot_be_parsed():
    raw = _raw(period_label_raw="", period_start="2024-01-01", period_end="2024-03-31", period_basis="calendar")
    fact, _ = build_fact(raw, _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert str(fact["period_start"]) == "2024-01-01"
    assert fact["period_basis"] == "calendar"


def test_a_semantic_fact_keeps_its_text_and_has_no_number():
    raw = _raw(
        predicate="chief executive officer",
        value_raw="Sahil Barua",
        value_type="string",
        evidence_quote="Performance highlights",
    )
    fact, _ = build_fact(raw, _chunk(), None, PAGE_OFFSETS, doc_id="doc1")
    assert fact["value_num"] is None and fact["value_text"] == "Sahil Barua"


# ---------------------------------------------------------------------------
# The gate must ignore markup the renderer added, without going blind
# ---------------------------------------------------------------------------

MARKED_UP = (
    "## **Performance highlights**\n\n"
    "**FY24 revenue from services** was **Rs 8,142 Cr**, up _12.7 per cent_ year on year.\n"
    "| Item | FY23 | FY24 |\n|---|---|---|\n| **Revenue** | 7,225 | **8,142** |\n"
)


def test_quote_matches_through_markdown_emphasis():
    match = verify_quote("FY24 revenue from services was Rs 8,142 Cr", MARKED_UP)
    assert match.found and match.method == "markup_insensitive"
    assert "8,142" in MARKED_UP[match.start : match.end]


def test_quote_matches_through_table_pipes():
    match = verify_quote("Revenue 7,225 8,142", MARKED_UP)
    assert match.found
    assert "7,225" in MARKED_UP[match.start : match.end]


def test_quote_matches_through_curly_quotes_and_dashes():
    chunk = "Mar ’20 to Mar ’24 shows the trend – tonnage rose."
    assert verify_quote("Mar '20 to Mar '24 shows the trend - tonnage rose.", chunk).found


def test_ignoring_markup_does_not_make_the_gate_accept_inventions():
    assert not verify_quote("Revenue fell to 3,000 Cr in FY24 across Europe", MARKED_UP).found


def test_offsets_stay_exact_after_markup_insensitive_matching():
    match = verify_quote("Performance highlights", MARKED_UP)
    assert MARKED_UP[match.start : match.end] == "Performance highlights"
