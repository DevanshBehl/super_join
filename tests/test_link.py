"""Rule engine and candidate funnel tests. Deterministic, no network."""

from __future__ import annotations

import importlib

import pytest

from app.link import evaluate_pair, make_link_id


def fact(**overrides):
    base = {
        "fact_id": "a" * 32,
        "doc_id": "doc1",
        "chunk_id": "chunk1",
        "entity_id": "ent1",
        "subject_canonical": "delhivery",
        "predicate_canonical": "revenue_from_services",
        "value_type": "numeric",
        "value_raw": "8,142",
        "value_num": 8142.0,
        "value_text": None,
        "unit_canonical": "INR",
        "unit_dimension": "currency",
        "currency": "INR",
        "period_start": "2023-04-01",
        "period_end": "2024-03-31",
        "period_basis": "fiscal_in",
        "period_label_raw": "FY24",
        "scope_tags": ["reported"],
        "qualifiers": {},
        "filename": "doc-a.pdf",
        "as_of_date": "2024-03-31",
    }
    base.update(overrides)
    return base


def names(decision):
    return [check["name"] for check in decision.trace]


# ---------------------------------------------------------------------------
# Rule 1 and 2: comparability and currency
# ---------------------------------------------------------------------------


def test_different_value_types_are_unrelated():
    decision = evaluate_pair(fact(), fact(fact_id="b" * 32, value_type="string", value_text="Sahil Barua"))
    assert decision.relation == "unrelated"
    assert names(decision)[0] == "value_type"


def test_different_dimensions_are_unrelated():
    a = fact(unit_canonical="tonne", unit_dimension="mass", currency=None)
    b = fact(fact_id="b" * 32, unit_canonical="day", unit_dimension="duration", currency=None)
    decision = evaluate_pair(a, b)
    assert decision.relation == "unrelated"
    assert "dimension" in decision.explanation


def test_different_currencies_without_a_stated_rate_are_unrelated():
    a = fact()
    b = fact(fact_id="b" * 32, currency="USD", unit_canonical="USD", value_raw="$980 mn", value_num=980e6)
    decision = evaluate_pair(a, b)
    assert decision.relation == "unrelated"
    assert "exchange rate" in decision.explanation


def test_different_currencies_with_a_stated_rate_go_to_the_adjudicator():
    a = fact(qualifiers={"exchange_rate": "83.2 INR per USD"})
    b = fact(fact_id="b" * 32, currency="USD", unit_canonical="USD", value_num=980e6, value_raw="$980 mn")
    decision = evaluate_pair(a, b)
    assert decision.needs_llm, "a stated rate is a judgement call, not a rule"


# ---------------------------------------------------------------------------
# Rule 3: unit reconciliation
# ---------------------------------------------------------------------------


def test_the_same_tonnage_in_two_units_is_reconciled_by_unit():
    a = fact(value_raw="1.4 Mn Tons", value_num=1_400_000, unit_canonical="tonne",
             unit_dimension="mass", currency=None)
    b = fact(fact_id="b" * 32, value_raw="1,429", value_num=1.429e9, unit_canonical="kg",
             unit_dimension="mass", currency=None)
    decision = evaluate_pair(a, b)
    assert decision.relation == "reconciled_by_unit"
    conversion = [c for c in decision.trace if c["name"] == "unit_conversion"][0]
    assert conversion["outcome"] == "converted"
    assert "kg" in conversion["detail"] and "tonne" in conversion["detail"]


def test_basis_points_reconcile_against_percent():
    a = fact(value_raw="0.25%", value_num=0.25, unit_canonical="percent", unit_dimension="ratio", currency=None)
    b = fact(fact_id="b" * 32, value_raw="25 bps", value_num=25.0, unit_canonical="bps",
             unit_dimension="ratio", currency=None)
    assert evaluate_pair(a, b).relation == "reconciled_by_unit"


# ---------------------------------------------------------------------------
# Rule 4: periods
# ---------------------------------------------------------------------------


def test_a_quarter_inside_a_year_refines_rather_than_contradicts():
    year = fact(value_raw="8,142", value_num=8142.0)
    quarter = fact(fact_id="b" * 32, value_raw="2,076", value_num=2076.0,
                   period_start="2024-01-01", period_end="2024-03-31", period_label_raw="Q4 FY24")
    decision = evaluate_pair(year, quarter)
    assert decision.relation == "refines"
    assert "inside" in decision.explanation
    assert [c for c in decision.trace if c["name"] == "period_comparison"][0]["outcome"] == "nested"


def test_disjoint_periods_are_reconciled_by_period():
    fy24 = fact()
    fy23 = fact(fact_id="b" * 32, value_raw="7,225", value_num=7225.0,
                period_start="2022-04-01", period_end="2023-03-31", period_label_raw="FY23")
    decision = evaluate_pair(fy24, fy23)
    assert decision.relation == "reconciled_by_period"


# ---------------------------------------------------------------------------
# Rule 5: scope
# ---------------------------------------------------------------------------


def test_adjusted_against_reported_is_reconciled_by_scope():
    reported = fact(predicate_canonical="ebitda", value_raw="127", value_num=127.0, scope_tags=["reported"])
    adjusted = fact(fact_id="b" * 32, predicate_canonical="ebitda", value_raw="76", value_num=76.0,
                    scope_tags=["adjusted"])
    decision = evaluate_pair(reported, adjusted)
    assert decision.relation == "reconciled_by_scope"
    assert "adjusted" in decision.explanation


def test_standalone_against_consolidated_is_reconciled_by_scope():
    a = fact(scope_tags=["standalone"], value_num=7000.0, value_raw="7,000")
    b = fact(fact_id="b" * 32, scope_tags=["consolidated"], value_num=8142.0, value_raw="8,142")
    assert evaluate_pair(a, b).relation == "reconciled_by_scope"


def test_a_scope_difference_does_not_override_agreement():
    a = fact(scope_tags=["adjusted"])
    b = fact(fact_id="b" * 32, scope_tags=["reported"])
    assert evaluate_pair(a, b).relation == "corroborates"


# ---------------------------------------------------------------------------
# Rule 6: vintage
# ---------------------------------------------------------------------------


def test_the_same_claim_from_two_vintages_is_reconciled_by_vintage():
    survey = fact(predicate_canonical="real_gdp_growth", value_raw="6.4", value_num=6.4,
                  unit_canonical="percent", unit_dimension="ratio", currency=None,
                  as_of_date="2025-01-31", filename="survey.pdf", scope_tags=["estimate"])
    rbi = fact(fact_id="b" * 32, predicate_canonical="real_gdp_growth", value_raw="6.5", value_num=6.5,
               unit_canonical="percent", unit_dimension="ratio", currency=None,
               as_of_date="2025-05-29", filename="rbi.pdf", scope_tags=["estimate"])
    decision = evaluate_pair(survey, rbi)
    assert decision.relation == "reconciled_by_vintage"
    assert "rbi.pdf" in decision.explanation and "survey.pdf" in decision.explanation


# ---------------------------------------------------------------------------
# Rule 7: numeric agreement and rounding
# ---------------------------------------------------------------------------


def test_values_agreeing_within_claimed_precision_corroborate():
    a = fact(value_raw="1.4 Mn Tons", value_num=1_400_000, unit_canonical="tonne",
             unit_dimension="mass", currency=None)
    b = fact(fact_id="b" * 32, value_raw="1,429", value_num=1_429_000, unit_canonical="tonne",
             unit_dimension="mass", currency=None)
    decision = evaluate_pair(a, b)
    assert decision.relation == "corroborates"
    assert decision.relative_delta == pytest.approx(0.0203, abs=1e-3)


def test_a_rounding_difference_corroborates_and_says_so():
    a = fact(value_raw="578", value_num=578.0)
    b = fact(fact_id="b" * 32, value_raw="579", value_num=579.0)
    decision = evaluate_pair(a, b)
    assert decision.relation == "corroborates"
    assert "rounding" in decision.explanation


def test_a_material_gap_is_escalated_rather_than_guessed():
    a = fact(value_raw="8,142", value_num=8142.0)
    b = fact(fact_id="b" * 32, value_raw="6,000", value_num=6000.0)
    decision = evaluate_pair(a, b)
    assert decision.needs_llm and decision.relation is None
    assert decision.trace[-1]["outcome"] == "material_gap"


def test_the_trace_records_every_check_in_order():
    decision = evaluate_pair(fact(), fact(fact_id="b" * 32))
    order = names(decision)
    assert order[:3] == ["value_type", "unit_dimension", "currency"]
    assert "numeric_comparison" in order and order[-1] == "agreement"
    for check in decision.trace:
        assert set(check) == {"name", "inputs", "outcome", "detail"}


# ---------------------------------------------------------------------------
# Non numeric facts
# ---------------------------------------------------------------------------


def test_addresses_written_differently_corroborate():
    a = fact(value_type="string", value_num=None, unit_canonical=None, unit_dimension=None, currency=None,
             value_text="Plot 5, MG Road, Bengaluru - 560001", predicate_canonical="registered_office")
    b = fact(fact_id="b" * 32, value_type="string", value_num=None, unit_canonical=None, unit_dimension=None,
             currency=None, value_text="Plot No 5, M G Rd., Bengaluru 560001",
             predicate_canonical="registered_office")
    decision = evaluate_pair(a, b)
    assert decision.relation == "corroborates"
    assert "postal code" in decision.explanation


def test_a_role_change_across_periods_supersedes_rather_than_contradicts():
    old = fact(value_type="string", value_num=None, unit_canonical=None, unit_dimension=None, currency=None,
               value_text="Kapil Bharati", predicate_canonical="chief_technology_officer",
               period_start="2022-01-01", period_end="2022-12-31", period_label_raw="2022")
    new = fact(fact_id="b" * 32, value_type="string", value_num=None, unit_canonical=None,
               unit_dimension=None, currency=None, value_text="Amit Agarwal",
               predicate_canonical="chief_technology_officer", period_start="2024-01-01",
               period_end="2024-12-31", period_label_raw="2024")
    decision = evaluate_pair(old, new)
    assert decision.relation == "reconciled_by_period"


def test_conflicting_text_in_the_same_period_is_escalated():
    a = fact(value_type="string", value_num=None, unit_canonical=None, unit_dimension=None, currency=None,
             value_text="Kapil Bharati", predicate_canonical="chief_technology_officer")
    b = fact(fact_id="b" * 32, value_type="string", value_num=None, unit_canonical=None, unit_dimension=None,
             currency=None, value_text="Amit Agarwal", predicate_canonical="chief_technology_officer")
    assert evaluate_pair(a, b).needs_llm


# ---------------------------------------------------------------------------
# Link identity
# ---------------------------------------------------------------------------


def test_link_id_is_order_independent():
    assert make_link_id("a", "b") == make_link_id("b", "a")
    assert make_link_id("a", "b") != make_link_id("a", "c")


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "link.duckdb"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    import config

    importlib.reload(config)
    from app import db as db_module

    importlib.reload(db_module)
    db_module.close()
    from app import link as link_module

    importlib.reload(link_module)
    return db_module, link_module


def _seed(db, doc_id, collection, facts):
    db.upsert_document({"doc_id": doc_id, "filename": f"{doc_id}.pdf", "collection": collection})
    rows = []
    for fact_id, predicate, entity, dimension, vector, chunk in facts:
        rows.append(
            {
                "fact_id": fact_id, "doc_id": doc_id, "chunk_id": chunk,
                "subject_canonical": "delhivery", "predicate_canonical": predicate,
                "entity_id": entity, "value_type": "numeric", "value_num": 1.0,
                "unit_dimension": dimension, "evidence_id": f"ev-{fact_id}",
            }
        )
        db.insert_evidence([{"evidence_id": f"ev-{fact_id}", "fact_id": fact_id, "doc_id": doc_id,
                             "quote": "q", "verified": True}])
        db.upsert_fact_embeddings([(fact_id, "canon", vector)])
    db.insert_facts(rows)


def test_blocking_keeps_only_comparable_pairs(seeded):
    db, link = seeded
    same = [1.0, 0.0, 0.0, 0.0]
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", same, "ch1")])
    _seed(db, "d2", "c1", [
        ("f2", "revenue", "e1", "currency", same, "ch2"),       # same predicate and dimension
        ("f3", "tonnage", "e9", "mass", same, "ch3"),           # different entity and dimension
    ])
    pairs = link.generate_candidates()
    got = {tuple(sorted([p["fact_a"], p["fact_b"]])) for p in pairs}
    assert ("f1", "f2") in got
    assert ("f1", "f3") not in got


def test_pairs_from_the_same_chunk_are_dropped(seeded):
    db, link = seeded
    same = [1.0, 0.0, 0.0, 0.0]
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", same, "ch1"),
                           ("f2", "revenue", "e1", "currency", same, "ch1")])
    assert link.generate_candidates() == []


def test_collections_do_not_link_across_by_default(seeded):
    db, link = seeded
    same = [1.0, 0.0, 0.0, 0.0]
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", same, "ch1")])
    _seed(db, "d2", "c2", [("f2", "revenue", "e1", "currency", same, "ch2")])
    assert link.generate_candidates() == []


def test_distant_vectors_are_dropped_by_the_cosine_threshold(seeded):
    db, link = seeded
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", [1.0, 0.0, 0.0, 0.0], "ch1")])
    _seed(db, "d2", "c1", [("f2", "revenue", "e1", "currency", [0.0, 1.0, 0.0, 0.0], "ch2")])
    assert link.generate_candidates() == []


def test_a_pair_is_only_ever_produced_once(seeded):
    db, link = seeded
    same = [1.0, 0.0, 0.0, 0.0]
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", same, "ch1")])
    _seed(db, "d2", "c1", [("f2", "revenue", "e1", "currency", same, "ch2")])
    pairs = link.generate_candidates()
    assert len(pairs) == 1
    assert pairs[0]["fact_a"] < pairs[0]["fact_b"], "pairs are ordered deterministically"


def test_incremental_generation_only_touches_the_new_document(seeded):
    db, link = seeded
    same = [1.0, 0.0, 0.0, 0.0]
    _seed(db, "d1", "c1", [("f1", "revenue", "e1", "currency", same, "ch1")])
    _seed(db, "d2", "c1", [("f2", "revenue", "e1", "currency", same, "ch2")])
    db.insert_links([{"link_id": link.make_link_id("f1", "f2"), "fact_a": "f1", "fact_b": "f2",
                      "relation": "corroborates"}])
    _seed(db, "d3", "c1", [("f4", "revenue", "e1", "currency", same, "ch4")])
    pairs = link.generate_candidates(doc_id="d3")
    involved = {p["fact_a"] for p in pairs} | {p["fact_b"] for p in pairs}
    assert "f4" in involved
    assert all("f4" in (p["fact_a"], p["fact_b"]) for p in pairs)


# ---------------------------------------------------------------------------
# A null unit dimension used to act as a wildcard that matched anything. That
# is right when both sides assert the same predicate and one document simply
# omitted the unit, and wrong otherwise: it let a revenue in dollars be
# "corroborated" by a shipment count, because blocking had already paired them
# on the shared entity.
# ---------------------------------------------------------------------------


def test_revenue_does_not_corroborate_a_bare_count():
    revenue = fact(
        predicate_canonical="revenue",
        value_raw="$1 billion", value_num=1_000_000_000.0,
        unit_canonical="USD", unit_dimension="currency", currency="USD",
    )
    shipments = fact(
        fact_id="b" * 32, doc_id="doc2",
        predicate_canonical="count",
        value_raw="740 Mn", value_num=740_000_000.0,
        unit_canonical=None, unit_dimension=None, currency=None,
    )
    decision = evaluate_pair(revenue, shipments)
    assert decision.relation == "unrelated"
    assert "different predicates" in decision.explanation


def test_two_unitless_facts_with_different_predicates_are_unrelated():
    nominee = fact(
        predicate_canonical="non_executive_nominee_directors_count",
        value_raw="3", value_num=3.0,
        unit_canonical=None, unit_dimension=None, currency=None,
    )
    executive = fact(
        fact_id="b" * 32, doc_id="doc2",
        predicate_canonical="executive_directors_count",
        value_raw="three (3)", value_num=3.0,
        unit_canonical=None, unit_dimension=None, currency=None,
    )
    decision = evaluate_pair(nominee, executive)
    assert decision.relation == "unrelated"


def test_a_missing_unit_is_still_a_wildcard_for_the_same_predicate():
    # The behaviour the wildcard was written for must survive: one document
    # states the unit, the other omits it, but both assert the same predicate.
    stated = fact(unit_canonical="INR", unit_dimension="currency", currency="INR")
    omitted = fact(
        fact_id="b" * 32, doc_id="doc2",
        unit_canonical=None, unit_dimension=None, currency=None,
    )
    decision = evaluate_pair(stated, omitted)
    assert decision.relation != "unrelated"


def test_different_predicates_sharing_a_dimension_are_still_compared():
    a = fact(predicate_canonical="revenue_from_services")
    b = fact(fact_id="b" * 32, doc_id="doc2", predicate_canonical="total_income")
    decision = evaluate_pair(a, b)
    assert decision.relation != "unrelated"
