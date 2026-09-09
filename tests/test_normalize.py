"""Normalizer tests. These are the parts of the system that must never be wrong."""

from __future__ import annotations

from datetime import date

import pytest

from app import normalize as N


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("8,142", 8142.0),
        ("1,00,000", 100000.0),          # Indian digit grouping
        ("1,000,000", 1000000.0),        # western digit grouping
        ("8142", 8142.0),
        ("12.75", 12.75),
        ("-3.2", -3.2),
        ("(452)", -452.0),               # parentheses are negative
        ("₹(452) Cr", -452.0),
        ("(6.3%)", -6.3),
        ("0", 0.0),
    ],
)
def test_parse_number_values(raw, expected):
    assert N.parse_number(raw).value == pytest.approx(expected)


def test_parse_number_records_approximation_instead_of_dropping_it():
    assert N.parse_number("~1.4").qualifiers["bound"] == "approximate"
    assert N.parse_number("over 8,000").qualifiers["bound"] == "lower_bound"
    assert N.parse_number("at least 25").qualifiers["bound"] == "lower_bound"
    assert N.parse_number("less than 5").qualifiers["bound"] == "upper_bound"
    assert N.parse_number("approximately 12.5").value == pytest.approx(12.5)


def test_parse_number_range_keeps_both_ends_and_uses_the_midpoint():
    parsed = N.parse_number("6.3 to 6.8")
    assert parsed.qualifiers["range_low"] == pytest.approx(6.3)
    assert parsed.qualifiers["range_high"] == pytest.approx(6.8)
    assert parsed.value == pytest.approx(6.55)


def test_a_fiscal_span_is_not_read_as_a_range():
    parsed = N.parse_number("FY 2023-24")
    assert "range_low" not in parsed.qualifiers


def test_no_number_returns_none():
    assert N.parse_number("not a number at all").value is None
    assert N.parse_number("").value is None


@pytest.mark.parametrize(
    "raw, place", [("1.4", 0.1), ("1,429", 1.0), ("8,142", 1.0), ("0.05", 0.01), ("12", 1.0)]
)
def test_significant_digit_place(raw, place):
    assert N.significant_digit_place(raw) == pytest.approx(place)


# ---------------------------------------------------------------------------
# Magnitude and currency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, factor, label",
    [
        ("8,142 Cr", 1e7, "cr"),
        ("₹15 crore", 1e7, "crore"),
        ("5 lakh", 1e5, "lakh"),
        ("1.4 Mn", 1e6, "mn"),
        ("2.8 Bn+", 1e9, "bn"),
        ("1,429 thousand tonnes", 1e3, "thousand"),
        ("12 trillion", 1e12, "trillion"),
        ("3.5 lakh crore", 1e12, "lakh crore"),
        ("8,142", 1.0, None),
    ],
)
def test_detect_magnitude(text, factor, label):
    got_factor, got_label = N.detect_magnitude(text)
    assert got_factor == pytest.approx(factor)
    assert got_label == label


def test_a_bare_letter_is_not_treated_as_a_magnitude():
    assert N.detect_magnitude("the m and b of it")[0] == 1.0
    assert N.detect_magnitude("740 m")[0] == pytest.approx(1e6)


@pytest.mark.parametrize(
    "text, code",
    [("₹8,142 Cr", "INR"), ("Rs. 500", "INR"), ("INR 12", "INR"), ("US$ 4.2 bn", "USD"),
     ("$300 million", "USD"), ("EUR 10", "EUR"), ("1.4 Mn Tons", None)],
)
def test_detect_currency(text, code):
    assert N.detect_currency(text) == code


def test_currencies_are_never_converted_between_each_other():
    inr = N.canonical_unit("", "₹100 Cr")
    usd = N.canonical_unit("", "$100 mn")
    assert inr.dimension == usd.dimension == "currency"
    assert inr.canonical != usd.canonical


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit_raw, value_raw, canonical, dimension",
    [
        ("Mn Tons", "1.4", "tonne", "mass"),
        ("thousand tonnes", "1,429", "tonne", "mass"),
        ("", "1.6%", "percent", "ratio"),
        ("per cent", "6.3", "percent", "ratio"),
        ("bps", "25", "bps", "ratio"),
        ("", "₹8,142 Cr", "INR", "currency"),
        ("sq ft", "20,000", "sq_ft", "area"),
        ("days", "45", "day", "duration"),
        ("shipments", "740", "count", "count"),
        ("", "a chairman", None, None),
    ],
)
def test_canonical_unit(unit_raw, value_raw, canonical, dimension):
    info = N.canonical_unit(unit_raw, value_raw)
    assert info.canonical == canonical
    assert info.dimension == dimension


def test_conversion_within_a_dimension():
    tonne = N.canonical_unit("tonnes")
    kg = N.canonical_unit("kg")
    assert N.convert(1.0, tonne, kg) == pytest.approx(1000.0)
    assert N.convert(1000.0, kg, tonne) == pytest.approx(1.0)


def test_conversion_across_dimensions_is_refused():
    assert N.convert(1.0, N.canonical_unit("tonnes"), N.canonical_unit("days")) is None
    assert N.dimensions_comparable(N.canonical_unit("tonnes"), N.canonical_unit("kg")) is True
    assert N.dimensions_comparable(N.canonical_unit("tonnes"), N.canonical_unit("%")) is False


def test_basis_points_convert_to_percent():
    assert N.convert(25.0, N.canonical_unit("bps"), N.canonical_unit("%")) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Whole values
# ---------------------------------------------------------------------------


def test_the_headline_corroboration_pair_normalizes_to_the_same_quantity():
    a = N.normalize_value("1.4 Mn Tons")
    b = N.normalize_value("1,429", "thousand tonnes")
    assert a.value_num == pytest.approx(1_400_000)
    assert b.value_num == pytest.approx(1_429_000)
    assert a.unit_dimension == b.unit_dimension == "mass"
    delta = N.relative_delta(a.value_num, b.value_num)
    tolerance = N.relative_tolerance("1.4", "1,429", a.value_num, b.value_num)
    assert delta < tolerance, "one decimal place of precision must cover this gap"


def test_a_real_disagreement_stays_a_disagreement():
    a, b = 1.4, 1.9
    assert N.relative_delta(a, b) > N.relative_tolerance("1.4", "1.9", a, b)


def test_rounding_of_a_derived_difference_is_within_tolerance():
    # A stated rise of 578 where the arithmetic gives 579.
    assert N.relative_delta(578, 579) < N.relative_tolerance("578", "579", 578, 579)


def test_magnitude_is_kept_for_explanation_and_folded_into_the_number():
    value = N.normalize_value("₹8,142 Cr")
    assert value.value_num == pytest.approx(8.142e10)
    assert value.magnitude == pytest.approx(1e7)
    assert value.magnitude_label == "cr"
    assert value.currency == "INR"
    assert value.value_text == "₹8,142 Cr"


def test_percentage_is_never_scaled_by_a_neighbouring_magnitude_word():
    value = N.normalize_value("1.6%", "per cent")
    assert value.value_num == pytest.approx(1.6)
    assert value.magnitude == 1.0


def test_non_numeric_values_keep_their_text():
    value = N.normalize_value("Sahil Barua", value_type="string")
    assert value.value_num is None
    assert value.value_text == "Sahil Barua"


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, start, end, basis",
    [
        ("FY24", date(2023, 4, 1), date(2024, 3, 31), "fiscal_in"),
        ("FY 2023-24", date(2023, 4, 1), date(2024, 3, 31), "fiscal_in"),
        ("FY2023-24", date(2023, 4, 1), date(2024, 3, 31), "fiscal_in"),
        ("2024-25", date(2024, 4, 1), date(2025, 3, 31), "fiscal_in"),
        ("Q4 FY24", date(2024, 1, 1), date(2024, 3, 31), "fiscal_in"),
        ("Q1 FY24", date(2023, 4, 1), date(2023, 6, 30), "fiscal_in"),
        ("H1 FY24", date(2023, 4, 1), date(2023, 9, 30), "fiscal_in"),
        ("9M FY24", date(2023, 4, 1), date(2023, 12, 31), "fiscal_in"),
        ("CY2024", date(2024, 1, 1), date(2024, 12, 31), "calendar"),
        ("2024", date(2024, 1, 1), date(2024, 12, 31), "calendar"),
        ("March 2024", date(2024, 3, 1), date(2024, 3, 31), "calendar"),
    ],
)
def test_parse_period(label, start, end, basis):
    period = N.parse_period(label)
    assert (period.start, period.end, period.basis) == (start, end, basis)


@pytest.mark.parametrize(
    "label", ["as of March 31, 2024", "as at 31 March 2024", "as on 2024-03-31"]
)
def test_point_in_time_has_equal_start_and_end(label):
    period = N.parse_period(label)
    assert period.start == period.end == date(2024, 3, 31)
    assert period.basis == "point_in_time"
    assert period.is_point


def test_fiscal_start_month_is_configurable_and_the_basis_is_recorded():
    indian = N.parse_period("FY24")
    american = N.parse_period("FY24", fiscal_start_month=10)
    assert indian.start == date(2023, 4, 1)
    assert american.start == date(2023, 10, 1)
    assert indian.basis == "fiscal_in", "the assumption is stored, not hidden"


def test_unparseable_period_is_unknown_not_guessed():
    period = N.parse_period("sometime recently")
    assert period.start is None and period.basis == "unknown"


def test_period_containment_and_overlap():
    year = N.parse_period("FY24")
    quarter = N.parse_period("Q4 FY24")
    other = N.parse_period("FY23")
    assert N.period_contains(year, quarter)
    assert not N.period_contains(quarter, year)
    assert N.periods_overlap(year, quarter)
    assert not N.periods_overlap(year, other)
    assert N.periods_equal(year, N.parse_period("FY 2023-24"))


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_scope_tags_are_mapped_to_the_shared_vocabulary_and_sorted():
    assert N.normalize_scope_tags(["Adj.", "Consolidated"]) == ["adjusted", "consolidated"]
    assert N.normalize_scope_tags(["Pro Forma"]) == ["pro_forma"]
    assert N.normalize_scope_tags([]) == []


def test_unknown_scope_tags_survive_as_snake_case():
    assert N.normalize_scope_tags(["Excluding One Off Items"]) == ["excluding_one_off_items"]


@pytest.mark.parametrize(
    "a, b",
    [
        (["adjusted"], ["reported"]),
        (["standalone"], ["consolidated"]),
        (["estimate"], ["provisional"]),
        (["segment"], ["total"]),
        (["since_inception"], ["total"]),
    ],
)
def test_known_scope_shifts_are_detected(a, b):
    shifted, reasons = N.scope_shift(a, b)
    assert shifted and reasons


def test_identical_scopes_do_not_shift():
    assert N.scope_shift(["reported"], ["reported"]) == (False, [])


# ---------------------------------------------------------------------------
# Subjects, predicates, addresses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Delhivery Limited", "delhivery"),
        ("Delhivery Ltd.", "delhivery"),
        ("  The  Reserve Bank of India ", "reserve bank of india"),
        ("Acme Private Limited", "acme"),
        ("Express Parcel (segment)", "express parcel"),
    ],
)
def test_canonical_subject(raw, expected):
    assert N.canonical_subject(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Revenue from services", "revenue_from_services"),
        ("  EBITDA Margin  ", "ebitda_margin"),
        ("the real GDP growth of", "real_gdp_growth"),
    ],
)
def test_canonical_predicate(raw, expected):
    assert N.canonical_predicate(raw) == expected


def test_address_normalizer_extracts_the_postal_code_separately():
    result = N.normalize_address("Plot 5, 2nd Flr, MG Rd., Bengaluru - 560001")
    assert result.postal_code == "560001"
    assert "ROAD" in result.tokens and "FLOOR" in result.tokens
    assert "560001" not in result.normalized


def test_addresses_written_differently_normalize_to_the_same_string():
    a = N.normalize_address("N24-N34, Air Cargo Logistics Centre-II, New Delhi 110037")
    b = N.normalize_address("N24 - N34 Air Cargo Logistics Centre II, New Delhi, 110037.")
    assert a.postal_code == b.postal_code == "110037"
    assert a.normalized == b.normalized


def test_different_addresses_do_not_collapse():
    a = N.normalize_address("Plot 5, MG Road, Bengaluru 560001")
    b = N.normalize_address("Plot 5, MG Road, Bengaluru 560002")
    assert a.postal_code != b.postal_code


def test_clean_text_folds_unicode_and_whitespace():
    assert N.clean_text("Revenue  was  8,142") == "Revenue was 8,142"
    assert N.snake_case("Revenue From Services!") == "revenue_from_services"
