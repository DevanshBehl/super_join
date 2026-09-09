"""Deterministic normalizers. No LLM, no I/O, no globals, fully unit tested.

Everything here is a pure function, because these are the parts of the system
that must be right every single time. The extraction model is allowed to be
fuzzy; the arithmetic is not.

Two conventions run through the module:

*   ``value_num`` has the magnitude folded in but keeps its own unit. So
    ``1.4 Mn Tons`` becomes ``1_400_000`` tonnes, not kilograms. Conversion to
    a dimension's base unit happens only when two facts are compared, which
    keeps the stored number recognisable next to the raw string.
*   Nothing is discarded. Approximations, ranges, and bounds become qualifiers
    rather than being rounded away.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Magnitudes
# ---------------------------------------------------------------------------

MAGNITUDES: dict[str, float] = {
    "hundred": 1e2,
    "k": 1e3,
    "'000": 1e3,
    "000s": 1e3,
    "thousand": 1e3,
    "thousands": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "mn": 1e6,
    "m": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "cr": 1e7,
    "crore": 1e7,
    "crores": 1e7,
    "cr.": 1e7,
    "bn": 1e9,
    "b": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "tn": 1e12,
    "trillion": 1e12,
    "trillions": 1e12,
}

# Compound forms must be tried before their parts.
COMPOUND_MAGNITUDES: dict[str, float] = {
    "lakh crore": 1e12,
    "lakh crores": 1e12,
    "thousand crore": 1e10,
    "thousand crores": 1e10,
}

# Single letter magnitudes are only trusted when glued to a number, because a
# bare "m" or "b" in prose usually means something else.
_AMBIGUOUS_MAGNITUDES = {"m", "b", "k"}

# ---------------------------------------------------------------------------
# Currencies
# ---------------------------------------------------------------------------

CURRENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"₹", "INR"),
    (r"\brs\.?\b", "INR"),
    (r"\binr\b", "INR"),
    (r"\brupees?\b", "INR"),
    (r"\bus\s*\$", "USD"),
    (r"\busd\b", "USD"),
    (r"\bus\s*dollars?\b", "USD"),
    (r"\$", "USD"),
    (r"€", "EUR"),
    (r"\beur\b", "EUR"),
    (r"£", "GBP"),
    (r"\bgbp\b", "GBP"),
    (r"¥", "JPY"),
    (r"\bjpy\b", "JPY"),
)

# ---------------------------------------------------------------------------
# Units. Each entry maps an alias to (canonical, dimension, factor to base).
# ---------------------------------------------------------------------------

_UNIT_TABLE: tuple[tuple[tuple[str, ...], str, str, float], ...] = (
    # ratio
    (("%", "percent", "per cent", "pct", "percentage", "percentage points", "ppt"), "percent", "ratio", 1.0),
    (("bps", "bp", "basis points", "basis point"), "bps", "ratio", 0.01),
    (("x", "times"), "times", "ratio_multiple", 1.0),
    # mass
    (("tonne", "tonnes", "ton", "tons", "mt", "metric ton", "metric tonnes", "metric tonne"), "tonne", "mass", 1000.0),
    (("kg", "kgs", "kilogram", "kilograms"), "kg", "mass", 1.0),
    (("g", "gram", "grams"), "g", "mass", 0.001),
    # count
    (("shipments", "shipment", "parcels", "parcel", "units", "unit", "count", "nos", "no.", "number"), "count", "count", 1.0),
    (("people", "persons", "employees", "headcount", "staff"), "people", "count", 1.0),
    (("customers", "clients"), "customers", "count", 1.0),
    (("orders", "order"), "orders", "count", 1.0),
    (("stores", "facilities", "centres", "centers", "hubs", "gateways"), "facilities", "count", 1.0),
    # area
    (("sq ft", "sq. ft.", "sqft", "square feet", "square foot", "sq.ft"), "sq_ft", "area", 0.09290304),
    (("sq m", "sqm", "square metres", "square meters", "sq. m."), "sq_m", "area", 1.0),
    (("acre", "acres"), "acre", "area", 4046.8564224),
    # length
    (("km", "kilometre", "kilometres", "kilometer", "kilometers"), "km", "length", 1000.0),
    (("m", "metre", "metres", "meter", "meters"), "m", "length", 1.0),
    # duration
    (("day", "days"), "day", "duration", 1.0),
    (("week", "weeks"), "week", "duration", 7.0),
    (("month", "months"), "month", "duration", 30.436875),
    (("year", "years", "yrs"), "year", "duration", 365.2425),
    (("hour", "hours", "hrs", "hr"), "hour", "duration", 1 / 24),
    (("minute", "minutes", "mins"), "minute", "duration", 1 / 1440),
)

_UNIT_LOOKUP: dict[str, tuple[str, str, float]] = {}
for aliases, canonical, dimension, factor in _UNIT_TABLE:
    for alias in aliases:
        _UNIT_LOOKUP.setdefault(alias, (canonical, dimension, factor))

CURRENCY_DIMENSION = "currency"

# ---------------------------------------------------------------------------
# Scope vocabulary
# ---------------------------------------------------------------------------

SCOPE_ALIASES: dict[str, str] = {
    "consol": "consolidated",
    "consolidated": "consolidated",
    "group": "consolidated",
    "standalone": "standalone",
    "stand alone": "standalone",
    "separate": "standalone",
    "proforma": "pro_forma",
    "pro forma": "pro_forma",
    "pro-forma": "pro_forma",
    "adj": "adjusted",
    "adj.": "adjusted",
    "adjusted": "adjusted",
    "reported": "reported",
    "actual": "reported",
    "actuals": "reported",
    "provisional": "provisional",
    "prov": "provisional",
    "revised": "revised",
    "re": "revised",
    "first revised": "revised",
    "second revised": "revised",
    "estimate": "estimate",
    "estimated": "estimate",
    "advance estimate": "estimate",
    "ae": "estimate",
    "projection": "forecast",
    "projected": "forecast",
    "forecast": "forecast",
    "budgeted": "forecast",
    "be": "forecast",
    "segment": "segment",
    "total": "total",
    "since inception": "since_inception",
    "cumulative": "since_inception",
    "excluding traded goods": "excluding_traded_goods",
    "ex traded goods": "excluding_traded_goods",
    "yoy": "year_on_year",
    "qoq": "quarter_on_quarter",
    "seasonally adjusted": "seasonally_adjusted",
    "annualised": "annualised",
    "annualized": "annualised",
    "per cent of gdp": "percent_of_gdp",
    "percent of gdp": "percent_of_gdp",
}

# Groups of tags where holding different members explains a value difference.
SCOPE_SHIFT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"adjusted", "reported"}),
    frozenset({"standalone", "consolidated"}),
    frozenset({"pro_forma", "reported"}),
    frozenset({"estimate", "provisional", "revised", "reported", "forecast"}),
    frozenset({"segment", "total"}),
    frozenset({"since_inception", "total"}),
    frozenset({"excluding_traded_goods", "total"}),
    frozenset({"seasonally_adjusted", "reported"}),
    frozenset({"annualised", "reported"}),
)

LEGAL_SUFFIXES: tuple[str, ...] = (
    "private limited",
    "public limited company",
    "limited liability partnership",
    "limited",
    "ltd",
    "pvt",
    "plc",
    "llp",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "gmbh",
    "s.a.",
    "n.v.",
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Unicode normalise, replace exotic dashes and spaces, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[   ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_accents(text: str) -> str:
    """Strip combining marks so accented spellings match their plain form."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def snake_case(text: str) -> str:
    text = fold_accents(clean_text(text)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d[\d,٬]*(?:\.\d+)?")
_APPROX_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:^|\s)(?:~|approx\.?|approximately|about|around|roughly|nearly|almost|circa|c\.)\s*", "approximate"),
    (r"(?:^|\s)(?:over|more than|greater than|above|at least|in excess of|\+|>=|>)\s*", "lower_bound"),
    (r"(?:^|\s)(?:under|less than|below|up to|at most|fewer than|<=|<)\s*", "upper_bound"),
)
_RANGE_RE = re.compile(
    r"(?P<low>\d[\d,]*(?:\.\d+)?)\s*(?:-|to|and)\s*(?P<high>\d[\d,]*(?:\.\d+)?)", re.IGNORECASE
)


@dataclass(slots=True)
class NumberParse:
    value: float | None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    digits_text: str = ""


def _strip_grouping(token: str) -> str:
    """Remove digit group separators. Indian and western grouping both work."""
    return token.replace(",", "").replace("٬", "")


def parse_number(raw: str) -> NumberParse:
    """Parse the numeric content of a raw string, keeping every qualifier.

    Handles Indian and western digit grouping, parentheses as negative,
    percent and basis point suffixes, ranges, and approximation words.
    """
    text = clean_text(raw)
    if not text:
        return NumberParse(None)

    qualifiers: dict[str, Any] = {}
    negative_parens = bool(re.search(r"\(\s*[-+]?[\d.,]", text)) and ")" in text

    for pattern, name in _APPROX_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            qualifiers["bound"] = name
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    range_match = _RANGE_RE.search(text)
    if range_match and not _looks_like_period(raw):
        low = float(_strip_grouping(range_match.group("low")))
        high = float(_strip_grouping(range_match.group("high")))
        if high > low:
            qualifiers["range_low"] = low
            qualifiers["range_high"] = high
            midpoint = (low + high) / 2
            return NumberParse(midpoint, qualifiers, range_match.group(0))

    match = _NUMBER_RE.search(text)
    if not match:
        return NumberParse(None, qualifiers)
    token = match.group(0)
    try:
        value = float(_strip_grouping(token))
    except ValueError:
        return NumberParse(None, qualifiers)
    if negative_parens and value > 0:
        value = -value
        qualifiers["parentheses_negative"] = True
    return NumberParse(value, qualifiers, token)


def significant_digit_place(raw: str) -> float | None:
    """The place value of the last significant digit written in a raw string.

    ``1.4`` returns 0.1 and ``1,429`` returns 1.0. This is what lets the rule
    engine treat ``1.4 Mn`` and ``1,429 thousand`` as agreeing while keeping
    ``1.4`` and ``1.9`` apart.
    """
    match = _NUMBER_RE.search(clean_text(raw))
    if not match:
        return None
    token = _strip_grouping(match.group(0))
    if "." in token:
        return 10.0 ** -len(token.split(".", 1)[1])
    return 1.0


# ---------------------------------------------------------------------------
# Magnitude and currency
# ---------------------------------------------------------------------------


def detect_currency(*texts: str) -> str | None:
    blob = clean_text(" ".join(t for t in texts if t)).lower()
    if not blob:
        return None
    for pattern, code in CURRENCY_PATTERNS:
        if re.search(pattern, blob):
            return code
    return None


def detect_magnitude(*texts: str) -> tuple[float, str | None]:
    """Return ``(multiplier, label)``. Missing magnitude is ``(1.0, None)``."""
    blob = clean_text(" ".join(t for t in texts if t)).lower()
    if not blob:
        return 1.0, None
    for phrase, factor in COMPOUND_MAGNITUDES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", blob):
            return factor, phrase
    for token, factor in sorted(MAGNITUDES.items(), key=lambda kv: -len(kv[0])):
        if token in _AMBIGUOUS_MAGNITUDES:
            # Only trust a single letter when it is attached to a number.
            if re.search(rf"\d\s*{re.escape(token)}\b", blob):
                return factor, token
            continue
        pattern = rf"(?<![a-z]){re.escape(token)}(?![a-z])" if token.isalpha() else re.escape(token)
        if re.search(pattern, blob):
            return factor, token
    return 1.0, None


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UnitInfo:
    canonical: str | None
    dimension: str | None
    to_base: float = 1.0


def canonical_unit(unit_raw: str, value_raw: str = "", currency: str | None = None) -> UnitInfo:
    """Resolve a unit alias to a canonical unit plus a dimension.

    A currency wins over everything else, because a rupee figure is a currency
    figure whatever noun follows it.
    """
    blob = clean_text(f"{unit_raw} {value_raw}").lower()
    currency = currency or detect_currency(unit_raw, value_raw)
    if currency and not re.search(r"%|per cent|percent", blob):
        return UnitInfo(currency, CURRENCY_DIMENSION, 1.0)
    if not blob:
        return UnitInfo(None, None, 1.0)

    for alias in sorted(_UNIT_LOOKUP, key=len, reverse=True):
        if alias.isalpha():
            pattern = rf"(?<![a-z]){re.escape(alias)}(?![a-z])"
        else:
            pattern = re.escape(alias)
        if re.search(pattern, blob):
            if alias in _AMBIGUOUS_MAGNITUDES and not re.search(rf"\d\s*{re.escape(alias)}\b", blob):
                continue
            canonical, dimension, factor = _UNIT_LOOKUP[alias]
            return UnitInfo(canonical, dimension, factor)
    return UnitInfo(None, None, 1.0)


def convert(value: float, source: UnitInfo, target: UnitInfo) -> float | None:
    """Convert within a dimension. Returns None when the dimensions differ."""
    if source.dimension is None or source.dimension != target.dimension:
        return None
    if not target.to_base:
        return None
    return value * source.to_base / target.to_base


def dimensions_comparable(a: UnitInfo, b: UnitInfo) -> bool:
    if a.dimension is None and b.dimension is None:
        return True
    return a.dimension == b.dimension


# ---------------------------------------------------------------------------
# Whole value normalization
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedValue:
    value_num: float | None
    value_text: str | None
    unit_canonical: str | None
    unit_dimension: str | None
    unit_to_base: float
    magnitude: float | None
    magnitude_label: str | None
    currency: str | None
    qualifiers: dict[str, Any] = field(default_factory=dict)

    @property
    def unit_info(self) -> UnitInfo:
        return UnitInfo(self.unit_canonical, self.unit_dimension, self.unit_to_base)


def normalize_value(value_raw: str, unit_raw: str = "", value_type: str = "numeric") -> NormalizedValue:
    """Fold magnitude into the number and resolve unit, dimension, and currency."""
    currency = detect_currency(value_raw, unit_raw)
    unit = canonical_unit(unit_raw, value_raw, currency)

    if value_type != "numeric":
        return NormalizedValue(
            value_num=None,
            value_text=clean_text(value_raw),
            unit_canonical=unit.canonical,
            unit_dimension=unit.dimension,
            unit_to_base=unit.to_base,
            magnitude=None,
            magnitude_label=None,
            currency=currency,
        )

    parsed = parse_number(value_raw)
    magnitude, label = detect_magnitude(value_raw, unit_raw)
    # A percentage is never scaled by a magnitude word sitting next to it.
    if unit.dimension == "ratio":
        magnitude, label = 1.0, None

    value_num = None if parsed.value is None else parsed.value * magnitude
    qualifiers = dict(parsed.qualifiers)
    for key in ("range_low", "range_high"):
        if key in qualifiers:
            qualifiers[key] = qualifiers[key] * magnitude

    return NormalizedValue(
        value_num=value_num,
        value_text=clean_text(value_raw),
        unit_canonical=unit.canonical,
        unit_dimension=unit.dimension,
        unit_to_base=unit.to_base,
        magnitude=magnitude,
        magnitude_label=label,
        currency=currency,
        qualifiers=qualifiers,
    )


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

DEFAULT_FISCAL_START_MONTH = 4  # India: 1 April to 31 March.

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_FY_SPAN = re.compile(r"\bf\.?y\.?\s*(\d{4})\s*[-/]\s*(\d{2,4})\b", re.IGNORECASE)
_FY_SHORT = re.compile(r"\bf\.?y\.?\s*'?(\d{2}|\d{4})\b", re.IGNORECASE)
_BARE_SPAN = re.compile(r"\b(\d{4})\s*[-/]\s*(\d{2})\b")
_QUARTER = re.compile(r"\bq([1-4])\s*[:\-]?\s*(?:of\s*)?(f\.?y\.?\s*'?\d{2,4}(?:\s*[-/]\s*\d{2,4})?|c\.?y\.?\s*\d{4}|\d{4}(?:\s*[-/]\s*\d{2,4})?)", re.IGNORECASE)
_HALF = re.compile(r"\bh([12])\s*[:\-]?\s*(f\.?y\.?\s*'?\d{2,4}(?:\s*[-/]\s*\d{2,4})?|c\.?y\.?\s*\d{4}|\d{4}(?:\s*[-/]\s*\d{2,4})?)", re.IGNORECASE)
_NM = re.compile(r"\b(\d{1,2})\s*m\s*[:\-]?\s*(f\.?y\.?\s*'?\d{2,4}(?:\s*[-/]\s*\d{2,4})?|\d{4}(?:\s*[-/]\s*\d{2,4})?)", re.IGNORECASE)
_CY = re.compile(r"\b(?:c\.?y\.?|calendar\s+year)\s*(\d{4})\b", re.IGNORECASE)
_AS_OF = re.compile(r"\bas\s+(?:of|at|on)\b", re.IGNORECASE)
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MDY = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?[\s,-]+(\d{{4}}|\d{{2}})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b(19|20)(\d{2})\b")


@dataclass(slots=True)
class Period:
    start: date | None
    end: date | None
    basis: str = "unknown"
    label: str = ""

    @property
    def is_point(self) -> bool:
        return self.start is not None and self.start == self.end


def _looks_like_period(raw: str) -> bool:
    text = clean_text(raw)
    return bool(
        _FY_SPAN.search(text)
        or _FY_SHORT.search(text)
        or _QUARTER.search(text)
        or _CY.search(text)
        or (_BARE_SPAN.search(text) and not re.search(r"\d\s*[-/]\s*\d+\s*(?:%|per cent)", text))
    )


def _month_end(year: int, month: int) -> date:
    """Last calendar day of a month."""
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


def _expand_year(token: str, century_hint: int = 2000) -> int:
    token = token.strip()
    if len(token) == 4:
        return int(token)
    return century_hint + int(token)


def _fiscal_span(end_year: int, fiscal_start_month: int) -> tuple[date, date]:
    """Fiscal year labelled by its ending year, for example FY24 ends March 2024."""
    if fiscal_start_month == 1:
        return date(end_year, 1, 1), date(end_year, 12, 31)
    start = date(end_year - 1, fiscal_start_month, 1)
    end = _month_end(end_year, fiscal_start_month - 1)
    return start, end


def parse_period(label: str, fiscal_start_month: int = DEFAULT_FISCAL_START_MONTH) -> Period:
    """Parse a period label into start, end, and basis.

    The basis is stored rather than silently resolved. ``fiscal_in`` means an
    Indian style fiscal year was assumed, and a reader can see that assumption
    instead of having to guess whether it was made.
    """
    text = clean_text(label)
    if not text:
        return Period(None, None, "unknown", "")

    lowered = text.lower()

    match = _QUARTER.search(text)
    if match:
        quarter = int(match.group(1))
        base = parse_period(match.group(2), fiscal_start_month)
        if base.start is not None:
            start_month_index = (quarter - 1) * 3
            start = _add_months(base.start, start_month_index)
            end = _month_end(*_add_months_ym(base.start, start_month_index + 2))
            return Period(start, end, base.basis, text)

    match = _HALF.search(text)
    if match:
        half = int(match.group(1))
        base = parse_period(match.group(2), fiscal_start_month)
        if base.start is not None:
            offset = (half - 1) * 6
            start = _add_months(base.start, offset)
            end = _month_end(*_add_months_ym(base.start, offset + 5))
            return Period(start, end, base.basis, text)

    match = _NM.search(text)
    if match:
        months = int(match.group(1))
        base = parse_period(match.group(2), fiscal_start_month)
        if base.start is not None and 1 <= months <= 12:
            end = _month_end(*_add_months_ym(base.start, months - 1))
            return Period(base.start, end, base.basis, text)

    match = _CY.search(text)
    if match:
        year = int(match.group(1))
        return Period(date(year, 1, 1), date(year, 12, 31), "calendar", text)

    match = _FY_SPAN.search(text)
    if match:
        start_year = int(match.group(1))
        end_year = _expand_year(match.group(2), (start_year // 100) * 100)
        if end_year < start_year:
            end_year += 100
        start, end = _fiscal_span(end_year, fiscal_start_month)
        return Period(start, end, "fiscal_in", text)

    match = _FY_SHORT.search(text)
    if match:
        end_year = _expand_year(match.group(1))
        start, end = _fiscal_span(end_year, fiscal_start_month)
        return Period(start, end, "fiscal_in", text)

    # A specific day beats a month or a year.
    day = _parse_exact_date(text)
    if day is not None:
        return Period(day, day, "point_in_time", text)

    match = _BARE_SPAN.search(text)
    if match:
        start_year = int(match.group(1))
        end_year = _expand_year(match.group(2), (start_year // 100) * 100)
        if end_year <= start_year:
            end_year += 100
        if end_year - start_year == 1:
            start, end = _fiscal_span(end_year, fiscal_start_month)
            return Period(start, end, "fiscal_in", text)
        return Period(date(start_year, 1, 1), date(end_year, 12, 31), "calendar", text)

    match = _MONTH_YEAR.search(text)
    if match:
        month = _MONTHS[match.group(1).lower()]
        year = _expand_year(match.group(2))
        start = date(year, month, 1)
        end = _month_end(year, month)
        basis = "point_in_time" if _AS_OF.search(lowered) else "calendar"
        return Period(start if basis == "calendar" else end, end, basis, text)

    match = _BARE_YEAR.search(text)
    if match:
        year = int(match.group(0))
        return Period(date(year, 1, 1), date(year, 12, 31), "calendar", text)

    return Period(None, None, "unknown", text)


def _parse_exact_date(text: str) -> date | None:
    match = _ISO.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = _DMY.search(text)
    if match:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1)))
        except ValueError:
            return None
    match = _MDY.search(text)
    if match:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(1).lower()], int(match.group(2)))
        except ValueError:
            return None
    return None


def _add_months_ym(anchor: date, months: int) -> tuple[int, int]:
    index = anchor.year * 12 + (anchor.month - 1) + months
    return index // 12, index % 12 + 1


def _add_months(anchor: date, months: int) -> date:
    year, month = _add_months_ym(anchor, months)
    return date(year, month, 1)


def periods_overlap(a: Period, b: Period) -> bool:
    if not (a.start and a.end and b.start and b.end):
        return False
    return a.start <= b.end and b.start <= a.end


def period_contains(outer: Period, inner: Period) -> bool:
    if not (outer.start and outer.end and inner.start and inner.end):
        return False
    return outer.start <= inner.start and inner.end <= outer.end and (
        outer.start < inner.start or inner.end < outer.end
    )


def periods_equal(a: Period, b: Period) -> bool:
    return bool(a.start and b.start) and a.start == b.start and a.end == b.end


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def normalize_scope_tags(tags: list[str] | None) -> list[str]:
    """Map free tags onto the shared vocabulary, keeping unknown tags as is."""
    out: list[str] = []
    for tag in tags or []:
        cleaned = clean_text(str(tag)).lower().strip(" .,")
        if not cleaned:
            continue
        mapped = SCOPE_ALIASES.get(cleaned) or SCOPE_ALIASES.get(cleaned.replace("_", " ")) or snake_case(cleaned)
        if mapped and mapped not in out:
            out.append(mapped)
    return sorted(out)


def scope_shift(tags_a: list[str], tags_b: list[str]) -> tuple[bool, list[str]]:
    """Does the symmetric difference contain a known scope shifting pair?"""
    set_a, set_b = set(tags_a or []), set(tags_b or [])
    difference = set_a ^ set_b
    if not difference:
        return False, []
    reasons: list[str] = []
    for group in SCOPE_SHIFT_GROUPS:
        in_a = set_a & group
        in_b = set_b & group
        if in_a and in_b and in_a != in_b:
            reasons.append(f"{sorted(in_a)[0]} versus {sorted(in_b)[0]}")
        elif (in_a or in_b) and (group & difference) and not (in_a and in_b):
            reasons.append(f"{sorted(group & difference)[0]} present on one side only")
    return bool(reasons), sorted(set(reasons))


# ---------------------------------------------------------------------------
# Subjects and predicates
# ---------------------------------------------------------------------------


def canonical_subject(subject: str) -> str:
    """Lowercase, drop legal suffixes and punctuation, collapse whitespace."""
    text = fold_accents(clean_text(subject)).lower()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9&/ .-]+", " ", text)
    for suffix in LEGAL_SUFFIXES:
        text = re.sub(rf"(?:,|\s)\s*{re.escape(suffix)}\.?\s*$", " ", text)
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[.\-]+$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_predicate(predicate: str) -> str:
    """A snake_case predicate with filler words removed."""
    text = snake_case(predicate)
    text = re.sub(r"^(the|a|an)_", "", text)
    text = re.sub(r"_(of|for|in|at|on|the)$", "", text)
    return text


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------

_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "opp": "opposite",
    "bldg": "building",
    "flr": "floor",
    "fl": "floor",
    "apt": "apartment",
    "ste": "suite",
    "no": "number",
    "nr": "near",
    "ind": "industrial",
    "estt": "estate",
    "pkwy": "parkway",
    "sec": "sector",
    "ph": "phase",
    "dist": "district",
    "extn": "extension",
    "ext": "extension",
    "mkt": "market",
    "twp": "township",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}
_POSTAL_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.IGNORECASE)


@dataclass(slots=True)
class NormalizedAddress:
    normalized: str
    postal_code: str | None
    tokens: list[str]


def normalize_address(address: str) -> NormalizedAddress:
    """Uppercase, expand abbreviations, drop punctuation, pull out the postal code.

    The postal code is compared on its own because it is the single highest
    signal token in an address and it survives every formatting difference.
    """
    text = clean_text(address).upper()
    postal_match = _POSTAL_RE.search(text)
    postal = None
    if postal_match:
        postal = re.sub(r"\s+", "", postal_match.group(1)).upper()
        text = text.replace(postal_match.group(1), " ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    words = [w for w in text.split() if w]
    expanded = [_ADDRESS_ABBREVIATIONS.get(w.lower(), w.lower()).upper() for w in words]
    normalized = " ".join(expanded)
    return NormalizedAddress(normalized=normalized, postal_code=postal, tokens=expanded)


# ---------------------------------------------------------------------------
# Numeric agreement tolerance
# ---------------------------------------------------------------------------


def relative_tolerance(raw_a: str, raw_b: str, value_a: float, value_b: float, floor: float = 0.005) -> float:
    """Tolerance driven by the precision each side actually claims.

    A value written as ``1.4`` claims one decimal place, so anything within
    half of that place agrees with it. The looser of the two sides wins,
    because agreement cannot be stricter than the coarser claim.
    """
    scale = max(abs(value_a), abs(value_b), 1e-12)
    tolerances = [floor]
    for raw, value in ((raw_a, value_a), (raw_b, value_b)):
        place = significant_digit_place(raw)
        if place is None or value == 0:
            continue
        written = parse_number(raw).value
        if written in (None, 0):
            continue
        # The raw string may be written in a magnitude, so the place value has
        # to be scaled by the same factor that was folded into the number.
        scaled_place = place * abs(value) / abs(written)
        tolerances.append((scaled_place / 2) / scale)
    return max(tolerances)


def relative_delta(a: float, b: float) -> float:
    """Symmetric relative difference. Zero against zero is zero."""
    scale = max(abs(a), abs(b))
    if scale == 0:
        return 0.0
    return abs(a - b) / scale
