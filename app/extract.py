"""Gemini structured extraction and the evidence verification gate.

Extraction is allowed to be imperfect. Evidence verification is not: a fact
whose quote cannot be located in its own source chunk never enters the active
set. It is written to the database with ``verified = false`` and shows up in
the quarantine view, because a silent drop would hide exactly the failures a
reviewer wants to see.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

from rapidfuzz import fuzz

import config
from app import db, llm, registry
from app.chunking import Chunk
from app.models import RawDocumentMeta, RawFact, RawFactList
from app.normalize import (
    canonical_predicate,
    canonical_subject,
    clean_text,
    normalize_scope_tags,
    normalize_value,
    parse_period,
)
from app.parsing import ParsedDocument, resolve_page

# ---------------------------------------------------------------------------
# Cheap pre filter: do not spend a call on a chunk that cannot contain a fact
# ---------------------------------------------------------------------------

_BOILERPLATE_MARKERS = (
    "safe harbour",
    "safe harbor",
    "disclaimer",
    "forward-looking statement",
    "forward looking statement",
    "no representation or warranty",
    "table of contents",
    "this page has been intentionally left blank",
    "all rights reserved",
)
_DOT_LEADER = re.compile(r"\.{4,}|\s\.\s\.\s\.")


def is_boilerplate(text: str, kind: str = "prose") -> tuple[bool, str]:
    """Decide whether to skip a chunk before paying for a call.

    The rules are structural, not corpus specific: too little text to hold a
    claim, a page of dot leaders, almost no letters, or a block whose only
    content is a legal disclaimer.
    """
    stripped = text.strip()
    if len(stripped) < config.CHUNK_MIN_CHARS:
        return True, "too_short"

    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines:
        leader_lines = sum(1 for line in lines if _DOT_LEADER.search(line))
        if leader_lines / len(lines) > 0.3:
            return True, "table_of_contents"

    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters / max(len(stripped), 1) < config.BOILERPLATE_MIN_ALPHA_RATIO and kind != "table":
        return True, "low_alpha_ratio"

    lowered = stripped.lower()
    if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
        digits = sum(1 for ch in stripped if ch.isdigit())
        if digits / max(len(stripped), 1) < 0.01:
            return True, "boilerplate_notice"
    return False, ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_INSTRUCTIONS = """\
You extract checkable facts from one excerpt of a document.

A fact is an assertion the document makes about a named subject, carrying a
value, that a careful reader could check by pointing at one specific sentence
or one specific table cell in the excerpt below. If you cannot point at such a
span, it is not a fact and you must not report it.

Extract both kinds:
1. Numeric facts: financial figures, counts, rates, ratios, margins, growth
   percentages, dates, tonnages, headcounts, areas.
2. Semantic facts: roles and appointments, statuses, addresses, corporate
   relationships, definitions, and qualitative claims that are checkable.

Rules that decide whether a fact is accepted:
- evidence_quote MUST be a verbatim contiguous substring of the excerpt,
  copied character for character. Inventing, paraphrasing, merging separate
  lines, or tidying a quote invalidates the fact and it will be discarded.
  Keep the quote short enough to be a single span and long enough to contain
  both the subject and the value where possible.
- subject is the named thing the claim is about, written as the document
  writes it.
- predicate is a snake_case property name. Reuse a predicate from the known
  vocabulary below whenever the meaning matches. Coin a new snake_case
  predicate only when nothing in the vocabulary means the same thing.
- value_raw is the value exactly as printed, including the currency symbol,
  the magnitude word, brackets, and the percent sign if present. Do not
  reformat it and do not do arithmetic.
- period_label_raw is the period exactly as the document expresses it, for
  example "FY24", "Q4 FY24", "2024-25", "as of March 31, 2024". Leave it empty
  when the document states no period. Separately give your best structured
  guess at period_start and period_end as ISO dates, and period_basis as one
  of fiscal_in, calendar, point_in_time, unknown.
- scope_tags describe what qualifies the figure. The vocabulary is open. These
  are illustrative and not exhaustive: consolidated, standalone, pro_forma,
  adjusted, reported, segment, total, provisional, revised, estimate,
  forecast, since_inception, excluding_traded_goods. Invent a new tag when the
  document qualifies a figure in a way none of these covers.
- qualifiers is a list of key and value pairs for anything else that changes
  how the value should be read, for example a footnote reference, a basis of
  preparation, or a stated exchange rate.
- confidence is between 0 and 1. Add a one line extraction_note whenever you
  are unsure, for example when a number is read from a chart data label and
  the axis category is ambiguous.

Return at most {max_facts} facts, choosing the most substantive ones. Return an
empty list when the excerpt contains no checkable claim.

Known predicate vocabulary, most used first:
{vocabulary}

Document context:
{context}

Excerpt (page index {page_index}{page_label_note}):
<<<EXCERPT
{chunk}
EXCERPT
"""

DOCMETA_INSTRUCTIONS = """\
Read the opening pages of a document and report its bibliographic details.

Give the title as printed. Give the publishing organisation. Give a short
lowercase doc_type such as prospectus, annual_report, earnings_presentation,
survey, staff_report, or press_release. Give as_of_date as the ISO date the
document reports as of, which for a financial report is the end of the period
it covers. Give published_date as the ISO date it was issued. Give
primary_subject as the single organisation or economy the document is mainly
about. Leave any field empty when the pages do not say. Do not guess.

Opening pages:
<<<PAGES
{pages}
PAGES
"""


def build_extraction_prompt(
    chunk: Chunk,
    *,
    context: str,
    vocabulary: str,
    page_label: str | None = None,
) -> str:
    note = f", printed page label {page_label}" if page_label else ""
    return EXTRACTION_INSTRUCTIONS.format(
        max_facts=config.MAX_FACTS_PER_CHUNK,
        vocabulary=vocabulary,
        context=context,
        page_index=chunk.page_index_start,
        page_label_note=note,
        chunk=chunk.text,
    )


def document_context(document: dict[str, Any]) -> str:
    parts = [
        f"filename: {document.get('filename', '')}",
        f"collection: {document.get('collection', '')}",
    ]
    for key in ("title", "publisher", "doc_type", "as_of_date", "published_date"):
        value = document.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuoteMatch:
    start: int | None
    end: int | None
    score: float
    method: str

    @property
    def found(self) -> bool:
        return self.start is not None and self.end is not None


# Characters the markdown renderer adds that are not in the printed page. A
# model quoting what it can see will not reproduce them, so a quote can be
# perfectly faithful and still fail a literal match against the canonical text.
_DECORATION = set("*_~`#")
_UNICODE_FOLD = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
}


def _normalize_with_map(text: str, *, drop_decoration: bool = False) -> tuple[str, list[int]]:
    """Whitespace collapsed, case folded text plus a map back to original indexes.

    With ``drop_decoration`` the markdown emphasis characters and table pipes
    are removed as well, so a quote taken from the visible page still matches
    text that carries renderer added markup. Because those characters are
    dropped rather than substituted, every surviving character still maps back
    to its exact index in the original string.
    """
    out: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for i, ch in enumerate(text):
        ch = _UNICODE_FOLD.get(ch, ch)
        if drop_decoration and ch in _DECORATION:
            continue
        if ch.isspace() or (drop_decoration and ch == "|"):
            if previous_space:
                continue
            out.append(" ")
            index_map.append(i)
            previous_space = True
        else:
            out.append(ch.lower())
            index_map.append(i)
            previous_space = False
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


def verify_quote(quote: str, chunk_text: str, threshold: float | None = None) -> QuoteMatch:
    """Locate a quote inside its source chunk.

    Three attempts, in order: exact substring, whitespace and case insensitive
    substring, then a fuzzy partial match above the configured threshold. The
    returned offsets are always indexes into the original chunk text.
    """
    threshold = config.EVIDENCE_FUZZ_THRESHOLD if threshold is None else threshold
    quote = (quote or "").strip()
    if len(quote) < config.EVIDENCE_MIN_QUOTE_CHARS or not chunk_text:
        return QuoteMatch(None, None, 0.0, "too_short")

    exact = chunk_text.find(quote)
    if exact >= 0:
        return QuoteMatch(exact, exact + len(quote), 100.0, "exact")

    norm_chunk, chunk_map = _normalize_with_map(chunk_text)
    norm_quote, _ = _normalize_with_map(quote)
    if not norm_quote:
        return QuoteMatch(None, None, 0.0, "too_short")

    position = norm_chunk.find(norm_quote)
    if position >= 0:
        start = chunk_map[position]
        end = chunk_map[min(position + len(norm_quote) - 1, len(chunk_map) - 1)] + 1
        return QuoteMatch(start, end, 100.0, "normalized")

    # Third attempt: ignore the markup the renderer added to the page text.
    plain_chunk, plain_map = _normalize_with_map(chunk_text, drop_decoration=True)
    plain_quote, _ = _normalize_with_map(quote, drop_decoration=True)
    if plain_quote:
        position = plain_chunk.find(plain_quote)
        if position >= 0:
            start = plain_map[position]
            end = plain_map[min(position + len(plain_quote) - 1, len(plain_map) - 1)] + 1
            return QuoteMatch(start, end, 100.0, "markup_insensitive")

    alignment = fuzz.partial_ratio_alignment(plain_quote or norm_quote, plain_chunk or norm_chunk)
    if alignment is None or alignment.score < threshold:
        score = 0.0 if alignment is None else float(alignment.score)
        return QuoteMatch(None, None, score, "fuzzy_failed")
    start = plain_map[min(alignment.dest_start, len(plain_map) - 1)]
    end = plain_map[min(max(alignment.dest_end - 1, 0), len(plain_map) - 1)] + 1
    return QuoteMatch(start, end, float(alignment.score), "fuzzy")


# ---------------------------------------------------------------------------
# Fact assembly
# ---------------------------------------------------------------------------


def make_fact_id(doc_id: str, chunk_id: str, raw: RawFact) -> str:
    parts = [
        doc_id,
        chunk_id,
        canonical_subject(raw.subject),
        canonical_predicate(raw.predicate),
        clean_text(raw.value_raw).lower(),
        clean_text(raw.period_label_raw).lower(),
        ",".join(sorted(normalize_scope_tags(raw.scope_tags))),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def build_fact(
    raw: RawFact,
    chunk: Chunk,
    document: ParsedDocument | None,
    page_offsets: Sequence[tuple[int, int, int, str | None]],
    *,
    doc_id: str,
    source_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one raw fact and verify its evidence. Returns the two rows."""
    fact_id = make_fact_id(doc_id, chunk.chunk_id, raw)
    evidence_id = hashlib.sha256(f"{fact_id}\x00{raw.evidence_quote}".encode("utf-8")).hexdigest()[:32]

    match = verify_quote(raw.evidence_quote, chunk.text)
    page_index: int | None = chunk.page_index_start
    page_label: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: list[list[float]] = []

    if match.found:
        char_start = chunk.char_start + (match.start or 0)
        char_end = chunk.char_start + (match.end or 0)
        resolved = resolve_page(page_offsets, char_start)
        if resolved is not None:
            page_index, page_label = resolved
        if source_path and page_index is not None:
            from app.parsing import find_bboxes

            bbox = find_bboxes(source_path, page_index, raw.evidence_quote)
    else:
        for start, end, index, label in page_offsets:
            if index == chunk.page_index_start:
                page_label = label
                break

    value = normalize_value(raw.value_raw, raw.unit_raw, raw.value_type)
    period = parse_period(raw.period_label_raw) if raw.period_label_raw else None
    period_start = period.start if period else None
    period_end = period.end if period else None
    period_basis = period.basis if period else "unknown"
    if period_start is None and raw.period_start:
        # Fall back to the model's structured guess only when parsing found nothing.
        guess = parse_period(raw.period_start)
        period_start = guess.start
        period_end = parse_period(raw.period_end).end if raw.period_end else guess.end
        period_basis = raw.period_basis or "unknown"

    qualifiers: dict[str, Any] = {q.key: q.value for q in raw.qualifiers if q.key}
    qualifiers.update({k: v for k, v in value.qualifiers.items()})
    if raw.extraction_note:
        qualifiers["extraction_note"] = raw.extraction_note

    fact = {
        "fact_id": fact_id,
        "doc_id": doc_id,
        "chunk_id": chunk.chunk_id,
        "subject_raw": clean_text(raw.subject),
        "subject_canonical": canonical_subject(raw.subject),
        "entity_id": None,
        "predicate": clean_text(raw.predicate),
        "predicate_canonical": canonical_predicate(raw.predicate),
        "value_raw": clean_text(raw.value_raw),
        "value_type": raw.value_type,
        "value_num": value.value_num,
        "value_text": value.value_text,
        "unit_raw": clean_text(raw.unit_raw) or None,
        "unit_canonical": value.unit_canonical,
        "unit_dimension": value.unit_dimension,
        "magnitude": value.magnitude,
        "magnitude_label": value.magnitude_label,
        "currency": value.currency,
        "period_start": period_start,
        "period_end": period_end,
        "period_label_raw": clean_text(raw.period_label_raw) or None,
        "period_basis": period_basis,
        "scope_tags": normalize_scope_tags(raw.scope_tags),
        "qualifiers": qualifiers,
        "confidence": float(raw.confidence),
        "evidence_id": evidence_id,
        "extractor_version": config.EXTRACTOR_VERSION,
    }
    evidence = {
        "evidence_id": evidence_id,
        "fact_id": fact_id,
        "doc_id": doc_id,
        "page_index": page_index,
        "page_label": page_label,
        "quote": raw.evidence_quote,
        "char_start": char_start,
        "char_end": char_end,
        "bbox": bbox,
        "verified": match.found,
        "match_score": match.score,
        "match_method": match.method,
    }
    return fact, evidence


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def extract_chunk(
    chunk: Chunk,
    *,
    context: str,
    vocabulary: str,
    page_label: str | None = None,
    recorder: Any | None = None,
) -> tuple[list[RawFact], bool]:
    """One structured call for one chunk. Returns the facts and a cache flag."""
    prompt = build_extraction_prompt(chunk, context=context, vocabulary=vocabulary, page_label=page_label)
    result = llm.generate_structured(
        stage="extract",
        prompt=prompt,
        schema=RawFactList,
        model=config.EXTRACTION_MODEL,
        prompt_version=config.EXTRACTION_PROMPT_VERSION,
        cache_content_hash=llm.content_hash(chunk.chunk_id, chunk.text),
        thinking=config.EXTRACTION_THINKING,
        recorder=recorder,
    )
    if not result.ok:
        return [], result.cached
    facts = list(result.parsed.facts)[: config.MAX_FACTS_PER_CHUNK]
    return facts, result.cached


def infer_document_meta(
    document: ParsedDocument, *, recorder: Any | None = None
) -> RawDocumentMeta | None:
    """Ask for title, publisher, type, and dates from the opening pages."""
    pages = document.pages[: config.DOCMETA_PAGES]
    body = "\n\n".join(page.text[:4000] for page in pages)
    if not body.strip():
        return None
    result = llm.generate_structured(
        stage="docmeta",
        prompt=DOCMETA_INSTRUCTIONS.format(pages=body),
        schema=RawDocumentMeta,
        model=config.DOCMETA_MODEL,
        prompt_version=config.DOCMETA_PROMPT_VERSION,
        cache_content_hash=llm.content_hash(document.doc_id, "docmeta"),
        thinking=config.EXTRACTION_THINKING,
        recorder=recorder,
    )
    return result.parsed if result.ok else None


def current_vocabulary() -> str:
    return registry.render_vocabulary_hint()
