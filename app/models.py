"""Pydantic v2 models. These are the single source of truth for every shape.

Three families live here:

*   ``Raw*`` models are exactly what Gemini is asked to return. They are kept
    deliberately flat and free of unions because structured output schemas do
    not support free form objects.
*   Storage models mirror the DuckDB tables one to one.
*   API models are the read shapes the routes and templates consume.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ValueType = Literal["numeric", "string", "boolean", "date"]
PeriodBasis = Literal["fiscal_in", "calendar", "point_in_time", "unknown"]
ChunkKind = Literal["prose", "table", "mixed"]
DecidedBy = Literal["rule", "llm"]
Relation = Literal[
    "corroborates",
    "contradicts",
    "reconciled_by_period",
    "reconciled_by_scope",
    "reconciled_by_unit",
    "reconciled_by_vintage",
    "refines",
    "unrelated",
]

RELATIONS: tuple[str, ...] = (
    "corroborates",
    "contradicts",
    "reconciled_by_period",
    "reconciled_by_scope",
    "reconciled_by_unit",
    "reconciled_by_vintage",
    "refines",
    "unrelated",
)

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# What the extraction model is asked to return
# ---------------------------------------------------------------------------


class RawQualifier(BaseModel):
    """A free form key and value pair. A list of these stands in for an object."""

    model_config = ConfigDict(extra="ignore")

    key: str
    value: str


class RawFact(BaseModel):
    """One extracted assertion, exactly as the extraction model returns it."""

    model_config = ConfigDict(extra="ignore")

    subject: str = Field(description="The named thing the document makes the claim about.")
    predicate: str = Field(description="snake_case property being asserted about the subject.")
    value_raw: str = Field(description="The value exactly as written in the document.")
    value_type: ValueType = "numeric"
    unit_raw: str = Field(default="", description="Unit as written, empty when there is none.")
    period_label_raw: str = Field(default="", description="Period exactly as the document writes it.")
    period_start: str = Field(default="", description="Best guess ISO start date, or empty.")
    period_end: str = Field(default="", description="Best guess ISO end date, or empty.")
    period_basis: PeriodBasis = "unknown"
    scope_tags: list[str] = Field(default_factory=list)
    qualifiers: list[RawQualifier] = Field(default_factory=list)
    evidence_quote: str = Field(description="Verbatim contiguous substring of the source chunk.")
    confidence: Confidence = 0.5
    extraction_note: str = ""


class RawFactList(BaseModel):
    model_config = ConfigDict(extra="ignore")

    facts: list[RawFact] = Field(default_factory=list)


class RawDocumentMeta(BaseModel):
    """Document level metadata inferred from the opening pages."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    publisher: str = ""
    doc_type: str = ""
    as_of_date: str = Field(default="", description="The reporting date the document speaks as of.")
    published_date: str = ""
    primary_subject: str = Field(default="", description="Main entity the document is about.")


class RawAdjudication(BaseModel):
    """What the adjudicator returns for one residual pair."""

    model_config = ConfigDict(extra="ignore")

    relation: Relation
    confidence: Confidence = 0.5
    explanation: str = ""


# ---------------------------------------------------------------------------
# Storage models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    doc_id: str
    filename: str
    collection: str
    title: str | None = None
    publisher: str | None = None
    doc_type: str | None = None
    as_of_date: date | None = None
    published_date: date | None = None
    n_pages: int = 0
    ingested_at: datetime | None = None
    parser_version: str = ""
    source_path: str | None = None
    status: str = "pending"


class Page(BaseModel):
    doc_id: str
    page_index: int
    page_label: str | None = None
    text: str
    char_start: int
    char_end: int


class ChunkRow(BaseModel):
    chunk_id: str
    doc_id: str
    page_index_start: int
    page_index_end: int
    text: str
    char_start: int
    char_end: int
    kind: ChunkKind


class Fact(BaseModel):
    fact_id: str
    doc_id: str
    chunk_id: str
    subject_raw: str
    subject_canonical: str
    entity_id: str | None = None
    predicate: str
    predicate_canonical: str
    value_raw: str
    value_type: ValueType
    value_num: float | None = None
    value_text: str | None = None
    unit_raw: str | None = None
    unit_canonical: str | None = None
    unit_dimension: str | None = None
    magnitude: float | None = None
    magnitude_label: str | None = None
    currency: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    period_label_raw: str | None = None
    period_basis: PeriodBasis = "unknown"
    scope_tags: list[str] = Field(default_factory=list)
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    evidence_id: str | None = None
    extractor_version: str = ""
    created_at: datetime | None = None


class Evidence(BaseModel):
    evidence_id: str
    fact_id: str
    doc_id: str
    page_index: int | None = None
    page_label: str | None = None
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    bbox: list[list[float]] = Field(default_factory=list)
    verified: bool = False
    match_score: float = 0.0
    match_method: str = "none"


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str = "unknown"
    aliases: list[str] = Field(default_factory=list)


class PredicateEntry(BaseModel):
    predicate_canonical: str
    description: str = ""
    expected_value_type: str = "numeric"
    expected_unit_dimension: str | None = None
    example_fact_id: str | None = None
    first_seen_doc_id: str | None = None
    n_facts: int = 0
    created_at: datetime | None = None


class RuleCheck(BaseModel):
    """One step of the deterministic chain. The list of these is the explanation."""

    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    detail: str = ""


class Link(BaseModel):
    link_id: str
    fact_a: str
    fact_b: str
    relation: Relation
    confidence: float = 0.5
    decided_by: DecidedBy = "rule"
    rule_trace: list[RuleCheck] = Field(default_factory=list)
    explanation: str = ""
    numeric_delta: float | None = None
    relative_delta: float | None = None
    created_at: datetime | None = None


class Run(BaseModel):
    run_id: str
    doc_id: str | None = None
    stage: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_items: int = 0
    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API read models
# ---------------------------------------------------------------------------


class FactView(Fact):
    """A fact with its evidence and source document attached, for the UI."""

    evidence: Evidence | None = None
    document_title: str | None = None
    filename: str | None = None
    collection: str | None = None


class LinkView(Link):
    fact_a_view: FactView | None = None
    fact_b_view: FactView | None = None


class FunnelStats(BaseModel):
    blocked_pairs: int = 0
    embedded_facts: int = 0
    retrieved_pairs: int = 0
    deduped_pairs: int = 0
    rule_decided: int = 0
    llm_adjudicated: int = 0


class Stats(BaseModel):
    n_documents: int = 0
    n_pages: int = 0
    n_chunks: int = 0
    n_facts: int = 0
    n_quarantined: int = 0
    verification_rate: float = 0.0
    n_entities: int = 0
    n_predicates: int = 0
    n_links: int = 0
    links_by_relation: dict[str, int] = Field(default_factory=dict)
    funnel: FunnelStats = Field(default_factory=FunnelStats)
    n_llm_calls: int = 0
    n_cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds_by_stage: dict[str, float] = Field(default_factory=dict)
