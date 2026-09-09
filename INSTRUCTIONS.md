# Fact Knowledge Layer

Author: Devansh Behl

> Status: sections 1 to 5, 7 and 11 are complete. Sections 6, 8, 9 and 10 are
> marked PENDING RUN and will be written from the output of
> `python scripts/ingest_all.py` over both starter collections, because they
> must quote numbers the system actually produced rather than numbers I expect
> it to produce.

## 1. What this is

Documents disagree with each other constantly, and most of the time they are
not actually wrong: one reports a quarter and the other a year, one is adjusted
and the other reported, one was published before a revision. A reader can only
tell the difference by looking at the exact sentence each number came from.

This system reads PDFs, extracts assertions that a careful reader could check
by pointing at one sentence or one table cell, and refuses to keep any
assertion whose quote it cannot find in the source text. It then compares facts
across documents and decides whether they corroborate, contradict, or only
appear to conflict because of a difference in period, scope, unit, or data
vintage, and it shows the ordered checks that produced each verdict.

## 2. Quickstart

```bash
git clone https://github.com/Mayan10/superjoin.git
cd superjoin
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put a Google AI Studio key in .env as GEMINI_API_KEY=...

python scripts/ingest_all.py          # ingests both starter collections
uvicorn app.main:app --reload         # then open http://127.0.0.1:8000
```

Two useful extras:

```bash
python scripts/demo_cases.py          # prints the four required cases from the database
pytest                                # the full suite, no network needed
```

The pipeline runs without a key as well (`LLM_ENABLED=0`). Parsing, chunking,
page labels, offsets and the whole deterministic layer work; extraction and
linking simply produce nothing, which is the expected behaviour rather than a
crash.

## 3. Architecture

```
   PDF bytes
      |
      v
 [1] parse ........ PyMuPDF text + pymupdf4llm markdown per page,
      |             retention checked, page labels read from the margins,
      |             one document buffer with exact char offsets per page
      v
 [2] chunk ........ per page, target 1800 chars, 200 overlap,
      |             markdown tables never split, slides stay whole
      v
 [3] extract ...... one Gemini structured call per non boilerplate chunk,
      |             predicate registry injected as a vocabulary hint
      v
 [4] verify ....... exact, then whitespace normalized, then fuzzy match of the
      |             quote inside its own chunk. Failures are quarantined,
      |             never silently dropped.  <-- hard gate
      v
 [5] normalize .... number, magnitude, currency, unit, dimension, period,
      |             scope tags, subject, predicate.  Pure functions.
      v
 [6] entities ..... exact, alias, fuzzy, embedding, address.  New entity last.
      |
      v
 [7] funnel ....... block by collection, entity or predicate, value type and
      |             unit dimension, then cosine top k, then dedupe
      v
 [8] rules ........ ordered chain, every check appended to rule_trace
      |                 |
      |                 +--> settled: zero tokens
      v
 [9] adjudicate ... only the residue reaches Gemini, one call per pair
      |
      v
 DuckDB (data/knowledge.duckdb) + traces (data/traces.jsonl)
      |
      v
 FastAPI JSON API + four server rendered pages
```

**Parse.** Each page is read twice: `page.get_text("text")` for a faithful
character dump, and `pymupdf4llm` markdown so tables survive as pipe tables.
The markdown is only trusted when it retains at least 95 percent of the plain
text's alphanumeric characters, because its layout analysis drops content on
graphics heavy pages. On the pages where it fails the canonical text becomes
the plain text with genuine tables appended, filtered by a quality heuristic
that rejects the pseudo tables table finders hallucinate over charts. Every
page's canonical text goes into one document buffer with recorded start and end
offsets, so any offset produced later resolves to exactly one page. Printed page
labels are read from the top and bottom tenth of the page and stored separately
from the zero based PDF index, because in curated excerpts the two diverge.
Pages under the sparse text threshold are recorded in `runs.errors` as likely
image only. There is no OCR: the gap is reported honestly instead.

**Chunk.** Chunks are pure slices of the buffer, so offsets survive unchanged.
Table blocks are atomic. Documents detected as slide decks, by producer
metadata or by median page length, keep one chunk per slide.

**Extract.** One structured Gemini call per chunk, with a schema derived from
the Pydantic models. Boilerplate, tables of contents and disclaimers are
skipped by a cheap structural filter before any call is made. The prompt
injects the top predicates already in the registry so the vocabulary converges
instead of fragmenting.

**Verify.** The gate. A quote is located by exact match, then whitespace and
case insensitive match, then `rapidfuzz.partial_ratio_alignment` above a
threshold of 90. Success records real character offsets, resolves the page, and
tries `page.search_for` for a bounding box. Failure sets `verified = false`,
keeps the fact out of the active set, and leaves it in the `quarantine` view.

**Normalize.** Pure functions, no model, fully unit tested. Magnitude is folded
into `value_num` while the original stays in `magnitude` and `value_raw`.
Conversion to a dimension's base unit happens only at comparison time.

**Entities.** A ladder from cheapest to most expensive, creating a new entity
only when all four earlier stages fail. Every raw form that resolves onto an
entity is kept as an alias, and near threshold decisions are logged either way.

**Funnel and rules.** Described in section 5.

**Adjudicate.** Only pairs no rule could settle, with both quotes, both
normalized records, the trace so far, and both document titles and dates.

## 4. Data model

One DuckDB file. Every id is a content hash, so re-ingesting the same bytes
rewrites the same rows.

**documents**: `doc_id` sha256 of the file bytes; `filename`; `collection`;
`title`, `publisher`, `doc_type`, `as_of_date`, `published_date` inferred by
one call over the opening pages; `n_pages`; `ingested_at`; `parser_version`;
`source_path`; `status`; `buffer`, the concatenated canonical text that makes
every offset resolvable.

**pages**: `doc_id`; `page_index` zero based; `page_label` printed label or
null; `text` the canonical text used for all offsets; `text_plain` the raw
PyMuPDF dump; `char_start`, `char_end` into the buffer; `text_source` which of
the two parse paths produced the canonical text.

**chunks**: `chunk_id` hash of doc id plus normalized text; `doc_id`;
`page_index_start`, `page_index_end`; `text`; `char_start`, `char_end`; `kind`
one of prose, table, mixed.

**facts**: `fact_id`; `doc_id`; `chunk_id`; `subject_raw` and
`subject_canonical`; `entity_id`; `predicate` and `predicate_canonical`;
`value_raw` exactly as printed; `value_type`; `value_num` with magnitude
folded in; `value_text`; `unit_raw`, `unit_canonical`, `unit_dimension`;
`magnitude` and `magnitude_label` kept for explanation; `currency`;
`period_start`, `period_end`, `period_label_raw`, `period_basis`; `scope_tags`
JSON array; `qualifiers` JSON object; `confidence`; `evidence_id`;
`extractor_version`; `created_at`.

**evidence**: `evidence_id`; `fact_id`; `doc_id`; `page_index`; `page_label`;
`quote` verbatim; `char_start`, `char_end`; `bbox` best effort JSON;
`verified`; `match_score`; `match_method`.

**entities**: `entity_id`; `canonical_name`; `entity_type`; `aliases` JSON
array; `embedding`.

**predicate_registry**: `predicate_canonical`; `description`;
`expected_value_type`; `expected_unit_dimension`; `example_fact_id`;
`first_seen_doc_id`; `n_facts`; `created_at`.

**fact_embeddings**: `fact_id`; `canonical_string`; `embedding`.

**links**: `link_id`; `fact_a`, `fact_b` ordered deterministically; `relation`;
`confidence`; `decided_by`; `rule_trace` JSON array of ordered checks;
`explanation`; `numeric_delta`; `relative_delta`; `created_at`.

**runs**: `run_id`; `doc_id`; `stage`; `started_at`, `finished_at`; `n_items`;
`n_llm_calls`; `tokens_in`, `tokens_out`; `errors` JSON; `notes` JSON.

**llm_cache** and **embedding_cache**: responses keyed by content hash, prompt
version, model and schema version.

Two views: `active_facts` is facts whose evidence verified, `quarantine` is
those whose evidence did not.

### Why this shape

A fixed envelope carries the things every comparison needs: subject, predicate,
value, unit, dimension, period, scope. Those columns are typed, indexed, and
the rule engine can rely on them. Everything else that a document might qualify
a number with goes into free form `qualifiers`, which costs nothing and loses
nothing. Between the two sits `predicate_registry`, which turns the open ended
part of the schema into a converging one: predicates are appended as they are
coined, and the most frequent are fed back into later extraction prompts. The
result evolves with the corpus without becoming a free-for-all, because the
model is nudged towards reusing what already exists but never prevented from
naming something new.

## 5. How facts are compared

### The funnel

Naive all pairs comparison is quadratic and mostly wasted: almost no two facts
in a corpus are about the same thing. Each stage narrows the set, and only the
last one costs tokens.

1. **Block.** Same collection, then same entity or same canonical predicate,
   then same value type and compatible unit dimension, then never two facts
   from the same chunk. Blocking runs inside the SQL join, so similarity is
   only ever computed for pairs that already passed. Fuzzy subject matching is
   realised through entity resolution rather than repeated here.
2. **Embed.** One vector per fact over
   `{subject_canonical} | {predicate_canonical} | {period_label_raw} | {scope_tags}`.
   The value is deliberately excluded, otherwise two facts that disagree would
   never become neighbours and no contradiction could be found.
3. **Retrieve.** `array_cosine_similarity` in DuckDB, top 12 per fact above
   0.80.
4. **Dedupe.** Pairs ordered by fact id so a pair is evaluated once, and pairs
   already carrying a link are skipped, which is what makes incremental ingest
   cheap.

Real surviving counts: PENDING RUN.

### The rule chain

Every check appends `{name, inputs, outcome, detail}` to `rule_trace`. The
trace is the explanation.

1. **Value type and dimension.** Different types, or two different dimensions,
   gives `unrelated`.
2. **Currency.** Two currencies with no stated rate gives `unrelated`.
   Currencies are never converted. If one document states a rate, the pair goes
   to the adjudicator, which may use the rate and must cite it.
3. **Unit conversion.** Convert within the dimension. If a conversion fired and
   the values now agree, the relation is `reconciled_by_unit` and the
   explanation names the conversion.
4. **Periods.** Disjoint gives `reconciled_by_period`. Nested gives `refines`,
   naming the containment, which is how a quarter inside a fiscal year avoids
   being read as a contradiction. Overlapping but unequal is flagged and the
   chain continues.
5. **Scope.** A symmetric difference containing a known scope shifting pair
   (adjusted against reported, standalone against consolidated, pro forma
   against actual, estimate against provisional against revised, segment
   against total, since inception against periodic) gives
   `reconciled_by_scope`.
6. **Vintage.** Same entity, predicate and period in two documents with
   different `as_of_date` and different values gives `reconciled_by_vintage`,
   naming which document is later.
7. **Numeric agreement.** Tolerance is derived from the precision each raw
   string actually claims. A value written `1.4` claims one decimal place, so
   anything within half of that place agrees with it, scaled by the magnitude
   folded into the number. The looser of the two sides wins. This is what lets
   `1.4 Mn Tons` and `1,429 thousand tons` corroborate while `1.4` and `1.9` do
   not, and it is what makes a stated rise of 578 against an arithmetic 579
   corroborate with the rounding named in the explanation.
8. **Residue.** Same subject, predicate, period, scope and unit with a material
   numeric gap is the only thing that reaches the model.

Steps 1 to 7 are marked `decided_by = 'rule'` and cost zero tokens. Rule
decided `unrelated` pairs are counted but not stored, because they are the bulk
of the funnel and keeping them would bury the interesting links.

Example trace: PENDING RUN.

## 6. The four cases

PENDING RUN. Each case will be quoted here exactly as
`python scripts/demo_cases.py` prints it, with the evidence quotes, page
references, normalized values, rule trace and explanation, plus the command
that reproduces it.

## 7. Engineering decisions and trade-offs

**PyMuPDF and pymupdf4llm over Marker or Unstructured.** Parsing stays CPU only
and takes seconds per document rather than minutes, with no torch in the
dependency tree. What was given up is layout quality on hard pages: the
markdown renderer drops text on graphics heavy pages, which is why the parser
measures retention and falls back rather than trusting it.

**DuckDB over a graph or vector database.** One file, no service to run, and
`array_cosine_similarity` is fast enough at this scale. Given up: approximate
nearest neighbour speed at millions of facts, and graph traversal queries. At
this corpus size the blocking join dominates anyway, and blocking is what makes
retrieval cheap, not the index.

**Rules before the model.** Deterministic checks are auditable, free, and
identical every run, and the trace they produce is a better explanation than
prose. Given up: coverage of cases nobody anticipated, which is exactly why the
residue is escalated rather than forced into a rule.

**Evidence verification as a hard gate.** A fact with an unfindable quote is
worse than no fact, because it looks checkable and is not. Given up: recall,
measurably. Quarantined facts are kept and shown rather than deleted, so the
cost of the gate is visible instead of hidden.

**Content hash caching.** The cache key hashes the content being reasoned
about, not the rendered prompt, because the prompt carries a registry hint that
grows over time. Hashing the prompt would make that hint invalidate the cache
and re-ingest would cost tokens again. Given up: prompt level cache precision,
in exchange for the re-ingest guarantee being literally true.

**No graph visualization.** A graph would show that two facts are connected
while hiding why, and the why is the entire deliverable. Given up: the demo
appeal of a node diagram.

**Two files beyond the specified layout.** `app/llm.py` holds the single Gemini
client, cache, retry policy and trace path, so no call can bypass any of them.
`app/pipeline.py` holds ingest orchestration, so `main.py` stays routes only.

## 8. Limitations

PENDING RUN for the measured parts. Known already:

- No OCR. Pages that yield almost no text are recorded in `runs.errors` and
  contribute nothing. The starter corpus contains such pages.
- Chart data labels are read as text without their axis context, so a value can
  be attached to the wrong category when a chart is mislabelled.
- Currencies are never converted unless a document states a rate.
- Cross collection linking is off by default.

## 9. Extensions implemented

PENDING RUN for the measurements. The mechanisms are in place: large PDFs via
per page streaming parse and per chunk extraction, many PDFs via the blocked
funnel, an evolving schema via `predicate_registry`, and incremental ingest via
content hashed ids plus a document scoped funnel.

## 10. Demo guide

PENDING RUN.

## 11. Video script

PENDING RUN.

## Acceptance checks

| Check | Result |
| --- | --- |
| Fresh clone plus quickstart produces a working system | PENDING RUN |
| Re-ingest makes zero model calls and zero duplicate rows | PENDING RUN |
| A seventh unseen PDF adds facts and links without touching existing rows | PENDING RUN |
| Every fact in the UI has a verified quote on the stated page | PENDING RUN |
| Every link shows a rule trace or a model explanation | PENDING RUN |
| `scripts/demo_cases.py` finds all four cases with no hardcoded ids | PENDING RUN |
| The pytest suite passes without a network connection | 207 passed |
