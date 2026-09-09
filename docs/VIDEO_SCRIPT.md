# Fact Knowledge Layer — demo script (target 2:50)

Author: Devansh Behl
Read at a normal pace, roughly 150 words per minute. Word count: ~415.
Numbers marked [read from screen] are whatever the live UI shows — do not memorise them.

---

## 0:00 — 0:18 · The problem
**Screen:** two PDFs open side by side, the same metric highlighted in each with different numbers.

> "Two documents report the same metric and give you two different numbers. Most of the time neither is wrong — one is a quarter and the other a year, one is adjusted and the other reported, one was published before a revision. The only way to tell is to look at the exact sentence each number came from. This system does that automatically."

## 0:18 — 0:40 · Ingest and the evidence gate
**Screen:** terminal, `python scripts/ingest_all.py` already part-run; cut to the Documents page.

> "PDFs go through one pipeline: parse, chunk, extract, verify, normalize. Extraction is a structured model call per chunk, but nothing it returns is trusted. Every fact has to carry a verbatim quote, and that quote is matched back into the source text — exact, then normalized, then fuzzy. If it can't be found, the fact is quarantined, not deleted. That gate is the whole design."

## 0:40 — 1:10 · Facts
**Screen:** Facts page, filter by entity, expand one row.

> "Here's the fact table. Every row expands to the sentence it came from, the page it's printed on, and character offsets into the document. Subject, predicate, value, unit, period and scope are typed columns — everything else the document qualified the number with is kept alongside. Predicates aren't a fixed list: they're coined as facts are extracted, and the frequent ones are fed back into later prompts so the vocabulary converges."

## 1:10 — 1:55 · The conflict inbox
**Screen:** Conflicts page. Open one corroboration, one contradiction, one reconciled pair. Expand the rule trace each time.

> "This is the conflict inbox. Both facts side by side, both quotes, and the ordered checks that decided them.
>
> This pair agrees — different units, different magnitudes, same claim, reconciled by unit conversion.
> This one is a real contradiction: same entity, same period, same scope, and a gap the tolerance can't absorb.
> And this one only looked like a conflict — a quarter nested inside a fiscal year. The rule names the containment.
>
> Eight ordered checks: type, currency, unit, period, scope, vintage, numeric agreement. Each one costs zero tokens. Only what no rule can settle reaches the model."

## 1:55 — 2:25 · Stats
**Screen:** Stats page — funnel, links by relation, quarantine.

> "The funnel is why this scales. Blocking runs inside the SQL join, so similarity is only computed for pairs that already share an entity and a unit dimension. [read from screen] percent of links were settled by rule alone. And the quarantine is on the same page — the cost of the evidence gate is shown, not hidden."

## 2:25 — 2:50 · Close
**Screen:** re-run ingest, show zero new model calls; then `pytest` output.

> "Every id is a content hash, so re-ingesting the same bytes makes no model calls and creates no duplicate rows. The full test suite runs with no network. One DuckDB file, four pages, and every claim on screen traceable to a sentence on a page."
