"""Storage tests. Every test gets its own DuckDB file, no network involved."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A freshly initialised database bound to a temporary file."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "traces.jsonl"))
    import config

    importlib.reload(config)
    from app import db as db_module

    importlib.reload(db_module)
    db_module.close()
    yield db_module
    db_module.close()


def _doc(db, doc_id="doc1"):
    db.upsert_document(
        {"doc_id": doc_id, "filename": "f.pdf", "collection": "c", "n_pages": 2, "title": "T"}
    )
    return doc_id


def _fact(db, fact_id, doc_id, verified=True, predicate="revenue"):
    db.insert_facts(
        [
            {
                "fact_id": fact_id,
                "doc_id": doc_id,
                "chunk_id": "ch1",
                "subject_raw": "Delhivery",
                "subject_canonical": "delhivery",
                "predicate": predicate,
                "predicate_canonical": predicate,
                "value_raw": "8,142",
                "value_type": "numeric",
                "value_num": 8142.0,
                "scope_tags": ["reported"],
                "qualifiers": {"note": "x"},
                "evidence_id": f"ev-{fact_id}",
            }
        ]
    )
    db.insert_evidence(
        [
            {
                "evidence_id": f"ev-{fact_id}",
                "fact_id": fact_id,
                "doc_id": doc_id,
                "page_index": 4,
                "page_label": "5",
                "quote": "revenue of 8,142",
                "char_start": 10,
                "char_end": 26,
                "verified": verified,
                "match_score": 100.0 if verified else 42.0,
            }
        ]
    )


def test_tables_and_views_exist(store):
    names = {r["name"] for r in store.query("SELECT table_name AS name FROM duckdb_tables()")}
    assert {"documents", "pages", "chunks", "facts", "evidence", "entities",
            "predicate_registry", "fact_embeddings", "links", "runs", "llm_cache"} <= names
    views = {r["name"] for r in store.query("SELECT view_name AS name FROM duckdb_views() WHERE NOT internal")}
    assert {"active_facts", "quarantine"} <= views


def test_reingest_is_idempotent(store):
    doc_id = _doc(store)
    for _ in range(3):
        _doc(store)
        _fact(store, "f1", doc_id)
    assert store.scalar("SELECT COUNT(*) FROM documents") == 1
    assert store.scalar("SELECT COUNT(*) FROM facts") == 1
    assert store.scalar("SELECT COUNT(*) FROM evidence") == 1


def test_quarantine_and_active_split_on_verification(store):
    doc_id = _doc(store)
    _fact(store, "good", doc_id, verified=True)
    _fact(store, "bad", doc_id, verified=False)
    assert [r["fact_id"] for r in store.query("SELECT fact_id FROM active_facts")] == ["good"]
    assert [r["fact_id"] for r in store.query("SELECT fact_id FROM quarantine")] == ["bad"]


def test_json_columns_round_trip(store):
    doc_id = _doc(store)
    _fact(store, "f1", doc_id)
    row = store.query("SELECT scope_tags, qualifiers FROM facts")[0]
    assert store.loads(row["scope_tags"], []) == ["reported"]
    assert store.loads(row["qualifiers"], {}) == {"note": "x"}


def test_predicate_registry_appends_then_counts(store):
    store.bump_predicate({"predicate_canonical": "revenue", "first_seen_doc_id": "d"})
    store.bump_predicate({"predicate_canonical": "revenue"})
    store.bump_predicate({"predicate_canonical": "ebitda"})
    rows = {r["predicate_canonical"]: r["n_facts"] for r in store.top_predicates(10)}
    assert rows == {"revenue": 2, "ebitda": 1}
    assert store.top_predicates(1)[0]["predicate_canonical"] == "revenue"


def test_llm_cache_key_covers_prompt_and_model(store):
    a = store.cache_key("h", "p1", "m1", "s1")
    assert a != store.cache_key("h", "p2", "m1", "s1")
    assert a != store.cache_key("h", "p1", "m2", "s1")
    assert store.get_llm_cache(a) is None
    store.put_llm_cache(a, "h", "p1", "m1", "s1", '{"ok": true}', 5, 6)
    assert store.get_llm_cache(a) == '{"ok": true}'


def test_delete_document_facts_clears_derived_rows(store):
    doc_id = _doc(store)
    _fact(store, "f1", doc_id)
    store.insert_links([{"link_id": "l1", "fact_a": "f1", "fact_b": "f1", "relation": "corroborates"}])
    store.delete_document_facts(doc_id)
    assert store.scalar("SELECT COUNT(*) FROM facts") == 0
    assert store.scalar("SELECT COUNT(*) FROM evidence") == 0
    assert store.scalar("SELECT COUNT(*) FROM links") == 0
    assert store.scalar("SELECT COUNT(*) FROM documents") == 1


def test_run_recorder_writes_a_row_and_a_trace(store, tmp_path):
    from app import tracing

    importlib.reload(tracing)
    with tracing.RunRecorder("extract", "doc1") as run:
        run.n_items = 7
        run.count_llm(cached=False, tokens_in=11, tokens_out=13)
        run.count_llm(cached=True)
        run.add_error("bad_quote", "did not verify", chunk_id="c1")
    row = store.query("SELECT stage, n_items, n_llm_calls, tokens_in, errors FROM runs")[0]
    assert row["stage"] == "extract" and row["n_items"] == 7
    assert row["n_llm_calls"] == 1 and row["tokens_in"] == 11
    assert store.loads(row["errors"], [])[0]["kind"] == "bad_quote"
    assert [t["type"] for t in tracing.read_traces()] == ["stage_start", "stage_end"]


def test_run_recorder_records_an_exception_then_reraises(store):
    from app import tracing

    importlib.reload(tracing)
    with pytest.raises(ValueError):
        with tracing.RunRecorder("parse", "doc1"):
            raise ValueError("boom")
    errors = store.loads(store.query("SELECT errors FROM runs")[0]["errors"], [])
    assert errors and errors[0]["kind"] == "exception"
