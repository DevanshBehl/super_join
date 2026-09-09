"""Entity resolution tests. The embedding stage is exercised with a stub, so
this file needs no network."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def resolver(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "e.duckdb"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "t.jsonl"))
    import config

    importlib.reload(config)
    from app import db as db_module

    importlib.reload(db_module)
    db_module.close()
    from app import entities as entities_module

    importlib.reload(entities_module)
    yield entities_module.EntityResolver()
    db_module.close()


def test_exact_and_alias_matching(resolver):
    first = resolver.resolve("Delhivery Limited")
    assert first.created and first.canonical_name == "delhivery"
    second = resolver.resolve("Delhivery Limited")
    assert second.method == "exact" and second.entity_id == first.entity_id
    third = resolver.resolve("Delhivery Ltd.")
    assert third.entity_id == first.entity_id, "a legal suffix is not a different company"
    fourth = resolver.resolve("The Delhivery")
    assert fourth.entity_id == first.entity_id, "a leading article is not a different company"


def test_accented_and_plain_spellings_resolve_together(resolver):
    base = resolver.resolve("Reserve Bank of India")
    again = resolver.resolve("Reserve Bank of Indía")
    assert again.entity_id == base.entity_id and again.method == "exact"


def test_fuzzy_merge_keeps_the_raw_form_as_an_alias(resolver):
    base = resolver.resolve("Delhivery Limited")
    again = resolver.resolve("Delhiverry")
    assert again.entity_id == base.entity_id and again.method == "fuzzy"
    assert resolver.entities[base.entity_id].aliases, "the raw form must be retained"


def test_clearly_different_subjects_stay_separate(resolver):
    a = resolver.resolve("Express Parcel")
    b = resolver.resolve("Supply Chain Services")
    assert a.entity_id != b.entity_id


def test_a_near_threshold_decision_is_recorded_for_review(resolver):
    resolver.resolve("Economic Survey 2024-25")
    resolver.resolve("Economic Survey 2023-24")
    assert resolver.close_calls, "a borderline pair must be logged either way"
    entry = resolver.close_calls[0]
    assert entry["decision"] in {"merged", "kept_separate"} and entry["score"] > 0


def test_embedding_stage_only_runs_when_string_matching_fails(resolver):
    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        # Two orthogonal vectors: nothing will ever be similar enough to merge.
        return [[1.0 if "gross" in t else 0.0, 0.0 if "gross" in t else 1.0] for t in texts]

    resolver.embed_fn = fake_embed
    first = resolver.resolve("gross merchandise value")
    second = resolver.resolve("total shipment count")
    assert first.entity_id != second.entity_id
    assert calls, "the embedding function should have been consulted"
    repeat = resolver.resolve("gross merchandise value")
    assert repeat.method == "exact"


def test_embedding_merge_when_vectors_agree(resolver):
    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    resolver.embed_fn = fake_embed
    first = resolver.resolve("current account deficit")
    second = resolver.resolve("shortfall on the external current account")
    assert second.entity_id == first.entity_id
    assert second.method == "embedding"


def test_addresses_written_differently_resolve_together(resolver):
    a = resolver.resolve("Plot 5, MG Road, Bengaluru - 560001")
    b = resolver.resolve("Plot No 5, M G Rd., Bengaluru 560001")
    assert b.entity_id == a.entity_id
    assert b.method in {"address", "fuzzy", "alias", "exact"}


def test_addresses_with_different_postal_codes_stay_separate(resolver):
    a = resolver.resolve("Plot 5, MG Road, Bengaluru - 560001")
    b = resolver.resolve("Plot 5, MG Road, Bengaluru - 110037")
    assert a.entity_id != b.entity_id


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("Delhivery Limited", "organisation"),
        ("N24-N34, Air Cargo Logistics Centre, New Delhi 110037", "address"),
        ("Mr. Sahil Barua", "person"),
        ("real gdp growth", "concept"),
    ],
)
def test_entity_type_guessing(resolver, subject, expected):
    from app.entities import guess_entity_type

    assert guess_entity_type(subject) == expected


def test_flush_writes_only_changed_rows(resolver):
    from app import db

    resolver.resolve("Delhivery Limited")
    assert resolver.flush() == 1
    assert resolver.flush() == 0
    assert db.scalar("SELECT COUNT(*) FROM entities") == 1


def test_reload_from_the_database_preserves_matching(resolver):
    from app import entities as entities_module

    first = resolver.resolve("International Monetary Fund")
    resolver.flush()
    reloaded = entities_module.EntityResolver().load()
    again = reloaded.resolve("International Monetary Fund")
    assert again.entity_id == first.entity_id and again.method == "exact"
