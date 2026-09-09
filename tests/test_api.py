"""API and page rendering tests against a seeded temporary database.

No network: ingest is never triggered, the rows are inserted directly.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.duckdb"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "api.jsonl"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    import config

    importlib.reload(config)
    from app import db as db_module

    importlib.reload(db_module)
    db_module.close()
    _seed(db_module)
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client, db_module
    db_module.close()


def _seed(db):
    db.upsert_document(
        {"doc_id": "d1", "filename": "deck.pdf", "collection": "delhivery", "title": "Q4 FY24 deck",
         "doc_type": "earnings_presentation", "as_of_date": "2024-03-31", "n_pages": 27, "status": "done"}
    )
    db.upsert_document(
        {"doc_id": "d2", "filename": "annual.pdf", "collection": "delhivery", "title": "Annual Report FY24",
         "doc_type": "annual_report", "as_of_date": "2024-03-31", "n_pages": 100, "status": "done"}
    )
    facts = []
    evidence = []
    for fact_id, doc_id, value_raw, value_num, verified, quote in [
        ("f1", "d1", "1.4 Mn Tons", 1_400_000.0, True, "PTL freight tonnage in FY24 was 1.4 Mn Tons"),
        ("f2", "d2", "1,429", 1_429_000.0, True, "Part-truckload tonnage 1,429 thousand tonnes"),
        ("f3", "d1", "9,999", 9999.0, False, "a quote that is not in any document"),
    ]:
        facts.append(
            {
                "fact_id": fact_id, "doc_id": doc_id, "chunk_id": f"c-{fact_id}",
                "subject_raw": "Delhivery Limited", "subject_canonical": "delhivery", "entity_id": "e1",
                "predicate": "ptl freight tonnage", "predicate_canonical": "ptl_freight_tonnage",
                "value_raw": value_raw, "value_type": "numeric", "value_num": value_num,
                "unit_canonical": "tonne", "unit_dimension": "mass", "period_label_raw": "FY24",
                "period_start": "2023-04-01", "period_end": "2024-03-31", "period_basis": "fiscal_in",
                "scope_tags": ["reported"], "qualifiers": {}, "confidence": 0.9,
                "evidence_id": f"ev-{fact_id}",
            }
        )
        evidence.append(
            {
                "evidence_id": f"ev-{fact_id}", "fact_id": fact_id, "doc_id": doc_id,
                "page_index": 5, "page_label": "5", "quote": quote, "char_start": 10, "char_end": 40,
                "verified": verified, "match_score": 100.0 if verified else 40.0,
                "match_method": "exact" if verified else "fuzzy_failed",
            }
        )
    db.insert_facts(facts)
    db.insert_evidence(evidence)
    db.bump_predicate({"predicate_canonical": "ptl_freight_tonnage", "first_seen_doc_id": "d1"}, 2)
    db.upsert_entity({"entity_id": "e1", "canonical_name": "delhivery", "entity_type": "organisation",
                      "aliases": ["Delhivery Limited"]})
    db.insert_links(
        [
            {
                "link_id": "l1", "fact_a": "f1", "fact_b": "f2", "relation": "corroborates",
                "confidence": 0.9, "decided_by": "rule",
                "rule_trace": [{"name": "unit_conversion", "inputs": {"from": "tonne"}, "outcome": "not_needed",
                                "detail": ""}],
                "explanation": "The same tonnage stated two ways.", "numeric_delta": 29000.0,
                "relative_delta": 0.0203,
            }
        ]
    )
    db.record_run({"run_id": "r1", "doc_id": "d1", "stage": "link", "n_items": 1, "n_llm_calls": 3,
                   "tokens_in": 100, "tokens_out": 20,
                   "notes": {"seconds": 1.5, "blocked_pairs": 40, "candidate_pairs": 6,
                             "rule_decided": 5, "residue_for_llm": 1}})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_health_and_documents(client):
    api, _ = client
    assert api.get("/api/health").json()["status"] == "ok"
    body = api.get("/api/documents").json()
    assert len(body["documents"]) == 2
    assert body["collections"] == ["delhivery"]
    assert body["documents"][0]["n_verified_facts"] >= 1


def test_document_status_reports_stages(client):
    api, _ = client
    body = api.get("/api/documents/d1/status").json()
    assert body["status"] == "done"
    assert body["n_facts"] == 2 and body["n_verified_facts"] == 1
    assert body["stages"][0]["stage"] == "link"
    assert api.get("/api/documents/nope/status").status_code == 404


def test_facts_endpoint_hides_unverified_by_default(client):
    api, _ = client
    body = api.get("/api/facts").json()
    ids = {f["fact_id"] for f in body["facts"]}
    assert ids == {"f1", "f2"}
    assert "f3" in {f["fact_id"] for f in api.get("/api/facts?include_unverified=true").json()["facts"]}


@pytest.mark.parametrize(
    "query, expected",
    [
        ("collection=delhivery", {"f1", "f2"}),
        ("doc_id=d1", {"f1"}),
        ("predicate=ptl_freight_tonnage", {"f1", "f2"}),
        ("period=FY24", {"f1", "f2"}),
        ("q=1.4", {"f1"}),
        ("collection=nothing", set()),
    ],
)
def test_fact_filters(client, query, expected):
    api, _ = client
    body = api.get(f"/api/facts?{query}").json()
    assert {f["fact_id"] for f in body["facts"]} == expected


def test_fact_detail_includes_evidence_and_links(client):
    api, _ = client
    body = api.get("/api/facts/f1").json()
    assert body["fact"]["quote"].startswith("PTL freight tonnage")
    assert body["fact"]["page_label"] == "5"
    assert body["links"][0]["relation"] == "corroborates"
    assert body["links"][0]["counterpart"]["fact_id"] == "f2"
    assert api.get("/api/facts/missing").status_code == 404


def test_links_endpoint_filters_and_expands_both_sides(client):
    api, _ = client
    body = api.get("/api/links?relation=corroborates").json()
    assert body["total"] == 1
    link = body["links"][0]
    assert link["fact_a_view"]["value_raw"] == "1.4 Mn Tons"
    assert link["fact_b_view"]["value_raw"] == "1,429"
    assert api.get("/api/links?relation=nonsense").status_code == 400
    assert api.get("/api/links?min_confidence=0.99").json()["total"] == 0


def test_link_detail_includes_the_rule_trace(client):
    api, _ = client
    body = api.get("/api/links/l1").json()
    assert body["rule_trace"][0]["name"] == "unit_conversion"
    assert api.get("/api/links/missing").status_code == 404


def test_quarantine_lists_only_unverified_facts(client):
    api, _ = client
    body = api.get("/api/quarantine").json()
    assert body["total"] == 1 and body["facts"][0]["fact_id"] == "f3"


def test_registry_and_stats(client):
    api, _ = client
    assert api.get("/api/registry").json()["predicates"][0]["predicate_canonical"] == "ptl_freight_tonnage"
    stats = api.get("/api/stats").json()
    assert stats["n_documents"] == 2
    assert stats["n_verified_facts"] == 2 and stats["n_quarantined"] == 1
    assert stats["verification_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert stats["funnel"]["blocked_pairs"] == 40
    assert stats["links_by_relation"] == {"corroborates": 1}


def test_export_contains_every_table(client):
    api, _ = client
    body = api.get("/api/export").json()
    assert {"documents", "facts", "evidence", "entities", "links", "predicate_registry", "runs", "stats"} <= set(body)
    assert len(body["facts"]) == 3


def test_upload_rejects_non_pdf_and_empty(client):
    api, _ = client
    assert api.post("/api/documents", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert api.post("/api/documents", files={"file": ("x.pdf", b"", "application/pdf")}).status_code == 400


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/facts", "/conflicts", "/stats"])
def test_pages_render(client, path):
    api, _ = client
    response = api.get(path)
    assert response.status_code == 200
    assert "Fact Knowledge Layer" in response.text


def test_facts_page_shows_the_verbatim_quote_and_both_values(client):
    api, _ = client
    text = api.get("/facts").text
    assert "PTL freight tonnage in FY24 was 1.4 Mn Tons" in text
    assert "1.4 Mn Tons" in text and "1.4 M" in text


def test_conflict_inbox_shows_both_quotes_and_the_trace(client):
    api, _ = client
    text = api.get("/conflicts").text
    assert "corroborates" in text
    assert "PTL freight tonnage in FY24 was 1.4 Mn Tons" in text
    assert "Part-truckload tonnage 1,429 thousand tonnes" in text
    assert "unit_conversion" in text
    assert "The same tonnage stated two ways." in text


def test_stats_page_shows_the_funnel_and_quarantine(client):
    api, _ = client
    text = api.get("/stats").text
    assert "Candidate funnel" in text and "Quarantine" in text
    assert "a quote that is not in any document" in text
