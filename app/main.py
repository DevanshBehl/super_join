"""FastAPI application: JSON API plus four server rendered pages.

Uploads run the pipeline in a background task and the UI polls the status
endpoint, so a large PDF does not hold a request open. Everything else is a
straight read from DuckDB.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from app import db, pipeline, registry
from app.models import RELATIONS
from app.parsing import compute_doc_id
from app.tracing import trace_event

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Open the database and record a startup line before serving."""
    config.ensure_dirs()
    db.connect()
    trace_event({"type": "startup", "db_path": str(config.DB_PATH)})
    yield


app = FastAPI(
    title="Fact Knowledge Layer",
    description="Extracts grounded facts from PDFs and reconciles them across documents.",
    version="1.0.0",
    lifespan=lifespan,
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Progress for documents currently being ingested, so the UI can show a stage.
_PROGRESS: dict[str, str] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(content=_json_safe(payload))


def _page_size(requested: int | None) -> int:
    if not requested:
        return config.API_PAGE_SIZE
    return max(1, min(requested, config.API_MAX_PAGE_SIZE))


# ---------------------------------------------------------------------------
# Documents and ingest
# ---------------------------------------------------------------------------


def _run_ingest(data: bytes, filename: str, collection: str, doc_id: str, stored_path: str) -> None:
    def progress(stage: str) -> None:
        _PROGRESS[doc_id] = stage

    try:
        pipeline.ingest_bytes(data, filename, collection, source_path=stored_path, progress=progress)
    except Exception as exc:  # noqa: BLE001 - a failed ingest must be visible, not silent
        _PROGRESS[doc_id] = f"failed: {type(exc).__name__}"
        db.set_document_status(doc_id, "failed")
        trace_event({"type": "ingest_failed", "doc_id": doc_id, "error": str(exc)[:500]})
    else:
        _PROGRESS.pop(doc_id, None)


@app.post("/api/documents")
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile,
    collection: str = Form(default=""),
) -> JSONResponse:
    """Accept a PDF, store it, and start the pipeline in the background."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file is larger than the configured limit")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="only PDF files are accepted")

    doc_id = compute_doc_id(data)
    name = (collection or "").strip() or config.DEFAULT_COLLECTION
    config.ensure_dirs()
    stored = config.UPLOAD_DIR / f"{doc_id[:16]}-{Path(file.filename).name}"
    stored.write_bytes(data)

    already = db.document_exists(doc_id)
    db.upsert_document(
        {
            "doc_id": doc_id, "filename": file.filename, "collection": name,
            "n_pages": 0, "status": "queued", "source_path": str(stored),
        }
    ) if not already else db.set_document_status(doc_id, "queued")
    _PROGRESS[doc_id] = "queued"
    background.add_task(_run_ingest, data, file.filename, name, doc_id, str(stored))
    return ok({"doc_id": doc_id, "collection": name, "already_ingested": already, "status": "queued"})


@app.get("/api/documents")
def api_documents() -> JSONResponse:
    documents = db.list_documents()
    for document in documents:
        document["progress_stage"] = _PROGRESS.get(document["doc_id"])
    return ok({"documents": documents, "collections": db.collections()})


@app.get("/api/documents/{doc_id}/status")
def api_document_status(doc_id: str) -> JSONResponse:
    document = db.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="unknown document")
    runs = db.document_runs(doc_id)
    return ok(
        {
            "doc_id": doc_id,
            "status": document.get("status"),
            "progress_stage": _PROGRESS.get(doc_id),
            "stages": [
                {
                    "stage": run["stage"], "n_items": run["n_items"], "n_llm_calls": run["n_llm_calls"],
                    "seconds": run["notes"].get("seconds"), "n_errors": len(run["errors"]),
                }
                for run in runs
            ],
            "n_facts": db.scalar("SELECT COUNT(*) FROM facts WHERE doc_id = ?", [doc_id], 0),
            "n_verified_facts": db.scalar("SELECT COUNT(*) FROM active_facts WHERE doc_id = ?", [doc_id], 0),
        }
    )


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@app.get("/api/facts")
def api_facts(
    doc_id: str | None = None,
    collection: str | None = None,
    entity_id: str | None = None,
    predicate: str | None = None,
    period: str | None = None,
    q: str | None = None,
    include_unverified: bool = False,
    page: int = Query(default=1, ge=1),
    page_size: int | None = None,
) -> JSONResponse:
    size = _page_size(page_size)
    rows, total = db.search_facts(
        doc_id=doc_id, collection=collection, entity_id=entity_id, predicate=predicate,
        period=period, q=q, verified=None if include_unverified else True,
        limit=size, offset=(page - 1) * size,
    )
    return ok({"facts": rows, "total": total, "page": page, "page_size": size})


@app.get("/api/facts/{fact_id}")
def api_fact(fact_id: str) -> JSONResponse:
    fact = db.get_fact(fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail="unknown fact")
    links = db.links_for_fact(fact_id)
    others = [l["fact_b"] if l["fact_a"] == fact_id else l["fact_a"] for l in links]
    counterparts = db.facts_by_ids(others)
    for item in links:
        other = item["fact_b"] if item["fact_a"] == fact_id else item["fact_a"]
        item["counterpart"] = counterparts.get(other)
    return ok({"fact": fact, "links": links})


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


@app.get("/api/links")
def api_links(
    relation: str | None = None,
    collection: str | None = None,
    doc_id: str | None = None,
    decided_by: str | None = None,
    min_confidence: float = 0.0,
    page: int = Query(default=1, ge=1),
    page_size: int | None = None,
) -> JSONResponse:
    if relation and relation not in RELATIONS:
        raise HTTPException(status_code=400, detail=f"relation must be one of {list(RELATIONS)}")
    size = _page_size(page_size)
    rows, total = db.search_links(
        relation=relation, collection=collection, doc_id=doc_id, decided_by=decided_by,
        min_confidence=min_confidence, limit=size, offset=(page - 1) * size,
    )
    facts = db.facts_by_ids([r["fact_a"] for r in rows] + [r["fact_b"] for r in rows])
    for row in rows:
        row["fact_a_view"] = facts.get(row["fact_a"])
        row["fact_b_view"] = facts.get(row["fact_b"])
    return ok({"links": rows, "total": total, "page": page, "page_size": size})


@app.get("/api/links/{link_id}")
def api_link(link_id: str) -> JSONResponse:
    link = db.get_link(link_id)
    if link is None:
        raise HTTPException(status_code=404, detail="unknown link")
    facts = db.facts_by_ids([link["fact_a"], link["fact_b"]])
    link["fact_a_view"] = facts.get(link["fact_a"])
    link["fact_b_view"] = facts.get(link["fact_b"])
    return ok(link)


# ---------------------------------------------------------------------------
# Quarantine, registry, stats, export
# ---------------------------------------------------------------------------


@app.get("/api/quarantine")
def api_quarantine(page: int = Query(default=1, ge=1), page_size: int | None = None) -> JSONResponse:
    size = _page_size(page_size)
    rows, total = db.quarantined_facts(limit=size, offset=(page - 1) * size)
    return ok({"facts": rows, "total": total, "page": page, "page_size": size})


@app.get("/api/registry")
def api_registry() -> JSONResponse:
    return ok({"predicates": registry.snapshot()})


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    return ok(db.stats())


@app.get("/api/export")
def api_export() -> JSONResponse:
    return ok(db.export_all())


@app.get("/api/health")
def api_health() -> JSONResponse:
    from app import llm

    return ok({"status": "ok", "llm_available": llm.llm_available(), "db_path": str(config.DB_PATH)})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {"collections": db.collections(), **context})


@app.get("/", response_class=HTMLResponse)
def page_documents(request: Request) -> HTMLResponse:
    documents = db.list_documents()
    for document in documents:
        document["progress_stage"] = _PROGRESS.get(document["doc_id"])
    return _render(request, "documents.html", {"documents": documents, "active": "documents"})


@app.get("/facts", response_class=HTMLResponse)
def page_facts(
    request: Request,
    collection: str | None = None,
    doc_id: str | None = None,
    predicate: str | None = None,
    period: str | None = None,
    q: str | None = None,
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    size = config.API_PAGE_SIZE
    rows, total = db.search_facts(
        doc_id=doc_id, collection=collection, predicate=predicate, period=period, q=q,
        limit=size, offset=(page - 1) * size,
    )
    return _render(
        request,
        "facts.html",
        {
            "facts": rows, "total": total, "page": page, "page_size": size, "active": "facts",
            "filters": {"collection": collection or "", "doc_id": doc_id or "",
                        "predicate": predicate or "", "period": period or "", "q": q or ""},
            "documents": db.list_documents(),
        },
    )


@app.get("/conflicts", response_class=HTMLResponse)
def page_conflicts(
    request: Request,
    relation: str | None = None,
    collection: str | None = None,
    decided_by: str | None = None,
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    size = 25
    rows, total = db.search_links(
        relation=relation, collection=collection, decided_by=decided_by,
        limit=size, offset=(page - 1) * size,
    )
    facts = db.facts_by_ids([r["fact_a"] for r in rows] + [r["fact_b"] for r in rows])
    for row in rows:
        row["fact_a_view"] = facts.get(row["fact_a"])
        row["fact_b_view"] = facts.get(row["fact_b"])
    counts = db.stats()["links_by_relation"]
    return _render(
        request,
        "conflicts.html",
        {
            "links": rows, "total": total, "page": page, "page_size": size, "active": "conflicts",
            "relations": RELATIONS, "counts": counts,
            "filters": {"relation": relation or "", "collection": collection or "",
                        "decided_by": decided_by or ""},
        },
    )


@app.get("/stats", response_class=HTMLResponse)
def page_stats(request: Request) -> HTMLResponse:
    quarantine, quarantine_total = db.quarantined_facts(limit=25)
    return _render(
        request,
        "stats.html",
        {
            "stats": db.stats(),
            "registry": registry.snapshot()[:60],
            "quarantine": quarantine,
            "quarantine_total": quarantine_total,
            "runs": db.query(
                "SELECT r.stage, r.n_items, r.n_llm_calls, r.tokens_in, r.tokens_out, r.notes,"
                " d.filename FROM runs r LEFT JOIN documents d ON d.doc_id = r.doc_id"
                " ORDER BY r.started_at DESC LIMIT 60"
            ),
            "active": "stats",
        },
    )


def _shortnum(value: Any) -> str:
    """Render a number compactly, keeping small values readable."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e7, "Cr"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= limit:
            return f"{number / limit:,.4g} {suffix}"
    return f"{number:,.6g}"


def _jsonpretty(value: Any) -> str:
    return json.dumps(_json_safe(value), indent=2, ensure_ascii=False)


templates.env.filters["shortnum"] = _shortnum
templates.env.filters["jsonpretty"] = _jsonpretty


def _from_json(value: Any) -> dict[str, Any]:
    """Decode a JSON column for a template. Bad or missing data renders empty."""
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


templates.env.filters["from_json"] = _from_json


# ---------------------------------------------------------------------------
# React single page app
#
# The four Jinja pages above are unchanged and stay on their documented URLs.
# The React build, when present, is served alongside them under /app. Nothing
# here runs if web/dist has not been built, so the backend still starts on a
# fresh clone with no Node toolchain installed.
# ---------------------------------------------------------------------------

WEB_DIST = BASE_DIR.parent / "web" / "dist"

if (WEB_DIST / "index.html").is_file():
    if (WEB_DIST / "assets").is_dir():
        app.mount("/app/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="spa-assets")

    @app.get("/app", response_class=HTMLResponse)
    @app.get("/app/{path:path}", response_class=HTMLResponse)
    def spa(path: str = "") -> HTMLResponse:
        """Serve the SPA shell for every client side route under /app."""
        return HTMLResponse((WEB_DIST / "index.html").read_text(encoding="utf-8"))
