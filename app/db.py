"""DuckDB storage: explicit DDL, idempotent writes, and query helpers.

One file at ``config.DB_PATH`` holds the entire knowledge layer. There is no
external service. Vector search is DuckDB's ``array_cosine_similarity`` over a
fixed width ``FLOAT[]`` column, which is fast enough at this scale and removes
a whole class of deployment problems.

Every id is a content hash, so re-ingesting the same bytes rewrites the same
rows instead of creating new ones.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import duckdb

import config

_LOCK = threading.RLock()
_CONNECTION: duckdb.DuckDBPyConnection | None = None

DIM = config.EMBEDDING_DIM

DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id          TEXT PRIMARY KEY,
        filename        TEXT NOT NULL,
        collection      TEXT NOT NULL,
        title           TEXT,
        publisher       TEXT,
        doc_type        TEXT,
        as_of_date      DATE,
        published_date  DATE,
        n_pages         INTEGER,
        ingested_at     TIMESTAMP,
        parser_version  TEXT,
        source_path     TEXT,
        status          TEXT DEFAULT 'pending',
        buffer          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pages (
        doc_id      TEXT NOT NULL,
        page_index  INTEGER NOT NULL,
        page_label  TEXT,
        text        TEXT,
        text_plain  TEXT,
        char_start  BIGINT,
        char_end    BIGINT,
        text_source TEXT,
        PRIMARY KEY (doc_id, page_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id         TEXT PRIMARY KEY,
        doc_id           TEXT NOT NULL,
        page_index_start INTEGER,
        page_index_end   INTEGER,
        text             TEXT,
        char_start       BIGINT,
        char_end         BIGINT,
        kind             TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        fact_id             TEXT PRIMARY KEY,
        doc_id              TEXT NOT NULL,
        chunk_id            TEXT NOT NULL,
        subject_raw         TEXT,
        subject_canonical   TEXT,
        entity_id           TEXT,
        predicate           TEXT,
        predicate_canonical TEXT,
        value_raw           TEXT,
        value_type          TEXT,
        value_num           DOUBLE,
        value_text          TEXT,
        unit_raw            TEXT,
        unit_canonical      TEXT,
        unit_dimension      TEXT,
        magnitude           DOUBLE,
        magnitude_label     TEXT,
        currency            TEXT,
        period_start        DATE,
        period_end          DATE,
        period_label_raw    TEXT,
        period_basis        TEXT,
        scope_tags          TEXT,
        qualifiers          TEXT,
        confidence          DOUBLE,
        evidence_id         TEXT,
        extractor_version   TEXT,
        created_at          TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence (
        evidence_id  TEXT PRIMARY KEY,
        fact_id      TEXT NOT NULL,
        doc_id       TEXT NOT NULL,
        page_index   INTEGER,
        page_label   TEXT,
        quote        TEXT,
        char_start   BIGINT,
        char_end     BIGINT,
        bbox         TEXT,
        verified     BOOLEAN,
        match_score  DOUBLE,
        match_method TEXT
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS entities (
        entity_id      TEXT PRIMARY KEY,
        canonical_name TEXT,
        entity_type    TEXT,
        aliases        TEXT,
        embedding      FLOAT[{DIM}],
        created_at     TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predicate_registry (
        predicate_canonical    TEXT PRIMARY KEY,
        description            TEXT,
        expected_value_type    TEXT,
        expected_unit_dimension TEXT,
        example_fact_id        TEXT,
        first_seen_doc_id      TEXT,
        n_facts                INTEGER DEFAULT 0,
        created_at             TIMESTAMP
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS fact_embeddings (
        fact_id          TEXT PRIMARY KEY,
        canonical_string TEXT,
        embedding        FLOAT[{DIM}]
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS embedding_cache (
        text_hash TEXT PRIMARY KEY,
        model     TEXT,
        embedding FLOAT[{DIM}]
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS links (
        link_id        TEXT PRIMARY KEY,
        fact_a         TEXT NOT NULL,
        fact_b         TEXT NOT NULL,
        relation       TEXT,
        confidence     DOUBLE,
        decided_by     TEXT,
        rule_trace     TEXT,
        explanation    TEXT,
        numeric_delta  DOUBLE,
        relative_delta DOUBLE,
        created_at     TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        doc_id      TEXT,
        stage       TEXT,
        started_at  TIMESTAMP,
        finished_at TIMESTAMP,
        n_items     INTEGER,
        n_llm_calls INTEGER,
        tokens_in   INTEGER,
        tokens_out  INTEGER,
        errors      TEXT,
        notes       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key      TEXT PRIMARY KEY,
        content_hash   TEXT,
        prompt_version TEXT,
        model          TEXT,
        schema_version TEXT,
        response       TEXT,
        tokens_in      INTEGER,
        tokens_out     INTEGER,
        created_at     TIMESTAMP
    )
    """,
)

VIEWS: tuple[str, ...] = (
    """
    CREATE OR REPLACE VIEW active_facts AS
    SELECT f.* FROM facts f
    JOIN evidence e ON e.evidence_id = f.evidence_id
    WHERE e.verified
    """,
    """
    CREATE OR REPLACE VIEW quarantine AS
    SELECT f.*, e.quote AS failed_quote, e.match_score, e.match_method
    FROM facts f
    JOIN evidence e ON e.evidence_id = f.evidence_id
    WHERE NOT e.verified
    """,
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_pred ON facts(predicate_canonical)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_fact ON evidence(fact_id)",
    "CREATE INDEX IF NOT EXISTS idx_links_a ON links(fact_a)",
    "CREATE INDEX IF NOT EXISTS idx_links_b ON links(fact_b)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
)


def connect(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return the shared connection, creating and initialising it on first use."""
    global _CONNECTION
    with _LOCK:
        if _CONNECTION is None:
            config.ensure_dirs()
            target = str(path or config.DB_PATH)
            _CONNECTION = duckdb.connect(target)
            init_db(_CONNECTION)
        return _CONNECTION


def close() -> None:
    global _CONNECTION
    with _LOCK:
        if _CONNECTION is not None:
            _CONNECTION.close()
            _CONNECTION = None


@contextmanager
def cursor() -> Iterator[duckdb.DuckDBPyConnection]:
    """Serialise access. DuckDB allows one writer, so a lock is the whole story."""
    conn = connect()
    with _LOCK:
        yield conn


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL:
        conn.execute(statement)
    for statement in INDEXES:
        conn.execute(statement)
    for statement in VIEWS:
        conn.execute(statement)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def rows_to_dicts(conn: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    result = conn.execute(sql, list(params))
    columns = [d[0] for d in result.description or []]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def query(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with cursor() as conn:
        return rows_to_dicts(conn, sql, params)


def scalar(sql: str, params: Sequence[Any] = (), default: Any = 0) -> Any:
    with cursor() as conn:
        row = conn.execute(sql, list(params)).fetchone()
    if row is None or row[0] is None:
        return default
    return row[0]


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Writes. All of them replace by primary key so re-ingest is idempotent.
# ---------------------------------------------------------------------------


def upsert_document(doc: dict[str, Any]) -> None:
    with cursor() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, filename, collection, title, publisher, doc_type, as_of_date,
             published_date, n_pages, ingested_at, parser_version, source_path, status, buffer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                doc["doc_id"],
                doc["filename"],
                doc["collection"],
                doc.get("title"),
                doc.get("publisher"),
                doc.get("doc_type"),
                _parse_date(doc.get("as_of_date")),
                _parse_date(doc.get("published_date")),
                doc.get("n_pages", 0),
                doc.get("ingested_at") or datetime.utcnow(),
                doc.get("parser_version", config.PARSER_VERSION),
                doc.get("source_path"),
                doc.get("status", "pending"),
                doc.get("buffer", ""),
            ],
        )


def set_document_status(doc_id: str, status: str) -> None:
    with cursor() as conn:
        conn.execute("UPDATE documents SET status = ? WHERE doc_id = ?", [status, doc_id])


def update_document_meta(doc_id: str, meta: dict[str, Any]) -> None:
    with cursor() as conn:
        conn.execute(
            """
            UPDATE documents SET title = ?, publisher = ?, doc_type = ?,
                   as_of_date = ?, published_date = ? WHERE doc_id = ?
            """,
            [
                meta.get("title"),
                meta.get("publisher"),
                meta.get("doc_type"),
                _parse_date(meta.get("as_of_date")),
                _parse_date(meta.get("published_date")),
                doc_id,
            ],
        )


def replace_pages(doc_id: str, pages: Iterable[dict[str, Any]]) -> int:
    rows = [
        [
            doc_id,
            p["page_index"],
            p.get("page_label"),
            p.get("text", ""),
            p.get("text_plain", ""),
            p.get("char_start", 0),
            p.get("char_end", 0),
            p.get("text_source", ""),
        ]
        for p in pages
    ]
    with cursor() as conn:
        conn.execute("DELETE FROM pages WHERE doc_id = ?", [doc_id])
        if rows:
            conn.executemany(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
    return len(rows)


def replace_chunks(doc_id: str, chunks: Iterable[dict[str, Any]]) -> int:
    rows = [
        [
            c["chunk_id"],
            doc_id,
            c["page_index_start"],
            c["page_index_end"],
            c["text"],
            c["char_start"],
            c["char_end"],
            c["kind"],
        ]
        for c in chunks
    ]
    with cursor() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
        if rows:
            conn.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def insert_facts(facts: Sequence[dict[str, Any]]) -> int:
    if not facts:
        return 0
    rows = [
        [
            f["fact_id"],
            f["doc_id"],
            f["chunk_id"],
            f.get("subject_raw"),
            f.get("subject_canonical"),
            f.get("entity_id"),
            f.get("predicate"),
            f.get("predicate_canonical"),
            f.get("value_raw"),
            f.get("value_type"),
            f.get("value_num"),
            f.get("value_text"),
            f.get("unit_raw"),
            f.get("unit_canonical"),
            f.get("unit_dimension"),
            f.get("magnitude"),
            f.get("magnitude_label"),
            f.get("currency"),
            _parse_date(f.get("period_start")),
            _parse_date(f.get("period_end")),
            f.get("period_label_raw"),
            f.get("period_basis", "unknown"),
            dumps(f.get("scope_tags", [])),
            dumps(f.get("qualifiers", {})),
            f.get("confidence", 0.5),
            f.get("evidence_id"),
            f.get("extractor_version", config.EXTRACTOR_VERSION),
            f.get("created_at") or datetime.utcnow(),
        ]
        for f in facts
    ]
    with cursor() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO facts VALUES (" + ", ".join(["?"] * 28) + ")", rows
        )
    return len(rows)


def insert_evidence(items: Sequence[dict[str, Any]]) -> int:
    if not items:
        return 0
    rows = [
        [
            e["evidence_id"],
            e["fact_id"],
            e["doc_id"],
            e.get("page_index"),
            e.get("page_label"),
            e.get("quote", ""),
            e.get("char_start"),
            e.get("char_end"),
            dumps(e.get("bbox", [])),
            bool(e.get("verified", False)),
            float(e.get("match_score", 0.0)),
            e.get("match_method", "none"),
        ]
        for e in items
    ]
    with cursor() as conn:
        conn.executemany("INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def delete_document_facts(doc_id: str) -> None:
    """Remove a document's derived rows so a re-ingest cannot duplicate them."""
    with cursor() as conn:
        conn.execute(
            "DELETE FROM links WHERE fact_a IN (SELECT fact_id FROM facts WHERE doc_id = ?)"
            " OR fact_b IN (SELECT fact_id FROM facts WHERE doc_id = ?)",
            [doc_id, doc_id],
        )
        conn.execute(
            "DELETE FROM fact_embeddings WHERE fact_id IN (SELECT fact_id FROM facts WHERE doc_id = ?)",
            [doc_id],
        )
        conn.execute("DELETE FROM evidence WHERE doc_id = ?", [doc_id])
        conn.execute("DELETE FROM facts WHERE doc_id = ?", [doc_id])


def upsert_entity(entity: dict[str, Any]) -> None:
    with cursor() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?)",
            [
                entity["entity_id"],
                entity["canonical_name"],
                entity.get("entity_type", "unknown"),
                dumps(entity.get("aliases", [])),
                entity.get("embedding"),
                entity.get("created_at") or datetime.utcnow(),
            ],
        )


def bump_predicate(entry: dict[str, Any], increment: int = 1) -> None:
    """Append a predicate to the registry or increase its count."""
    with cursor() as conn:
        existing = conn.execute(
            "SELECT n_facts FROM predicate_registry WHERE predicate_canonical = ?",
            [entry["predicate_canonical"]],
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO predicate_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    entry["predicate_canonical"],
                    entry.get("description", ""),
                    entry.get("expected_value_type", "numeric"),
                    entry.get("expected_unit_dimension"),
                    entry.get("example_fact_id"),
                    entry.get("first_seen_doc_id"),
                    increment,
                    datetime.utcnow(),
                ],
            )
        else:
            conn.execute(
                "UPDATE predicate_registry SET n_facts = n_facts + ? WHERE predicate_canonical = ?",
                [increment, entry["predicate_canonical"]],
            )


def recount_predicates() -> None:
    """Recompute registry counts from the facts table. Safe to call any time."""
    with cursor() as conn:
        conn.execute(
            """
            UPDATE predicate_registry AS r
            SET n_facts = COALESCE(
                (SELECT COUNT(*) FROM facts f WHERE f.predicate_canonical = r.predicate_canonical), 0)
            """
        )


def top_predicates(limit: int) -> list[dict[str, Any]]:
    return query(
        "SELECT predicate_canonical, description, expected_value_type, expected_unit_dimension, n_facts"
        " FROM predicate_registry ORDER BY n_facts DESC, predicate_canonical LIMIT ?",
        [limit],
    )


def upsert_fact_embeddings(items: Sequence[tuple[str, str, list[float]]]) -> int:
    if not items:
        return 0
    with cursor() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO fact_embeddings VALUES (?, ?, ?)",
            [[fid, text, vector] for fid, text, vector in items],
        )
    return len(items)


def get_cached_embeddings(hashes: Sequence[str], model: str) -> dict[str, list[float]]:
    if not hashes:
        return {}
    placeholders = ", ".join(["?"] * len(hashes))
    rows = query(
        f"SELECT text_hash, embedding FROM embedding_cache WHERE model = ? AND text_hash IN ({placeholders})",
        [model, *hashes],
    )
    return {r["text_hash"]: list(r["embedding"]) for r in rows if r["embedding"] is not None}


def put_cached_embeddings(items: Sequence[tuple[str, str, list[float]]]) -> None:
    if not items:
        return
    with cursor() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?)",
            [[h, model, vec] for h, model, vec in items],
        )


def insert_links(links: Sequence[dict[str, Any]]) -> int:
    if not links:
        return 0
    rows = [
        [
            l["link_id"],
            l["fact_a"],
            l["fact_b"],
            l["relation"],
            l.get("confidence", 0.5),
            l.get("decided_by", "rule"),
            dumps(l.get("rule_trace", [])),
            l.get("explanation", ""),
            l.get("numeric_delta"),
            l.get("relative_delta"),
            l.get("created_at") or datetime.utcnow(),
        ]
        for l in links
    ]
    with cursor() as conn:
        conn.executemany("INSERT OR REPLACE INTO links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def record_run(run: dict[str, Any]) -> None:
    with cursor() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run["run_id"],
                run.get("doc_id"),
                run.get("stage", ""),
                run.get("started_at"),
                run.get("finished_at"),
                run.get("n_items", 0),
                run.get("n_llm_calls", 0),
                run.get("tokens_in", 0),
                run.get("tokens_out", 0),
                dumps(run.get("errors", [])),
                dumps(run.get("notes", {})),
            ],
        )


# ---------------------------------------------------------------------------
# LLM response cache
# ---------------------------------------------------------------------------


def cache_key(content_hash: str, prompt_version: str, model: str, schema_version: str) -> str:
    return f"{content_hash}:{prompt_version}:{model}:{schema_version}"


def get_llm_cache(key: str) -> str | None:
    with cursor() as conn:
        row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", [key]).fetchone()
    return row[0] if row else None


def put_llm_cache(
    key: str,
    content_hash: str,
    prompt_version: str,
    model: str,
    schema_version: str,
    response: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    with cursor() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [key, content_hash, prompt_version, model, schema_version, response, tokens_in, tokens_out, datetime.utcnow()],
        )


def checkpoint() -> None:
    """Flush the write ahead log into the database file.

    DuckDB keeps committed transactions in a .wal until it checkpoints. Calling
    this at a document boundary means an interrupted run loses at most the
    document in flight, and never the response cache.
    """
    try:
        with cursor() as conn:
            conn.execute("CHECKPOINT")
    except Exception:  # pragma: no cover - a busy checkpoint is not fatal
        pass


def document_exists(doc_id: str) -> bool:
    return bool(scalar("SELECT COUNT(*) FROM documents WHERE doc_id = ?", [doc_id], 0))


# ---------------------------------------------------------------------------
# Read helpers used by the API and the templates
# ---------------------------------------------------------------------------


def _decode_fact(row: dict[str, Any]) -> dict[str, Any]:
    row["scope_tags"] = loads(row.get("scope_tags"), [])
    row["qualifiers"] = loads(row.get("qualifiers"), {})
    if "bbox" in row:
        row["bbox"] = loads(row.get("bbox"), [])
    return row


def list_documents() -> list[dict[str, Any]]:
    return query(
        """
        SELECT d.doc_id, d.filename, d.collection, d.title, d.publisher, d.doc_type,
               d.as_of_date, d.published_date, d.n_pages, d.ingested_at, d.status,
               (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id) AS n_chunks,
               (SELECT COUNT(*) FROM facts f WHERE f.doc_id = d.doc_id) AS n_facts,
               (SELECT COUNT(*) FROM active_facts f WHERE f.doc_id = d.doc_id) AS n_verified_facts
        FROM documents d
        ORDER BY d.collection, d.filename
        """
    )


def get_document(doc_id: str) -> dict[str, Any] | None:
    rows = query("SELECT * FROM documents WHERE doc_id = ?", [doc_id])
    if not rows:
        return None
    row = rows[0]
    row.pop("buffer", None)
    return row


def document_runs(doc_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT run_id, stage, started_at, finished_at, n_items, n_llm_calls, tokens_in,"
        " tokens_out, errors, notes FROM runs WHERE doc_id = ? ORDER BY started_at",
        [doc_id],
    )
    for row in rows:
        row["errors"] = loads(row.get("errors"), [])
        row["notes"] = loads(row.get("notes"), {})
    return rows


_FACT_SELECT = """
    SELECT f.*, d.collection, d.title AS document_title, d.filename,
           ev.quote, ev.page_index, ev.page_label, ev.char_start AS evidence_char_start,
           ev.char_end AS evidence_char_end, ev.verified, ev.match_score, ev.match_method, ev.bbox
    FROM facts f
    JOIN documents d ON d.doc_id = f.doc_id
    LEFT JOIN evidence ev ON ev.evidence_id = f.evidence_id
"""


def search_facts(
    *,
    doc_id: str | None = None,
    collection: str | None = None,
    entity_id: str | None = None,
    predicate: str | None = None,
    period: str | None = None,
    q: str | None = None,
    verified: bool | None = True,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    clauses: list[str] = []
    params: list[Any] = []
    if verified is not None:
        clauses.append("ev.verified = ?")
        params.append(verified)
    if doc_id:
        clauses.append("f.doc_id = ?")
        params.append(doc_id)
    if collection:
        clauses.append("d.collection = ?")
        params.append(collection)
    if entity_id:
        clauses.append("f.entity_id = ?")
        params.append(entity_id)
    if predicate:
        clauses.append("f.predicate_canonical = ?")
        params.append(predicate)
    if period:
        clauses.append("lower(COALESCE(f.period_label_raw, '')) LIKE ?")
        params.append(f"%{period.lower()}%")
    if q:
        clauses.append(
            "(lower(f.subject_raw) LIKE ? OR lower(f.predicate_canonical) LIKE ?"
            " OR lower(f.value_raw) LIKE ? OR lower(COALESCE(ev.quote, '')) LIKE ?)"
        )
        needle = f"%{q.lower()}%"
        params.extend([needle] * 4)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    total = int(
        scalar(
            "SELECT COUNT(*) FROM facts f JOIN documents d ON d.doc_id = f.doc_id"
            " LEFT JOIN evidence ev ON ev.evidence_id = f.evidence_id" + where,
            params,
            0,
        )
    )
    rows = query(
        _FACT_SELECT + where + " ORDER BY f.confidence DESC, f.fact_id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    return [_decode_fact(row) for row in rows], total


def get_fact(fact_id: str) -> dict[str, Any] | None:
    rows = query(_FACT_SELECT + " WHERE f.fact_id = ?", [fact_id])
    return _decode_fact(rows[0]) if rows else None


def facts_by_ids(fact_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not fact_ids:
        return {}
    unique = sorted(set(fact_ids))
    placeholders = ", ".join(["?"] * len(unique))
    rows = query(_FACT_SELECT + f" WHERE f.fact_id IN ({placeholders})", unique)
    return {row["fact_id"]: _decode_fact(row) for row in rows}


def search_links(
    *,
    relation: str | None = None,
    collection: str | None = None,
    doc_id: str | None = None,
    min_confidence: float = 0.0,
    decided_by: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    clauses = ["l.confidence >= ?"]
    params: list[Any] = [min_confidence]
    if relation:
        clauses.append("l.relation = ?")
        params.append(relation)
    if decided_by:
        clauses.append("l.decided_by = ?")
        params.append(decided_by)
    if collection:
        clauses.append("da.collection = ?")
        params.append(collection)
    if doc_id:
        clauses.append("(fa.doc_id = ? OR fb.doc_id = ?)")
        params.extend([doc_id, doc_id])
    where = " WHERE " + " AND ".join(clauses)
    base = """
        FROM links l
        JOIN facts fa ON fa.fact_id = l.fact_a
        JOIN facts fb ON fb.fact_id = l.fact_b
        JOIN documents da ON da.doc_id = fa.doc_id
    """
    total = int(scalar("SELECT COUNT(*) " + base + where, params, 0))
    rows = query(
        "SELECT l.* " + base + where + " ORDER BY l.confidence DESC, l.link_id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    for row in rows:
        row["rule_trace"] = loads(row.get("rule_trace"), [])
    return rows, total


def get_link(link_id: str) -> dict[str, Any] | None:
    rows = query("SELECT * FROM links WHERE link_id = ?", [link_id])
    if not rows:
        return None
    row = rows[0]
    row["rule_trace"] = loads(row.get("rule_trace"), [])
    return row


def links_for_fact(fact_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT * FROM links WHERE fact_a = ? OR fact_b = ? ORDER BY confidence DESC",
        [fact_id, fact_id],
    )
    for row in rows:
        row["rule_trace"] = loads(row.get("rule_trace"), [])
    return rows


def quarantined_facts(limit: int = 200, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    total = int(scalar("SELECT COUNT(*) FROM quarantine", [], 0))
    rows = query(
        """
        SELECT f.*, d.filename, d.collection, d.title AS document_title,
               ev.quote, ev.match_score, ev.match_method, ev.page_index, ev.page_label
        FROM facts f
        JOIN evidence ev ON ev.evidence_id = f.evidence_id
        JOIN documents d ON d.doc_id = f.doc_id
        WHERE NOT ev.verified
        ORDER BY ev.match_score DESC LIMIT ? OFFSET ?
        """,
        [limit, offset],
    )
    return [_decode_fact(row) for row in rows], total


def collections() -> list[str]:
    return [r["collection"] for r in query("SELECT DISTINCT collection FROM documents ORDER BY collection")]


def stats() -> dict[str, Any]:
    """Every headline number the stats page shows, read from the tables."""
    n_facts = int(scalar("SELECT COUNT(*) FROM facts", [], 0))
    n_verified = int(scalar("SELECT COUNT(*) FROM active_facts", [], 0))
    runs = query("SELECT stage, n_items, n_llm_calls, tokens_in, tokens_out, notes FROM runs")
    funnel = {"blocked_pairs": 0, "candidate_pairs": 0, "rule_decided": 0, "residue_for_llm": 0}
    seconds: dict[str, float] = {}
    cache_hits = 0
    for run in runs:
        notes = loads(run.get("notes"), {})
        for key in funnel:
            funnel[key] += int(notes.get(key, 0) or 0)
        cache_hits += int(notes.get("cache_hits", 0) or 0)
        seconds[run["stage"]] = round(seconds.get(run["stage"], 0.0) + float(notes.get("seconds", 0) or 0), 2)
    return {
        "n_documents": int(scalar("SELECT COUNT(*) FROM documents", [], 0)),
        "n_pages": int(scalar("SELECT COUNT(*) FROM pages", [], 0)),
        "n_chunks": int(scalar("SELECT COUNT(*) FROM chunks", [], 0)),
        "n_facts": n_facts,
        "n_verified_facts": n_verified,
        "n_quarantined": n_facts - n_verified,
        "verification_rate": round(n_verified / n_facts, 4) if n_facts else 0.0,
        "n_entities": int(scalar("SELECT COUNT(*) FROM entities", [], 0)),
        "n_predicates": int(scalar("SELECT COUNT(*) FROM predicate_registry", [], 0)),
        "n_links": int(scalar("SELECT COUNT(*) FROM links", [], 0)),
        "links_by_relation": {
            r["relation"]: r["n"]
            for r in query("SELECT relation, COUNT(*) AS n FROM links GROUP BY relation ORDER BY n DESC")
        },
        "links_by_decider": {
            r["decided_by"]: r["n"]
            for r in query("SELECT decided_by, COUNT(*) AS n FROM links GROUP BY decided_by")
        },
        "funnel": funnel,
        "n_llm_calls": int(scalar("SELECT COALESCE(SUM(n_llm_calls), 0) FROM runs", [], 0)),
        "n_cache_hits": cache_hits,
        "tokens_in": int(scalar("SELECT COALESCE(SUM(tokens_in), 0) FROM runs", [], 0)),
        "tokens_out": int(scalar("SELECT COALESCE(SUM(tokens_out), 0) FROM runs", [], 0)),
        "seconds_by_stage": seconds,
        "predicate_growth": query(
            "SELECT CAST(created_at AS DATE) AS day, COUNT(*) AS n_new"
            " FROM predicate_registry GROUP BY 1 ORDER BY 1"
        ),
    }


def export_all(limit_per_table: int = 100000) -> dict[str, Any]:
    """The whole knowledge layer as plain JSON friendly rows."""
    payload: dict[str, Any] = {}
    payload["documents"] = [
        {k: v for k, v in row.items() if k != "buffer"}
        for row in query("SELECT * FROM documents LIMIT ?", [limit_per_table])
    ]
    payload["facts"] = [
        _decode_fact(row) for row in query("SELECT * FROM facts LIMIT ?", [limit_per_table])
    ]
    payload["evidence"] = [
        {**row, "bbox": loads(row.get("bbox"), [])}
        for row in query("SELECT * FROM evidence LIMIT ?", [limit_per_table])
    ]
    payload["entities"] = [
        {k: v for k, v in row.items() if k != "embedding"} | {"aliases": loads(row.get("aliases"), [])}
        for row in query("SELECT * FROM entities LIMIT ?", [limit_per_table])
    ]
    payload["links"] = [
        {**row, "rule_trace": loads(row.get("rule_trace"), [])}
        for row in query("SELECT * FROM links LIMIT ?", [limit_per_table])
    ]
    payload["predicate_registry"] = query("SELECT * FROM predicate_registry LIMIT ?", [limit_per_table])
    payload["runs"] = [
        {**row, "errors": loads(row.get("errors"), []), "notes": loads(row.get("notes"), {})}
        for row in query("SELECT * FROM runs LIMIT ?", [limit_per_table])
    ]
    payload["stats"] = stats()
    return payload
