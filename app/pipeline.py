"""Ingest orchestration: parse, chunk, extract, verify, resolve, embed, link.

Each stage opens a ``RunRecorder``, so the ``runs`` table ends up holding a per
stage record of item counts, model calls, cache hits, timings, and errors for
every document ever ingested. That record is what the stats page and the
incremental ingest claims are read from, rather than a number written by hand.

Re-ingesting a document deliberately runs the whole pipeline again instead of
short circuiting. Every id is a content hash and every model response is
cached, so the second run rewrites identical rows and makes zero model calls.
That is a stronger guarantee than skipping, and it is checkable.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from app import adjudicate as adjudicate_module
from app import db, embed, entities, extract, link, registry
from app.chunking import chunk_document
from app.parsing import ParsedDocument, page_lookup, parse_pdf
from app.tracing import RunRecorder, trace_event

STAGES = ("parse", "chunk", "docmeta", "extract", "entities", "embed", "link", "adjudicate")


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    collection: str
    n_pages: int = 0
    n_chunks: int = 0
    n_chunks_skipped: int = 0
    n_facts: int = 0
    n_verified: int = 0
    n_quarantined: int = 0
    n_entities: int = 0
    n_predicates_new: int = 0
    blocked_pairs: int = 0
    candidate_pairs: int = 0
    rule_decided: int = 0
    rule_unrelated: int = 0
    llm_adjudicated: int = 0
    n_llm_calls: int = 0
    n_cache_hits: int = 0
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verification_rate(self) -> float:
        return self.n_verified / self.n_facts if self.n_facts else 0.0


def ingest_path(path: str | Path, collection: str, *, progress: Any | None = None) -> IngestResult:
    path = Path(path)
    return ingest_bytes(path.read_bytes(), path.name, collection, source_path=str(path), progress=progress)


def ingest_bytes(
    data: bytes,
    filename: str,
    collection: str,
    *,
    source_path: str | None = None,
    progress: Any | None = None,
) -> IngestResult:
    """Run the full pipeline for one document and return its counters."""

    def announce(stage: str) -> None:
        if progress is not None:
            progress(stage)

    # -- parse ---------------------------------------------------------------
    announce("parse")
    with RunRecorder("parse") as run:
        document = parse_pdf(source_path or filename, data=data)
        run.doc_id = document.doc_id
        run.n_items = document.n_pages
        for warning in document.warnings:
            run.add_error(str(warning.get("kind", "parse_warning")), str(warning.get("detail", "")),
                          page_index=warning.get("page_index"))
        db.upsert_document(
            {
                "doc_id": document.doc_id,
                "filename": filename,
                "collection": collection,
                "n_pages": document.n_pages,
                "ingested_at": datetime.utcnow(),
                "parser_version": document.parser_version,
                "source_path": source_path,
                "status": "parsing",
                "buffer": document.buffer,
            }
        )
        db.replace_pages(
            document.doc_id,
            [
                {
                    "page_index": page.page_index,
                    "page_label": page.page_label,
                    "text": page.text,
                    "text_plain": page.text_plain,
                    "char_start": page.char_start,
                    "char_end": page.char_end,
                    "text_source": page.text_source,
                }
                for page in document.pages
            ],
        )
        db.delete_document_facts(document.doc_id)
        result = IngestResult(document.doc_id, filename, collection, n_pages=document.n_pages)
        result.errors.extend(run.errors)
    result.seconds_by_stage["parse"] = run.notes["seconds"]

    # -- chunk ---------------------------------------------------------------
    announce("chunk")
    with RunRecorder("chunk", document.doc_id) as run:
        chunks = chunk_document(document)
        run.n_items = len(chunks)
        db.replace_chunks(
            document.doc_id,
            [
                {
                    "chunk_id": c.chunk_id, "page_index_start": c.page_index_start,
                    "page_index_end": c.page_index_end, "text": c.text,
                    "char_start": c.char_start, "char_end": c.char_end, "kind": c.kind,
                }
                for c in chunks
            ],
        )
        result.n_chunks = len(chunks)
    result.seconds_by_stage["chunk"] = run.notes["seconds"]

    # -- document metadata ---------------------------------------------------
    announce("docmeta")
    db.set_document_status(document.doc_id, "extracting")
    with RunRecorder("docmeta", document.doc_id) as run:
        meta = extract.infer_document_meta(document, recorder=run)
        if meta is not None:
            db.update_document_meta(
                document.doc_id,
                {
                    "title": meta.title or filename,
                    "publisher": meta.publisher or None,
                    "doc_type": meta.doc_type or None,
                    "as_of_date": meta.as_of_date or None,
                    "published_date": meta.published_date or None,
                },
            )
            run.n_items = 1
        result.n_llm_calls += run.n_llm_calls
        result.n_cache_hits += run.n_cache_hits
    result.seconds_by_stage["docmeta"] = run.notes["seconds"]

    # -- extract and verify --------------------------------------------------
    announce("extract")
    document_row = (db.query("SELECT * FROM documents WHERE doc_id = ?", [document.doc_id]) or [{}])[0]
    context = extract.document_context(document_row)
    vocabulary = registry.render_vocabulary_hint()
    offsets = page_lookup(document.pages)
    page_labels = {page.page_index: page.page_label for page in document.pages}

    with RunRecorder("extract", document.doc_id) as run:
        facts: list[dict[str, Any]] = []
        evidences: list[dict[str, Any]] = []
        work = []
        for chunk in chunks:
            skip, reason = extract.is_boilerplate(chunk.text, chunk.kind)
            if skip:
                result.n_chunks_skipped += 1
                trace_event({"type": "chunk_skipped", "chunk_id": chunk.chunk_id, "reason": reason})
                continue
            work.append(chunk)

        def run_chunk(chunk):
            return chunk, extract.extract_chunk(
                chunk, context=context, vocabulary=vocabulary,
                page_label=page_labels.get(chunk.page_index_start), recorder=run,
            )

        max_workers = max(1, min(config.LLM_MAX_CONCURRENCY, len(work) or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for chunk, (raw_facts, _cached) in pool.map(run_chunk, work):
                for raw in raw_facts:
                    fact, evidence = extract.build_fact(
                        raw, chunk, document, offsets, doc_id=document.doc_id, source_path=source_path,
                    )
                    facts.append(fact)
                    evidences.append(evidence)

        by_id = {fact["fact_id"]: fact for fact in facts}
        evidence_by_id = {ev["fact_id"]: ev for ev in evidences}
        facts = list(by_id.values())
        evidences = [evidence_by_id[fid] for fid in by_id]

        result.n_facts = len(facts)
        result.n_verified = sum(1 for ev in evidences if ev["verified"])
        result.n_quarantined = result.n_facts - result.n_verified
        run.n_items = result.n_facts
        run.notes["verification_rate"] = round(result.verification_rate, 4)
        run.notes["chunks_skipped"] = result.n_chunks_skipped
        for evidence in evidences:
            if not evidence["verified"]:
                run.add_error(
                    "evidence_unverified",
                    f"quote could not be located in its chunk (score {evidence['match_score']:.1f})",
                    fact_id=evidence["fact_id"],
                )
        # Persist immediately. Every later stage only enriches these rows, so a
        # failure after this point costs a stage, not the whole extraction.
        db.insert_facts(facts)
        db.insert_evidence(evidences)
        result.n_llm_calls += run.n_llm_calls
        result.n_cache_hits += run.n_cache_hits
    result.seconds_by_stage["extract"] = run.notes["seconds"]

    # -- entities ------------------------------------------------------------
    announce("entities")
    with RunRecorder("entities", document.doc_id) as run:
        resolver = entities.EntityResolver(embed_fn=embed.make_entity_embedder()).load()
        methods = resolver.assign(facts)
        written = resolver.flush()
        run.n_items = written
        run.notes["methods"] = methods
        if resolver.close_calls:
            run.notes["close_calls"] = resolver.close_calls[:20]
            for call in resolver.close_calls[:20]:
                run.add_error("entity_close_call", str(call), subject=call.get("subject"), decision=call.get("decision"))
        result.n_entities = written
        db.insert_facts(facts)  # idempotent rewrite, now carrying entity_id
        result.n_predicates_new = registry.register_facts(
            [f for f, ev in zip(facts, evidences) if ev["verified"]]
        )
        db.recount_predicates()
    result.seconds_by_stage["entities"] = run.notes["seconds"]

    # -- embed ---------------------------------------------------------------
    announce("embed")
    with RunRecorder("embed", document.doc_id) as run:
        verified_facts = [f for f, ev in zip(facts, evidences) if ev["verified"]]
        run.n_items = embed.embed_facts(verified_facts, recorder=run)
        result.n_llm_calls += run.n_llm_calls
        result.n_cache_hits += run.n_cache_hits
    result.seconds_by_stage["embed"] = run.notes["seconds"]

    # -- link ----------------------------------------------------------------
    announce("link")
    db.set_document_status(document.doc_id, "linking")
    with RunRecorder("link", document.doc_id) as run:
        result.blocked_pairs = link.count_blocked_pairs(document.doc_id)
        pairs = link.generate_candidates(document.doc_id)
        result.candidate_pairs = len(pairs)
        settled, residue = link.decide_pairs(pairs)
        storable = [s for s in settled if not s.pop("_skip_store", False)]
        result.rule_unrelated = len(settled) - len(storable)
        result.rule_decided = len(settled)
        db.insert_links([{k: v for k, v in s.items() if k != "similarity"} for s in storable])
        run.n_items = len(storable)
        run.notes.update(
            {
                "blocked_pairs": result.blocked_pairs,
                "candidate_pairs": result.candidate_pairs,
                "rule_decided": result.rule_decided,
                "rule_unrelated_not_stored": result.rule_unrelated,
                "residue_for_llm": len(residue),
            }
        )
    result.seconds_by_stage["link"] = run.notes["seconds"]

    # -- adjudicate ----------------------------------------------------------
    announce("adjudicate")
    with RunRecorder("adjudicate", document.doc_id) as run:
        decided = adjudicate_module.adjudicate(residue, recorder=run)
        db.insert_links([{k: v for k, v in d.items() if k != "similarity"} for d in decided])
        run.n_items = len(decided)
        result.llm_adjudicated = len(decided)
        result.n_llm_calls += run.n_llm_calls
        result.n_cache_hits += run.n_cache_hits
    result.seconds_by_stage["adjudicate"] = run.notes["seconds"]

    db.set_document_status(document.doc_id, "done")
    db.checkpoint()
    announce("done")
    return result


def ingest_directory(root: str | Path, *, collection: str | None = None) -> list[IngestResult]:
    """Ingest every PDF under a directory, using folder names as collections."""
    root = Path(root)
    results: list[IngestResult] = []
    for pdf in sorted(root.rglob("*.pdf")):
        name = collection or (pdf.parent.name if pdf.parent != root else config.DEFAULT_COLLECTION)
        results.append(ingest_path(pdf, name))
    return results
