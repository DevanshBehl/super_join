#!/usr/bin/env python3
"""Ingest every PDF under data/ into a collection named after its folder.

Usage:
    python scripts/ingest_all.py                 ingest everything under data/
    python scripts/ingest_all.py data/delhivery  ingest one collection
    python scripts/ingest_all.py --dry-run       list what would be ingested
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from app import db, llm, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the knowledge layer.")
    parser.add_argument("paths", nargs="*", default=[], help="files or folders, default is data/")
    parser.add_argument("--collection", default=None, help="override the collection name")
    parser.add_argument("--dry-run", action="store_true", help="list the files and exit")
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths] or [config.DATA_DIR]
    pdfs: list[tuple[Path, str]] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".pdf":
            pdfs.append((root, args.collection or root.parent.name))
            continue
        for pdf in sorted(root.rglob("*.pdf")):
            collection = args.collection or (pdf.parent.name if pdf.parent != root else root.name)
            pdfs.append((pdf, collection))

    if not pdfs:
        print(f"No PDFs found under {', '.join(str(r) for r in roots)}")
        return 1

    print(f"Found {len(pdfs)} PDF files.")
    for pdf, collection in pdfs:
        print(f"  {collection:20s} {pdf}")
    if args.dry_run:
        return 0

    if not llm.llm_available():
        print(
            "\nWarning: GEMINI_API_KEY is not set or the model layer is disabled.\n"
            "Parsing and chunking will run, but no facts will be extracted.\n"
        )

    db.connect()
    started = time.perf_counter()
    totals = {"facts": 0, "verified": 0, "links": 0, "llm_calls": 0, "cache_hits": 0}
    for index, (pdf, collection) in enumerate(pdfs, start=1):
        print(f"\n[{index}/{len(pdfs)}] {pdf.name} -> {collection}")
        stage_started = time.perf_counter()

        def progress(stage: str, _pdf: Path = pdf) -> None:
            print(f"    {stage} ...", flush=True)

        result = pipeline.ingest_path(pdf, collection, progress=progress)
        totals["facts"] += result.n_facts
        totals["verified"] += result.n_verified
        totals["links"] += result.rule_decided + result.llm_adjudicated
        totals["llm_calls"] += result.n_llm_calls
        totals["cache_hits"] += result.n_cache_hits
        print(
            f"    pages {result.n_pages}, chunks {result.n_chunks} "
            f"({result.n_chunks_skipped} skipped as boilerplate), "
            f"facts {result.n_facts} of which {result.n_verified} verified "
            f"({result.verification_rate * 100:.1f} per cent)"
        )
        print(
            f"    funnel: {result.blocked_pairs} blocked -> {result.candidate_pairs} retrieved -> "
            f"{result.rule_decided} settled by rules -> {result.llm_adjudicated} adjudicated"
        )
        print(
            f"    model calls {result.n_llm_calls}, cache hits {result.n_cache_hits}, "
            f"elapsed {time.perf_counter() - stage_started:.1f} s"
        )

    elapsed = time.perf_counter() - started
    stats = db.stats()
    print("\n" + "=" * 72)
    print(f"Ingested {len(pdfs)} documents in {elapsed:.1f} s")
    print(f"  facts:          {stats['n_facts']} extracted, {stats['n_verified_facts']} verified "
          f"({stats['verification_rate'] * 100:.1f} per cent), {stats['n_quarantined']} quarantined")
    print(f"  entities:       {stats['n_entities']}")
    print(f"  predicates:     {stats['n_predicates']}")
    print(f"  links:          {stats['n_links']} -> {stats['links_by_relation']}")
    print(f"  funnel:         {stats['funnel']}")
    print(f"  model calls:    {stats['n_llm_calls']} ({totals['cache_hits']} served from cache)")
    print(f"  tokens:         {stats['tokens_in']} in, {stats['tokens_out']} out")
    print(f"  database:       {config.DB_PATH}")
    print(f"  traces:         {config.TRACE_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
