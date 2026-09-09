"""Page aware chunking that never splits a table and never loses an offset.

Chunks are pure slices of the document buffer. A chunk's ``char_start`` and
``char_end`` are absolute buffer offsets, so a quote verified inside a chunk
resolves to a page without any further bookkeeping.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import config
from app.parsing import ParsedDocument, ParsedPage

_TABLE_LINE = re.compile(r"^\s*(\|.*\||\[table \d+\])\s*$")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_index_start: int
    page_index_end: int
    text: str
    char_start: int
    char_end: int
    kind: str  # "prose" | "table" | "mixed"


@dataclass(slots=True)
class _Segment:
    start: int
    end: int
    is_table: bool


def normalize_for_hash(text: str) -> str:
    """Whitespace insensitive form used for content hashing."""
    return " ".join(text.split()).lower()


def make_chunk_id(doc_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}\x00{normalize_for_hash(text)}".encode("utf-8"))
    return digest.hexdigest()[:32]


def _segment_page(text: str) -> list[_Segment]:
    """Split page text into alternating prose and table segments by line."""
    segments: list[_Segment] = []
    offset = 0
    current_start = 0
    current_is_table: bool | None = None
    for line in text.splitlines(keepends=True):
        is_table = bool(_TABLE_LINE.match(line))
        if current_is_table is None:
            current_is_table = is_table
        elif is_table != current_is_table:
            segments.append(_Segment(current_start, offset, current_is_table))
            current_start = offset
            current_is_table = is_table
        offset += len(line)
    if current_is_table is not None and offset > current_start:
        segments.append(_Segment(current_start, offset, current_is_table))
    return segments or [_Segment(0, len(text), False)]


def _classify(text: str, segments: list[_Segment], start: int, end: int) -> str:
    table_chars = sum(
        max(0, min(end, s.end) - max(start, s.start)) for s in segments if s.is_table
    )
    span = max(1, end - start)
    ratio = table_chars / span
    if ratio >= 0.7:
        return "table"
    if ratio > 0.0:
        return "mixed"
    return "prose"


def _split_prose(text: str, start: int, end: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Split a prose span at paragraph or sentence boundaries near the target."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + target, end)
        if stop < end:
            window = text[cursor:stop]
            for boundary in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(boundary)
                if idx > target // 2:
                    stop = cursor + idx + len(boundary)
                    break
        spans.append((cursor, stop))
        if stop >= end:
            break
        cursor = max(cursor + 1, stop - overlap)
    return spans


def chunk_page(
    page: ParsedPage,
    doc_id: str,
    *,
    whole_page: bool = False,
    target: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Chunk a single page. Table segments are kept whole."""
    target = config.CHUNK_TARGET_CHARS if target is None else target
    overlap = config.CHUNK_OVERLAP_CHARS if overlap is None else overlap
    min_chars = config.CHUNK_MIN_CHARS if min_chars is None else min_chars

    text = page.text
    if not text.strip():
        return []

    segments = _segment_page(text)
    if whole_page and len(text) <= config.SLIDE_PAGE_MAX_CHARS:
        local_spans: list[tuple[int, int]] = [(0, len(text))]
    else:
        local_spans = []
        buffered: list[tuple[int, int]] = []  # pending prose spans to merge

        def flush() -> None:
            if not buffered:
                return
            local_spans.extend(buffered)
            buffered.clear()

        pending_start: int | None = None
        pending_end = 0
        for segment in segments:
            if segment.is_table:
                if pending_start is not None:
                    buffered.extend(_split_prose(text, pending_start, pending_end, target, overlap))
                    pending_start = None
                flush()
                # A table is atomic. It joins the previous chunk only if both fit.
                length = segment.end - segment.start
                if local_spans and (segment.end - local_spans[-1][0]) <= target + length // 2:
                    last_start, _ = local_spans.pop()
                    local_spans.append((last_start, segment.end))
                else:
                    local_spans.append((segment.start, segment.end))
            else:
                if pending_start is None:
                    pending_start = segment.start
                pending_end = segment.end
        if pending_start is not None:
            buffered.extend(_split_prose(text, pending_start, pending_end, target, overlap))
        flush()

    chunks: list[Chunk] = []
    for local_start, local_end in local_spans:
        body = text[local_start:local_end]
        if len(body.strip()) < min_chars and chunks:
            # Fold a runt tail into the previous chunk rather than emit noise.
            previous = chunks[-1]
            merged_text = text[previous.char_start - page.char_start : local_end]
            chunks[-1] = Chunk(
                chunk_id=make_chunk_id(doc_id, merged_text),
                doc_id=doc_id,
                page_index_start=page.page_index,
                page_index_end=page.page_index,
                text=merged_text,
                char_start=previous.char_start,
                char_end=page.char_start + local_end,
                kind=_classify(text, segments, previous.char_start - page.char_start, local_end),
            )
            continue
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, body),
                doc_id=doc_id,
                page_index_start=page.page_index,
                page_index_end=page.page_index,
                text=body,
                char_start=page.char_start + local_start,
                char_end=page.char_start + local_end,
                kind=_classify(text, segments, local_start, local_end),
            )
        )
    return chunks


def chunk_document(document: ParsedDocument) -> list[Chunk]:
    """Chunk every page. Slide decks keep one chunk per slide."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for page in document.pages:
        for chunk in chunk_page(page, document.doc_id, whole_page=document.is_slide_deck):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks
