"""PDF parsing: pages, canonical text, character offsets, and page labels.

Design notes that matter for grounding:

*   Every downstream stage addresses text by character offsets into a single
    per document buffer. The buffer is the concatenation of the canonical text
    of each page, so any offset resolves back to exactly one page.
*   The canonical text of a page is the ``pymupdf4llm`` markdown rendering when
    that rendering keeps essentially all of the page's characters, because the
    markdown keeps tables as pipe tables. On pages where the layout analysis
    drops content, which happens on graphics heavy pages such as slides and
    charts, the markdown is rejected and the canonical text becomes the plain
    ``page.get_text`` output with any well formed tables appended. Nothing is
    ever silently lost.
*   Printed page labels are detected from the top and bottom margins and stored
    separately from the zero based PDF page index. They are frequently
    different, and in curated excerpts they jump.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pymupdf
import pymupdf4llm

import config

PAGE_SEPARATOR = "\n\n"

# A page whose markdown rendering keeps fewer than this fraction of the plain
# text alphanumeric characters is treated as a layout analysis failure.
MARKDOWN_MIN_RETENTION = 0.95

_ARABIC_LABEL = re.compile(r"^(?:page\s*)?(\d{1,4})$", re.IGNORECASE)
_ROMAN_LABEL = re.compile(r"^(?=[ivxlcdm]{1,7}$)m*(?:c[md]|d?c{0,3})(?:x[cl]|l?x{0,3})(?:i[xv]|v?i{0,3})$", re.IGNORECASE)
_SLIDE_PRODUCERS = re.compile(r"powerpoint|keynote|google slides|impress|canva", re.IGNORECASE)


@dataclass(slots=True)
class ParsedPage:
    """One PDF page with its canonical text located inside the document buffer."""

    page_index: int
    page_label: str | None
    text: str
    text_plain: str
    char_start: int
    char_end: int
    width: float
    height: float
    n_images: int
    text_source: str  # "markdown" | "text+tables" | "text"


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    filename: str
    path: str
    n_pages: int
    buffer: str
    pages: list[ParsedPage]
    metadata: dict[str, str]
    is_slide_deck: bool
    warnings: list[dict[str, object]] = field(default_factory=list)
    parser_version: str = config.PARSER_VERSION

    def page_for_offset(self, offset: int) -> ParsedPage | None:
        """Return the page containing a buffer offset, or None if out of range."""
        for page in self.pages:
            if page.char_start <= offset < page.char_end:
                return page
        # An offset landing exactly on a page end still belongs to that page.
        for page in self.pages:
            if page.char_start <= offset <= page.char_end:
                return page
        return None


def compute_doc_id(data: bytes) -> str:
    """Content hash of the file bytes. Re-uploading a file is idempotent."""
    return hashlib.sha256(data).hexdigest()


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def detect_page_label(page: pymupdf.Page, margin_fraction: float | None = None) -> str | None:
    """Find a printed page label in the top or bottom margin of a page.

    Looks for a bare integer, a ``Page N`` form, or a roman numeral standing on
    its own. The bottom margin wins over the top margin because footers carry
    the page number far more often than headers. Returns None when nothing
    convincing is present, which is a normal outcome.
    """
    fraction = config.PAGE_LABEL_MARGIN_FRACTION if margin_fraction is None else margin_fraction
    height = page.rect.height
    if height <= 0:
        return None
    top_limit = page.rect.y0 + height * fraction
    bottom_limit = page.rect.y1 - height * fraction

    top: list[tuple[float, str]] = []
    bottom: list[tuple[float, str]] = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        token = word.strip().strip(".,:;|[]()")
        if not token:
            continue
        if y1 <= top_limit:
            top.append((x0, token))
        elif y0 >= bottom_limit:
            bottom.append((x0, token))

    for band in (bottom, top):
        arabic = [t for _, t in band if _ARABIC_LABEL.match(t)]
        if arabic:
            match = _ARABIC_LABEL.match(arabic[0])
            assert match is not None
            return match.group(1)
        roman = [t for _, t in band if _ROMAN_LABEL.match(t) and len(t) > 1]
        if roman:
            return roman[0].lower()
    return None


def _table_is_useful(rows: Sequence[Sequence[str | None]]) -> bool:
    """Reject the pseudo tables that table finders hallucinate over charts.

    A real table has at least two rows and two columns, is mostly populated,
    and mostly holds distinct values. Chart artefacts fail all three: they are
    sparse and they repeat one fragment of a graphic label many times.
    """
    if len(rows) < 2 or not rows[0] or len(rows[0]) < 2:
        return False
    cells = [str(c).strip() for row in rows for c in row if c is not None and str(c).strip()]
    total = sum(len(row) for row in rows)
    if total == 0 or len(cells) / total < 0.5:
        return False
    return len(set(cells)) / len(cells) >= 0.6


def _tables_markdown(page: pymupdf.Page) -> str:
    """Render the page's genuine tables as pipe tables, or return an empty string."""
    try:
        found = page.find_tables()
    except Exception:  # pragma: no cover - depends on the PDF
        return ""
    parts: list[str] = []
    for index, table in enumerate(found.tables):
        try:
            rows = table.extract()
        except Exception:  # pragma: no cover - depends on the PDF
            continue
        if not _table_is_useful(rows):
            continue
        try:
            markdown = table.to_markdown().strip()
        except Exception:  # pragma: no cover - depends on the PDF
            continue
        if markdown:
            parts.append(f"[table {index + 1}]\n{markdown}")
    return "\n\n".join(parts)


def _looks_like_slide_deck(doc: pymupdf.Document, page_lengths: Sequence[int]) -> bool:
    meta = doc.metadata or {}
    blob = " ".join(str(meta.get(key) or "") for key in ("creator", "producer", "title"))
    if _SLIDE_PRODUCERS.search(blob):
        return True
    if not page_lengths:
        return False
    return statistics.median(page_lengths) <= config.SLIDE_PAGE_MAX_CHARS / 2


def parse_pdf(path: str | Path, data: bytes | None = None) -> ParsedDocument:
    """Parse a PDF into pages with canonical text and resolvable offsets."""
    path = Path(path)
    if data is None:
        data = path.read_bytes()
    doc_id = compute_doc_id(data)

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        n_pages = doc.page_count
        plain_pages = [doc[i].get_text("text") for i in range(n_pages)]

        markdown_pages: list[str] = ["" for _ in range(n_pages)]
        warnings: list[dict[str, object]] = []
        try:
            rendered = pymupdf4llm.to_markdown(
                doc,
                pages=list(range(n_pages)),
                page_chunks=True,
                show_progress=False,
            )
            for i, item in enumerate(rendered):
                if i < n_pages:
                    markdown_pages[i] = (item.get("text") or "").strip()
        except Exception as exc:  # pragma: no cover - depends on the PDF
            warnings.append({"stage": "parse", "kind": "markdown_failed", "detail": str(exc)[:400]})

        pages: list[ParsedPage] = []
        buffer_parts: list[str] = []
        cursor = 0
        for index in range(n_pages):
            page = doc[index]
            plain = plain_pages[index].strip()
            markdown = markdown_pages[index]
            plain_alnum = _alnum_count(plain)
            retention = (_alnum_count(markdown) / plain_alnum) if plain_alnum else 1.0

            if markdown and retention >= MARKDOWN_MIN_RETENTION:
                canonical, source = markdown, "markdown"
            else:
                tables = _tables_markdown(page)
                canonical = f"{plain}\n\n{tables}" if tables else plain
                source = "text+tables" if tables else "text"

            n_images = len(page.get_images(full=True))
            if len(plain) < config.SPARSE_PAGE_CHAR_THRESHOLD:
                warnings.append(
                    {
                        "stage": "parse",
                        "kind": "image_only_page" if n_images else "empty_page",
                        "page_index": index,
                        "n_chars": len(plain),
                        "n_images": n_images,
                        "detail": "page yielded almost no extractable text and was not sent to OCR",
                    }
                )

            char_start = cursor
            char_end = char_start + len(canonical)
            cursor = char_end + len(PAGE_SEPARATOR)
            buffer_parts.append(canonical)
            buffer_parts.append(PAGE_SEPARATOR)

            pages.append(
                ParsedPage(
                    page_index=index,
                    page_label=detect_page_label(page),
                    text=canonical,
                    text_plain=plain,
                    char_start=char_start,
                    char_end=char_end,
                    width=page.rect.width,
                    height=page.rect.height,
                    n_images=n_images,
                    text_source=source,
                )
            )

        metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
        is_deck = _looks_like_slide_deck(doc, [len(p) for p in plain_pages])
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            path=str(path),
            n_pages=n_pages,
            buffer="".join(buffer_parts),
            pages=pages,
            metadata=metadata,
            is_slide_deck=is_deck,
            warnings=warnings,
        )
    finally:
        doc.close()


def find_bboxes(path: str | Path, page_index: int, quote: str) -> list[list[float]]:
    """Best effort bounding boxes for a quote on a page. Empty list is fine."""
    cleaned = strip_markdown(quote).strip()
    if not cleaned:
        return []
    needle = cleaned[: config.EVIDENCE_BBOX_SEARCH_CHARS].strip()
    try:
        doc = pymupdf.open(str(path))
    except Exception:  # pragma: no cover - depends on the file
        return []
    try:
        if page_index >= doc.page_count:
            return []
        rects = doc[page_index].search_for(needle)
        return [[round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)] for r in rects[:4]]
    except Exception:  # pragma: no cover - depends on the file
        return []
    finally:
        doc.close()


_MARKDOWN_NOISE = re.compile(r"(\*\*|__|~~|^#{1,6}\s+|\[table \d+\])", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove the decoration the markdown renderer adds, keeping the words."""
    return _MARKDOWN_NOISE.sub("", text).replace("|", " ").strip()


def page_lookup(pages: Iterable[ParsedPage]) -> list[tuple[int, int, int, str | None]]:
    """Compact ``(char_start, char_end, page_index, page_label)`` offset table."""
    return [(p.char_start, p.char_end, p.page_index, p.page_label) for p in pages]


def resolve_page(
    lookup: Sequence[tuple[int, int, int, str | None]], offset: int
) -> tuple[int, str | None] | None:
    """Binary search the offset table. Returns ``(page_index, page_label)``."""
    low, high = 0, len(lookup) - 1
    while low <= high:
        mid = (low + high) // 2
        start, end, index, label = lookup[mid]
        if offset < start:
            high = mid - 1
        elif offset > end:
            low = mid + 1
        else:
            return index, label
    return None
