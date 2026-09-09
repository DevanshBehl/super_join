"""Chunking tests: offsets stay exact and tables are never split."""

from __future__ import annotations

import config
from app.chunking import chunk_page, chunk_document, make_chunk_id, normalize_for_hash
from app.parsing import ParsedDocument, ParsedPage

TABLE = "\n".join(
    ["| Item | FY23 | FY24 |", "|---|---|---|", "| Revenue | 7,225 | 8,142 |", "| EBITDA | (452) | 127 |"]
)


def _page(text: str, index: int = 0, char_start: int = 0) -> ParsedPage:
    return ParsedPage(
        page_index=index,
        page_label=str(index + 1),
        text=text,
        text_plain=text,
        char_start=char_start,
        char_end=char_start + len(text),
        width=595.0,
        height=842.0,
        n_images=0,
        text_source="markdown",
    )


def test_short_page_is_one_chunk():
    page = _page("A short paragraph about revenue of 8,142 crore in FY24.")
    chunks = chunk_page(page, "doc")
    assert len(chunks) == 1
    assert chunks[0].kind == "prose"
    assert chunks[0].char_start == 0


def test_chunk_offsets_are_absolute_and_exact():
    body = ("Sentence number one about tonnage. " * 200).strip()
    page = _page(body, index=3, char_start=5000)
    chunks = chunk_page(page, "doc")
    assert len(chunks) > 1
    for chunk in chunks:
        local = body[chunk.char_start - 5000 : chunk.char_end - 5000]
        assert local == chunk.text
        assert chunk.page_index_start == 3


def test_overlap_is_applied_between_prose_chunks():
    body = ("word " * 3000).strip()
    page = _page(body)
    chunks = chunk_page(page, "doc", target=1000, overlap=200)
    assert len(chunks) > 2
    gaps = [chunks[i + 1].char_start - chunks[i].char_end for i in range(len(chunks) - 1)]
    assert all(gap < 0 for gap in gaps), "consecutive chunks must overlap, not leave gaps"


def test_table_is_never_split_across_chunks():
    body = ("Prose before the table. " * 120) + "\n\n" + TABLE + "\n\n" + ("Prose after. " * 120)
    page = _page(body)
    chunks = chunk_page(page, "doc", target=600, overlap=100)
    holders = [c for c in chunks if "| Revenue | 7,225 | 8,142 |" in c.text]
    assert len(holders) >= 1
    for chunk in holders:
        assert "| EBITDA | (452) | 127 |" in chunk.text, "table rows were separated"


def test_table_only_page_is_tagged_table():
    page = _page(TABLE)
    chunks = chunk_page(page, "doc")
    assert chunks[0].kind == "table"


def test_mixed_page_is_tagged_mixed():
    page = _page("Some prose about the year. " * 12 + "\n" + TABLE)
    chunks = chunk_page(page, "doc")
    assert any(c.kind == "mixed" for c in chunks)


def test_slide_deck_keeps_one_chunk_per_page():
    text = "Slide headline. " * 150  # longer than the prose target
    assert len(text) > config.CHUNK_TARGET_CHARS
    page = _page(text)
    assert len(chunk_page(page, "doc", whole_page=True)) == 1
    assert len(chunk_page(page, "doc", whole_page=False)) > 1


def test_chunk_id_is_content_addressed_and_whitespace_insensitive():
    assert make_chunk_id("d", "Revenue 8,142") == make_chunk_id("d", " revenue   8,142 ")
    assert make_chunk_id("d", "a") != make_chunk_id("e", "a")
    assert normalize_for_hash("  A  B\n") == "a b"


def test_identical_pages_deduplicate_within_a_document():
    pages = [_page("Repeated boilerplate page.", 0, 0), _page("Repeated boilerplate page.", 1, 100)]
    doc = ParsedDocument(
        doc_id="doc",
        filename="f.pdf",
        path="f.pdf",
        n_pages=2,
        buffer="",
        pages=pages,
        metadata={},
        is_slide_deck=False,
    )
    assert len(chunk_document(doc)) == 1
