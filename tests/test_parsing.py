"""Parsing tests. Synthetic PDFs only, no network, no starter data dependency."""

from __future__ import annotations

import pymupdf
import pytest

from app.parsing import (
    PAGE_SEPARATOR,
    ParsedPage,
    compute_doc_id,
    detect_page_label,
    page_lookup,
    parse_pdf,
    resolve_page,
    strip_markdown,
    _table_is_useful,
)


def _build_pdf(tmp_path, pages: list[tuple[str, str | None]], name: str = "sample.pdf"):
    """Write a PDF where each page has body text and an optional footer label."""
    doc = pymupdf.open()
    for body, footer in pages:
        page = doc.new_page(width=400, height=600)
        page.insert_text((50, 120), body, fontsize=11)
        if footer is not None:
            page.insert_text((190, 570), footer, fontsize=9)
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def test_doc_id_is_content_hash_and_stable(tmp_path):
    a = _build_pdf(tmp_path, [("hello world", "1")], "a.pdf")
    b = _build_pdf(tmp_path, [("hello world", "1")], "b.pdf")
    assert compute_doc_id(a.read_bytes()) == compute_doc_id(a.read_bytes())
    assert len(compute_doc_id(a.read_bytes())) == 64
    # Different filename, same bytes intent: content decides, not the name.
    assert compute_doc_id(b.read_bytes()) == compute_doc_id(b.read_bytes())


def test_page_offsets_resolve_into_the_buffer(tmp_path):
    path = _build_pdf(tmp_path, [("alpha page one", "1"), ("beta page two", "2")])
    parsed = parse_pdf(path)
    assert parsed.n_pages == 2
    for page in parsed.pages:
        assert parsed.buffer[page.char_start : page.char_end] == page.text
    first, second = parsed.pages
    assert second.char_start == first.char_end + len(PAGE_SEPARATOR)


def test_page_label_is_read_from_the_footer_and_is_not_the_index(tmp_path):
    path = _build_pdf(tmp_path, [("content", "17"), ("content two", "18")])
    parsed = parse_pdf(path)
    assert [p.page_label for p in parsed.pages] == ["17", "18"]
    assert [p.page_index for p in parsed.pages] == [0, 1]


def test_missing_page_label_is_none_not_invented(tmp_path):
    path = _build_pdf(tmp_path, [("no footer here", None)])
    parsed = parse_pdf(path)
    assert parsed.pages[0].page_label is None


def test_roman_page_label(tmp_path):
    path = _build_pdf(tmp_path, [("front matter", "xiv")])
    parsed = parse_pdf(path)
    assert parsed.pages[0].page_label == "xiv"


def test_body_number_is_not_mistaken_for_a_page_label(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 300), "revenue was 8142", fontsize=11)
    path = tmp_path / "body.pdf"
    doc.save(str(path))
    doc.close()
    parsed = parse_pdf(path)
    assert parsed.pages[0].page_label is None


def test_sparse_page_is_recorded_as_a_warning_not_silently_dropped(tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=400, height=600)  # entirely empty page
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 120), "this page has plenty of readable text on it", fontsize=11)
    path = tmp_path / "sparse.pdf"
    doc.save(str(path))
    doc.close()
    parsed = parse_pdf(path)
    kinds = {w["kind"] for w in parsed.warnings}
    assert kinds & {"empty_page", "image_only_page"}
    assert any(w["page_index"] == 0 for w in parsed.warnings)


def test_resolve_page_finds_the_owning_page():
    pages = [
        ParsedPage(0, "1", "aaa", "aaa", 0, 3, 400, 600, 0, "text"),
        ParsedPage(1, "2", "bbbb", "bbbb", 5, 9, 400, 600, 0, "text"),
    ]
    lookup = page_lookup(pages)
    assert resolve_page(lookup, 1) == (0, "1")
    assert resolve_page(lookup, 6) == (1, "2")
    assert resolve_page(lookup, 100) is None


def test_strip_markdown_keeps_words_and_drops_decoration():
    assert strip_markdown("## **Revenue** | 8,142 |") == "Revenue   8,142"
    assert "table" not in strip_markdown("[table 1]\n|a|b|")


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([["Item", "FY23", "FY24"], ["Revenue", "7,225", "8,142"]], True),
        ([["a"]], False),
        ([["ss Parc", "ss Parc"], ["ss Parc", "ss Parc"]], False),
        ([["a", None], [None, None]], False),
    ],
)
def test_table_quality_filter(rows, expected):
    assert _table_is_useful(rows) is expected
