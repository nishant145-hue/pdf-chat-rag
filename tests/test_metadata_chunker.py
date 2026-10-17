import pytest

from app.services.metadata_chunker import chunk_pages


def test_chunk_pages_preserves_source_and_page_numbers():
    pages = [
        {"page": 1, "text": "Python is a programming language."},
        {"page": 2, "text": "Python supports functions and classes."},
    ]

    chunks = chunk_pages(
        pages,
        source="pythonlearn.pdf",
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 2

    assert chunks[0] == {
        "text": "Python is a programming language.",
        "source": "pythonlearn.pdf",
        "page": 1,
    }

    assert chunks[1] == {
        "text": "Python supports functions and classes.",
        "source": "pythonlearn.pdf",
        "page": 2,
    }


def test_chunk_pages_skips_empty_pages_without_renumbering():
    pages = [
        {"page": 1, "text": "First readable page."},
        {"page": 2, "text": "   "},
        {"page": 3, "text": "Third readable page."},
    ]

    chunks = chunk_pages(
        pages,
        source="sample.pdf",
        chunk_size=100,
        overlap=10,
    )

    assert [chunk["page"] for chunk in chunks] == [1, 3]


def test_chunk_pages_splits_long_pages_and_preserves_metadata():
    pages = [
        {"page": 4, "text": "one two three four five six seven eight"},
    ]

    chunks = chunk_pages(
        pages,
        source="sample.pdf",
        chunk_size=12,
        overlap=0,
    )

    assert len(chunks) > 1
    assert all(chunk["page"] == 4 for chunk in chunks)
    assert all(chunk["source"] == "sample.pdf" for chunk in chunks)
    assert all(len(chunk["text"]) <= 12 for chunk in chunks)


def test_chunk_pages_returns_empty_list_for_no_pages():
    assert chunk_pages([], source="sample.pdf") == []


@pytest.mark.parametrize("source", ["", "   "])
def test_chunk_pages_rejects_empty_source(source):
    with pytest.raises(ValueError, match="Source filename"):
        chunk_pages([], source=source)


def test_chunk_pages_rejects_invalid_page_number():
    with pytest.raises(ValueError, match="positive integer"):
        chunk_pages(
            [{"page": 0, "text": "Some text"}],
            source="sample.pdf",
        )


def test_chunk_pages_rejects_duplicate_page_numbers():
    pages = [
        {"page": 1, "text": "First text."},
        {"page": 1, "text": "Duplicate page."},
    ]

    with pytest.raises(ValueError, match="unique"):
        chunk_pages(pages, source="sample.pdf")


def test_chunk_pages_rejects_non_string_page_text():
    with pytest.raises(ValueError, match="Page text"):
        chunk_pages(
            [{"page": 1, "text": None}],
            source="sample.pdf",
        )


def test_chunk_pages_rejects_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pages(
            [{"page": 1, "text": "Some text."}],
            source="sample.pdf",
            chunk_size=0,
            overlap=0,
        )
