import pytest

from app.services.chunker import split_text
from app.services.recursive_chunker import recursive_chunk


def test_basic_chunker_handles_oversized_word():
    chunks = split_text(
        "extraordinary",
        chunk_size=5,
        overlap=1,
    )

    assert all(chunks)
    assert all(len(chunk) <= 5 for chunk in chunks)
    assert "".join(chunks) == "extraordinary"


def test_recursive_chunker_handles_empty_text():
    assert recursive_chunk("") == []


def test_recursive_chunker_handles_whitespace():
    assert recursive_chunk("   \n\n   ") == []


def test_recursive_chunker_handles_oversized_word():
    chunks = recursive_chunk(
        "extraordinary",
        chunk_size=5,
    )

    assert all(chunks)
    assert all(len(chunk) <= 5 for chunk in chunks)
    assert "".join(chunks) == "extraordinary"


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_recursive_chunker_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        recursive_chunk(
            "Some text",
            chunk_size=chunk_size,
        )