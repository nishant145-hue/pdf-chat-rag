import pytest

from app.services.chunker import split_text


def test_empty_text_returns_empty_list():
    assert split_text("") == []


def test_whitespace_returns_empty_list():
    assert split_text("   \n\t  ") == []


def test_short_text_returns_single_chunk():
    text = "Python is useful for AI"

    chunks = split_text(text, chunk_size=100, overlap=10)

    assert chunks == [text]


def test_text_is_split_into_multiple_chunks():
    text = " ".join(["python"] * 20)

    chunks = split_text(
        text,
        chunk_size=30,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunks_respect_maximum_size_for_normal_words():
    text = " ".join(["python"] * 20)

    chunks = split_text(
        text,
        chunk_size=30,
        overlap=5,
    )

    assert all(len(chunk) <= 30 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_configuration_raises_error(
    chunk_size,
    overlap,
):
    with pytest.raises(ValueError):
        split_text(
            "Some sample text",
            chunk_size=chunk_size,
            overlap=overlap,
        )