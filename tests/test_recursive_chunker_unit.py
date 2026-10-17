from app.services.recursive_chunker import (
    split_paragraphs,
    split_sentences,
    split_words,
    recursive_chunk,
)


def test_split_paragraphs():
    text = "First paragraph\n\nSecond paragraph"

    assert split_paragraphs(text) == [
        "First paragraph",
        "Second paragraph",
    ]


def test_split_sentences():
    text = "Python is useful. AI is growing."

    assert split_sentences(text) == [
        "Python is useful.",
        "AI is growing.",
    ]


def test_split_words():
    assert split_words("Python is useful") == [
        "Python",
        "is",
        "useful",
    ]


def test_short_text_returns_single_chunk():
    text = "Python is useful."

    assert recursive_chunk(text, chunk_size=100) == [text]


def test_long_text_is_split():
    text = " ".join(
        ["Python is useful for AI development."] * 10
    )

    chunks = recursive_chunk(text, chunk_size=50)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunks_respect_maximum_size():
    text = (
        "Python is useful. "
        "Machine learning is interesting. "
        "Artificial intelligence is growing."
    )

    chunks = recursive_chunk(text, chunk_size=30)

    assert all(len(chunk) <= 30 for chunk in chunks)