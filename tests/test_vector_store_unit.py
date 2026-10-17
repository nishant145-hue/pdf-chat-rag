import pytest

from app.services.vector_store import VectorStore


def test_build_index_stores_all_chunks():
    store = VectorStore()

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]

    chunks = [
        "Python introduction",
        "Machine learning",
        "Python programming",
    ]

    store.build_index(embeddings, chunks)

    assert store.is_loaded()
    assert store.total_chunks() == 3


def test_search_returns_nearest_chunks():
    store = VectorStore()

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]

    chunks = [
        "Python introduction",
        "Machine learning",
        "Python programming",
    ]

    store.build_index(embeddings, chunks)

    results = store.search([1.0, 0.0], k=2)

    assert results == [
        "Python introduction",
        "Python programming",
    ]


def test_search_without_index_raises_error():
    store = VectorStore()

    with pytest.raises(
        ValueError,
        match="Vector index has not been created",
    ):
        store.search([1.0, 0.0])


def test_search_returns_available_results_when_k_is_large():
    store = VectorStore()

    store.build_index(
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        chunks=[
            "First document",
            "Second document",
        ],
    )

    results = store.search([1.0, 0.0], k=10)

    assert len(results) == 2


def test_total_chunks_on_empty_store():
    store = VectorStore()

    assert store.total_chunks() == 0
    assert not store.is_loaded()


def test_build_index_rejects_empty_embeddings():
    store = VectorStore()

    with pytest.raises(ValueError):
        store.build_index([], [])

import math


def test_build_index_rejects_mismatched_counts():
    store = VectorStore()

    with pytest.raises(ValueError, match="must match"):
        store.build_index(
            embeddings=[[1.0, 0.0]],
            chunks=["First", "Second"],
        )


def test_search_rejects_wrong_dimensions():
    store = VectorStore()

    store.build_index(
        embeddings=[[1.0, 0.0]],
        chunks=["First document"],
    )

    with pytest.raises(ValueError, match="dimensions"):
        store.search([1.0, 0.0, 0.0])


def test_search_rejects_non_positive_k():
    store = VectorStore()

    store.build_index(
        embeddings=[[1.0, 0.0]],
        chunks=["First document"],
    )

    with pytest.raises(ValueError, match="positive integer"):
        store.search([1.0, 0.0], k=0)


def test_build_index_rejects_non_finite_values():
    store = VectorStore()

    with pytest.raises(ValueError, match="finite"):
        store.build_index(
            embeddings=[[math.nan, 0.0]],
            chunks=["Invalid vector"],
        )

def test_build_index_preserves_chunk_metadata():
    store = VectorStore()

    chunks = [
        {
            "text": "Python introduction",
            "source": "python.pdf",
            "page": 2,
        },
        {
            "text": "Machine learning introduction",
            "source": "ml.pdf",
            "page": 5,
        },
    ]

    store.build_index(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        chunks=chunks,
    )

    assert store.chunks == [
        "Python introduction",
        "Machine learning introduction",
    ]

    assert store.chunk_metadata == [
        {"source": "python.pdf", "page": 2},
        {"source": "ml.pdf", "page": 5},
    ]


def test_search_with_metadata_returns_source_and_page():
    store = VectorStore()

    store.build_index(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        chunks=[
            {
                "text": "Python introduction",
                "source": "python.pdf",
                "page": 2,
            },
            {
                "text": "Machine learning introduction",
                "source": "ml.pdf",
                "page": 5,
            },
        ],
    )

    results = store.search_with_metadata([1.0, 0.0], k=1)

    assert results == [
        {
            "text": "Python introduction",
            "source": "python.pdf",
            "page": 2,
        }
    ]


def test_build_index_supports_legacy_string_chunks():
    store = VectorStore()

    store.build_index(
        embeddings=[[1.0, 0.0]],
        chunks=["Legacy text chunk"],
    )

    assert store.search([1.0, 0.0]) == ["Legacy text chunk"]

    assert store.search_with_metadata([1.0, 0.0]) == [
        {
            "text": "Legacy text chunk",
            "source": None,
            "page": None,
        }
    ]


def test_build_index_rejects_invalid_page_metadata():
    store = VectorStore()

    with pytest.raises(ValueError, match="positive integer"):
        store.build_index(
            embeddings=[[1.0, 0.0]],
            chunks=[
                {
                    "text": "Some text",
                    "source": "sample.pdf",
                    "page": 0,
                }
            ],
        )
