import json

import pytest

from app.services.persistence import PersistenceService
from app.services.vector_store import VectorStore


@pytest.fixture
def persistence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return PersistenceService()


def test_save_and_load_chunk_records(persistence):
    records = [
        {
            "text": "Python introduction",
            "source": "python.pdf",
            "page": 2,
        },
        {
            "text": "Python functions",
            "source": "python.pdf",
            "page": 3,
        },
    ]

    persistence.save_chunks("python.pdf", records)

    loaded = persistence.load_chunks("python.pdf")

    assert loaded == records


def test_save_and_load_metadata(persistence):
    persistence.save_metadata(
        filename="python.pdf",
        chunk_count=2,
        embedding_model="all-MiniLM-L6-v2",
    )

    metadata = persistence.load_metadata("python.pdf")

    assert metadata["filename"] == "python.pdf"
    assert metadata["chunk_count"] == 2
    assert metadata["embedding_model"] == "all-MiniLM-L6-v2"


def test_exists_requires_all_persisted_files(persistence):
    assert persistence.exists("missing.pdf") is False

    persistence.save_chunks("sample.pdf", ["Sample text"])

    assert persistence.exists("sample.pdf") is False


def test_delete_removes_persisted_files(persistence):
    persistence.save_chunks("sample.pdf", ["Sample text"])
    persistence.save_metadata(
        filename="sample.pdf",
        chunk_count=1,
        embedding_model="all-MiniLM-L6-v2",
    )

    # The index file is required for exists() to return True.
    import faiss
    import numpy as np

    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    persistence.save_index("sample.pdf", index)

    assert persistence.exists("sample.pdf") is True

    persistence.delete("sample.pdf")

    assert persistence.exists("sample.pdf") is False


def test_vector_store_round_trips_metadata(persistence):
    store = VectorStore()
    store.persistence = persistence

    store.build_index(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        chunks=[
            {
                "text": "Python introduction",
                "source": "python.pdf",
                "page": 2,
            },
            {
                "text": "Machine learning",
                "source": "ml.pdf",
                "page": 5,
            },
        ],
    )

    store.save("python.pdf")

    restored = VectorStore()
    restored.persistence = persistence

    assert restored.load("python.pdf") is True
    assert restored.total_chunks() == 2

    results = restored.search_with_metadata([1.0, 0.0], k=1)

    assert results == [
        {
            "text": "Python introduction",
            "source": "python.pdf",
            "page": 2,
        }
    ]


def test_legacy_string_chunks_remain_loadable(persistence):
    store = VectorStore()
    store.persistence = persistence

    store.build_index(
        embeddings=[[1.0, 0.0]],
        chunks=["Legacy text"],
    )

    store.save("legacy.pdf")

    # Rewrite chunks in the original string-only format.
    persistence.save_chunks("legacy.pdf", ["Legacy text"])

    restored = VectorStore()
    restored.persistence = persistence

    assert restored.load("legacy.pdf") is True
    assert restored.search([1.0, 0.0]) == ["Legacy text"]

    assert restored.search_with_metadata([1.0, 0.0]) == [
        {
            "text": "Legacy text",
            "source": None,
            "page": None,
        }
    ]
