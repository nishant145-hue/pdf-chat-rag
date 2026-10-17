from unittest.mock import MagicMock, patch

import pytest

from app.services.retriever import Retriever


@pytest.fixture
def vector_store():
    store = MagicMock()
    store.is_loaded.return_value = True
    store.search.return_value = [
        "First candidate",
        "Second candidate",
        "Third candidate",
    ]
    return store


def test_retrieve_returns_reranked_chunks(vector_store):
    reranker = MagicMock()
    reranker.rerank.return_value = [
        "Third candidate",
        "First candidate",
    ]

    retriever = Retriever(vector_store, reranker=reranker)

    with patch(
        "app.services.retriever.create_embedding",
        return_value=[0.1, 0.2],
    ):
        results = retriever.retrieve("What is relevant?")

    assert results == ["Third candidate", "First candidate"]
    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        k=20,
    )
    reranker.rerank.assert_called_once()


def test_retrieve_skips_reranking_when_disabled(vector_store):
    retriever = Retriever(vector_store)

    with (
        patch("app.services.retriever.settings.ENABLE_RERANKING", False),
        patch(
            "app.services.retriever.create_embedding",
            return_value=[0.1, 0.2],
        ),
    ):
        results = retriever.retrieve("Find relevant information", top_k=2)

    assert results == ["First candidate", "Second candidate"]


def test_retrieve_rejects_empty_question(vector_store):
    retriever = Retriever(vector_store)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("   ")


def test_retrieve_rejects_uninitialized_vector_store(vector_store):
    vector_store.is_loaded.return_value = False
    retriever = Retriever(vector_store)

    with pytest.raises(RuntimeError, match="not initialized"):
        retriever.retrieve("A valid question")


def test_retrieve_returns_empty_when_no_candidates(vector_store):
    vector_store.search.return_value = []
    retriever = Retriever(vector_store)

    with (
        patch(
            "app.services.retriever.create_embedding",
            return_value=[0.1, 0.2],
        ),
        patch("app.services.retriever.settings.ENABLE_RERANKING", False),
    ):
        results = retriever.retrieve("A valid question")

    assert results == []


def test_retrieve_falls_back_when_reranking_fails(vector_store):
    reranker = MagicMock()
    reranker.rerank.side_effect = RuntimeError("Reranker unavailable")

    retriever = Retriever(vector_store, reranker=reranker)

    with patch(
        "app.services.retriever.create_embedding",
        return_value=[0.1, 0.2],
    ):
        results = retriever.retrieve("A valid question")

    assert results == [
        "First candidate",
        "Second candidate",
        "Third candidate",
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(vector_store, top_k):
    retriever = Retriever(vector_store)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("A valid question", top_k=top_k)


@pytest.mark.parametrize("candidate_k", [0, -1])
def test_retrieve_rejects_invalid_candidate_k(vector_store, candidate_k):
    retriever = Retriever(vector_store)

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.retrieve("A valid question", candidate_k=candidate_k)
