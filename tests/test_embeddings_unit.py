import pytest

from unittest.mock import MagicMock, patch

from app.services.embeddings import (
    create_embedding,
    create_embeddings,
    get_embedding_model,
)


@pytest.fixture(autouse=True)
def clear_embedding_model_cache():
    """Prevent cached models from leaking between tests."""
    get_embedding_model.cache_clear()
    yield
    get_embedding_model.cache_clear()


@patch("app.services.embeddings.get_embedding_model")
def test_create_embedding_returns_list(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [
        0.1,
        0.2,
        0.3,
    ]
    mock_get_model.return_value = mock_model

    result = create_embedding("Python is useful")

    assert result == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with(
        "Python is useful"
    )


@patch("app.services.embeddings.get_embedding_model")
def test_create_embeddings_returns_multiple_vectors(
    mock_get_model,
):
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    mock_get_model.return_value = mock_model

    result = create_embeddings([
        "First document",
        "Second document",
    ])

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_create_embedding_rejects_empty_text():
    with pytest.raises(ValueError):
        create_embedding("")


def test_create_embedding_rejects_whitespace():
    with pytest.raises(ValueError):
        create_embedding("   ")


def test_create_embeddings_returns_empty_list_for_empty_input():
    assert create_embeddings([]) == []


@pytest.mark.parametrize(
    "chunks",
    [
        ["valid text", ""],
        ["valid text", "   "],
    ],
)
def test_create_embeddings_rejects_empty_chunks(chunks):
    with pytest.raises(ValueError):
        create_embeddings(chunks)


def test_create_embeddings_rejects_non_list_input():
    with pytest.raises(TypeError):
        create_embeddings("not a list")