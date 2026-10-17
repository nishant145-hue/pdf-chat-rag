
from unittest.mock import MagicMock, patch

import pytest

from app.services.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    with (
        patch("app.services.rag_pipeline.VectorStore") as mock_store_class,
        patch("app.services.rag_pipeline.Retriever") as mock_retriever_class,
        patch.object(RAGPipeline, "process_document"),
    ):
        store = mock_store_class.return_value
        store.is_loaded.return_value = True

        retriever = mock_retriever_class.return_value
        retriever.reranker = None

        instance = RAGPipeline("sample.pdf")

        yield instance


def test_pipeline_initializes_with_pdf_filename(pipeline):
    assert pipeline.filename == "sample.pdf"
    assert pipeline.vector_store is not None
    assert pipeline.retriever is not None


def test_ask_rejects_empty_question(pipeline):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("")


def test_ask_rejects_whitespace_question(pipeline):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("   ")


def test_ask_rejects_question_when_vector_store_is_not_loaded(pipeline):
    pipeline.vector_store.is_loaded.return_value = False

    with pytest.raises(RuntimeError, match="Vector store is not initialized"):
        pipeline.ask("What is this document about?")


@patch("app.services.rag_pipeline.generate_answer")
def test_ask_uses_retrieved_context(mock_generate_answer, pipeline):
    mock_generate_answer.return_value = "The document describes RAG."

    pipeline.retriever.retrieve.return_value = [
        "RAG combines retrieval and generation.",
        "The retriever searches relevant chunks.",
    ]

    answer = pipeline.ask("What is RAG?")

    assert answer == "The document describes RAG."
    mock_generate_answer.assert_called_once_with(
        question="What is RAG?",
        context=(
            "RAG combines retrieval and generation.\n\n"
            "The retriever searches relevant chunks."
        ),
    )


@patch("app.services.rag_pipeline.generate_answer")
def test_ask_returns_fallback_when_no_chunks_found(
    mock_generate_answer, pipeline
):
    pipeline.retriever.retrieve.return_value = []

    answer = pipeline.ask("What is RAG?")

    assert answer == "I couldn't find that in the uploaded document."
    mock_generate_answer.assert_not_called()


def test_ask_with_citations_rejects_empty_question(pipeline):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask_with_citations("")


def test_ask_with_citations_rejects_unloaded_vector_store(pipeline):
    pipeline.vector_store.is_loaded.return_value = False

    with pytest.raises(RuntimeError, match="Vector store is not initialized"):
        pipeline.ask_with_citations("What is RAG?")


@patch("app.services.rag_pipeline.create_embedding")
def test_ask_with_citations_returns_fallback_when_no_candidates(
    mock_create_embedding, pipeline
):
    mock_create_embedding.return_value = [0.1, 0.2]
    pipeline.vector_store.search_with_metadata.return_value = []

    result = pipeline.ask_with_citations("What is RAG?")

    assert result == {
        "answer": "I couldn't find that in the uploaded document.",
        "citations": [],
    }


@patch("app.services.rag_pipeline.generate_answer")
@patch("app.services.rag_pipeline.create_embedding")
def test_ask_with_citations_returns_source_metadata(
    mock_create_embedding, mock_generate_answer, pipeline
):
    mock_create_embedding.return_value = [0.1, 0.2]
    mock_generate_answer.return_value = "RAG retrieves relevant context."

    pipeline.vector_store.search_with_metadata.return_value = [
        {
            "text": "RAG retrieves relevant context.",
            "source": "sample.pdf",
            "page": 3,
        }
    ]

    pipeline.retriever.reranker = MagicMock()
    pipeline.retriever.reranker.rerank.return_value = [
        "RAG retrieves relevant context."
    ]

    result = pipeline.ask_with_citations("What is RAG?")

    assert result["answer"] == "RAG retrieves relevant context."
    assert result["citations"] == [
        {
            "source": "sample.pdf",
            "page": 3,
            "text": "RAG retrieves relevant context.",
        }
    ]


def test_total_chunks_delegates_to_vector_store(pipeline):
    pipeline.vector_store.total_chunks.return_value = 7

    assert pipeline.total_chunks() == 7


def test_filename_only_returns_pdf_filename(pipeline):
    assert pipeline.filename_only() == "sample.pdf"
