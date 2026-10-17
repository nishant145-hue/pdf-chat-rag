
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.routes import manager, router
from app.main import app
from app.models.response_models import AnswerResponse, Citation


client = TestClient(app)


def test_answer_response_defaults_to_empty_citations():
    response = AnswerResponse(answer="This is an answer.")

    assert response.answer == "This is an answer."
    assert response.citations == []


def test_citation_model_accepts_source_metadata():
    citation = Citation(
        source="sample.pdf",
        page=3,
        text="This passage supports the answer.",
    )

    assert citation.source == "sample.pdf"
    assert citation.page == 3
    assert citation.text == "This passage supports the answer."


@patch("app.api.routes.manager.get_pipeline")
def test_ask_endpoint_returns_answer_and_citations(mock_get_pipeline):
    pipeline = MagicMock()
    pipeline.ask_with_citations.return_value = {
        "answer": "RAG retrieves relevant context.",
        "citations": [
            {
                "source": "sample.pdf",
                "page": 3,
                "text": "The retrieved passage supports the answer.",
            }
        ],
    }
    mock_get_pipeline.return_value = pipeline

    response = client.post(
        "/ask",
        json={
            "document_id": "test-document",
            "question": "What is RAG?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "RAG retrieves relevant context.",
        "citations": [
            {
                "source": "sample.pdf",
                "page": 3,
                "text": "The retrieved passage supports the answer.",
            }
        ],
    }

    pipeline.ask_with_citations.assert_called_once_with("What is RAG?")


@patch("app.api.routes.manager.get_pipeline")
def test_ask_endpoint_returns_empty_citations_when_none_found(
    mock_get_pipeline,
):
    pipeline = MagicMock()
    pipeline.ask_with_citations.return_value = {
        "answer": "I couldn't find that in the uploaded document.",
        "citations": [],
    }
    mock_get_pipeline.return_value = pipeline

    response = client.post(
        "/ask",
        json={
            "document_id": "test-document",
            "question": "What is RAG?",
        },
    )

    assert response.status_code == 200
    assert response.json()["citations"] == []


@patch("app.api.routes.manager.get_pipeline")
def test_ask_endpoint_returns_404_for_missing_document(
    mock_get_pipeline,
):
    mock_get_pipeline.return_value = None

    response = client.post(
        "/ask",
        json={
            "document_id": "missing-document",
            "question": "What is RAG?",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
