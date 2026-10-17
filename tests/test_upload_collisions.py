from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.routes import manager
from app.main import app


client = TestClient(app)


def test_uploads_with_same_filename_are_stored_separately(
    tmp_path,
    monkeypatch,
):
    """Uploading identical filenames must not overwrite existing files."""

    monkeypatch.chdir(tmp_path)

    with patch.object(
        manager,
        "add_document",
        side_effect=["document-1", "document-2"],
    ) as mock_add_document:

        first_response = client.post(
            "/upload",
            files={
                "file": (
                    "report.pdf",
                    b"%PDF-1.4\nFirst document",
                    "application/pdf",
                )
            },
        )

        second_response = client.post(
            "/upload",
            files={
                "file": (
                    "report.pdf",
                    b"%PDF-1.4\nSecond document",
                    "application/pdf",
                )
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    # The API must preserve the original filename.
    assert first_data["filename"] == "report.pdf"
    assert second_data["filename"] == "report.pdf"

    # Each upload must receive a different document ID.
    assert first_data["document_id"] != second_data["document_id"]

    # Both uploads must be processed independently.
    assert mock_add_document.call_count == 2

    first_call = mock_add_document.call_args_list[0].kwargs
    second_call = mock_add_document.call_args_list[1].kwargs

    first_path = Path(first_call["pdf_path"])
    second_path = Path(second_call["pdf_path"])

    # The physical files must have different paths.
    assert first_path != second_path

    # Neither upload may overwrite the other.
    assert first_path.exists()
    assert second_path.exists()

    assert first_path.read_bytes() == b"%PDF-1.4\nFirst document"
    assert second_path.read_bytes() == b"%PDF-1.4\nSecond document"

    # Both pipelines must receive the original filename.
    assert first_call["filename"] == "report.pdf"
    assert second_call["filename"] == "report.pdf"