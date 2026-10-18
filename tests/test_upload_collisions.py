from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.api.routes import manager
from app.main import app

client = TestClient(app)


def create_test_pdf(text: str) -> bytes:
    """Create a valid PDF with identifiable test metadata."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": text})


    buffer = BytesIO()
    writer.write(buffer)

    return buffer.getvalue()




def test_uploads_with_same_filename_are_stored_separately(
    tmp_path,
    monkeypatch,
):
    """Uploading identical filenames must not overwrite existing files."""
    monkeypatch.chdir(tmp_path)

    first_pdf = create_test_pdf("First document")
    second_pdf = create_test_pdf("Second document")

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
                    first_pdf,
                    "application/pdf",
                )
            },
        )

        second_response = client.post(
            "/upload",
            files={
                "file": (
                    "report.pdf",
                    second_pdf,
                    "application/pdf",
                )
            },
        )


    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    # The original filename should be preserved in the API response.
    assert first_data["filename"] == "report.pdf"
    assert second_data["filename"] == "report.pdf"

    # Each upload should receive a different document ID.
    assert first_data["document_id"] != second_data["document_id"]

    # Both uploads should be processed independently.
    assert mock_add_document.call_count == 2

    first_call = mock_add_document.call_args_list[0].kwargs
    second_call = mock_add_document.call_args_list[1].kwargs

    first_path = Path(first_call["pdf_path"])
    second_path = Path(second_call["pdf_path"])

    # Both uploads must be stored separately.
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    # The original filenames passed to the document manager are preserved.
    assert first_call["filename"] == "report.pdf"
    assert second_call["filename"] == "report.pdf"

    # The two uploaded files must not overwrite one another.
    assert first_path.read_bytes() != second_path.read_bytes()
