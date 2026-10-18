from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.routes import manager
from app.config import settings
from app.main import app


client = TestClient(app)


def test_upload_rejects_files_exceeding_configured_limit(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        settings,
        "MAX_UPLOAD_SIZE_BYTES",
        10,
    )

    oversized_content = b"%PDF-1.4\nThis file is too large."

    with patch.object(manager, "add_document") as mock_add:
        response = client.post(
            "/upload",
            files={
                "file": (
                    "large.pdf",
                    oversized_content,
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Uploaded PDF exceeds the maximum allowed size."
    )

    mock_add.assert_not_called()

    upload_dir = Path("uploads")

    assert upload_dir.exists()
    assert list(upload_dir.iterdir()) == []

def test_upload_rejects_non_pdf_content_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    from pathlib import Path
    from unittest.mock import patch

    monkeypatch.chdir(tmp_path)

    with patch.object(manager, "add_document") as mock_add:
        response = client.post(
            "/upload",
            files={
                "file": (
                    "fake.pdf",
                    b"This is a text file, not a PDF.",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 400
    mock_add.assert_not_called()
    assert list(Path("uploads").iterdir()) == []

def test_upload_rejects_pdf_over_page_limit_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    from io import BytesIO
    from pathlib import Path
    from unittest.mock import patch

    from pypdf import PdfWriter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 1)

    # Create a structurally valid PDF containing two pages.
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)

    buffer = BytesIO()
    writer.write(buffer)

    with patch.object(manager, "add_document") as mock_add:
        response = client.post(
            "/upload",
            files={
                "file": (
                    "too_many_pages.pdf",
                    buffer.getvalue(),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 400
    assert "maximum allowed page count of 1" in response.json()["detail"]

    mock_add.assert_not_called()
    assert list(Path("uploads").iterdir()) == []
