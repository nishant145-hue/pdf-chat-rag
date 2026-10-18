
from unittest.mock import MagicMock, patch

import pytest

from app.services.pdf_reader import validate_pdf_file


def test_validate_pdf_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"This is not a PDF.")

    with pytest.raises(ValueError, match="not a valid PDF"):
        validate_pdf_file(str(path))


def test_validate_pdf_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        validate_pdf_file(str(path))


def test_validate_pdf_accepts_parseable_pdf(tmp_path):
    path = tmp_path / "valid.pdf"
    path.write_bytes(b"%PDF-1.7\n")

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [MagicMock()]

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        validate_pdf_file(str(path))


def test_validate_pdf_rejects_pdf_without_pages(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.7\n")

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = []

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        with pytest.raises(ValueError, match="contains no pages"):
            validate_pdf_file(str(path))

def test_validate_pdf_accepts_document_at_page_limit(
    tmp_path,
    monkeypatch,
):
    from unittest.mock import MagicMock, patch

    from app.config import settings

    path = tmp_path / "at_limit.pdf"
    path.write_bytes(b"%PDF-1.7\n")

    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 3)

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [MagicMock() for _ in range(3)]

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        validate_pdf_file(str(path))


def test_validate_pdf_rejects_document_over_page_limit(
    tmp_path,
    monkeypatch,
):
    from unittest.mock import MagicMock, patch

    from app.config import settings

    path = tmp_path / "over_limit.pdf"
    path.write_bytes(b"%PDF-1.7\n")

    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 3)

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [MagicMock() for _ in range(4)]

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        with pytest.raises(
            ValueError,
            match="maximum allowed page count of 3",
        ):
            validate_pdf_file(str(path))


def test_validate_pdf_rejects_malformed_pdf(tmp_path):
    path = tmp_path / "malformed.pdf"
    path.write_bytes(
        b"%PDF-1.7\n"
        b"This is not a valid PDF structure."
    )

    with pytest.raises(ValueError, match="not a valid PDF"):
        validate_pdf_file(str(path))

def test_validate_pdf_rejects_encrypted_pdf_without_password(tmp_path):
    from io import BytesIO

    from pypdf import PdfWriter

    path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("test-password")

    with path.open("wb") as pdf_file:
        writer.write(pdf_file)

    with pytest.raises(ValueError, match="not a valid PDF"):
        validate_pdf_file(str(path))
