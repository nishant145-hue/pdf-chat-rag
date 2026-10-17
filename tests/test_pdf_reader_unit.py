from unittest.mock import MagicMock, patch

import pytest

from app.services.pdf_reader import (
    extract_pages_from_pdf,
    extract_text_from_pdf,
)


def test_extract_pages_preserves_original_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()

    mock_reader = MagicMock()
    mock_reader.pages = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    mock_reader.pages[0].extract_text.return_value = "First page"
    mock_reader.pages[1].extract_text.return_value = "   "
    mock_reader.pages[2].extract_text.return_value = "Third page"

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        pages = extract_pages_from_pdf(str(pdf_path))

    assert pages == [
        {"page": 1, "text": "First page"},
        {"page": 3, "text": "Third page"},
    ]


def test_extract_pages_rejects_missing_file(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pages_from_pdf(str(pdf_path))


def test_extract_pages_rejects_pdf_without_readable_text(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()

    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock(), MagicMock()]

    for page in mock_reader.pages:
        page.extract_text.return_value = ""

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        with pytest.raises(ValueError, match="No readable text"):
            extract_pages_from_pdf(str(pdf_path))


def test_extract_text_remains_compatible(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()

    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock(), MagicMock()]
    mock_reader.pages[0].extract_text.return_value = "First page"
    mock_reader.pages[1].extract_text.return_value = "Second page"

    with patch(
        "app.services.pdf_reader.PdfReader",
        return_value=mock_reader,
    ):
        text = extract_text_from_pdf(str(pdf_path))

    assert text == "First page\nSecond page"
