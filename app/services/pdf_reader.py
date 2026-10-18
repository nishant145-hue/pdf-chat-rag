
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import (
    FileNotDecryptedError,
    PdfReadError,
)

from app.config import settings


def validate_pdf_file(pdf_path: str) -> None:
    """
    Validate that a file has a PDF signature and can be parsed.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid, readable PDF.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with pdf_file.open("rb") as file:
        signature = file.read(5)

    if signature != b"%PDF-":
        raise ValueError("Uploaded file is not a valid PDF.")

    try:
        reader = PdfReader(pdf_file, strict=True)

        if reader.is_encrypted:
            raise ValueError(
                "Uploaded file is not a valid PDF: "
                "password-protected PDFs are not supported."
            )

        page_count = len(reader.pages)

        if page_count == 0:
            raise ValueError("PDF contains no pages.")

        if page_count > settings.MAX_PDF_PAGES:
            raise ValueError(
                f"PDF exceeds the maximum allowed page count "
                f"of {settings.MAX_PDF_PAGES} pages."
            )

    except ValueError:
        raise

    except (PdfReadError, FileNotDecryptedError, OSError) as exc:
        raise ValueError(
            "Uploaded file is not a valid PDF."
        ) from exc


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract readable text page by page.

    Returns:
        A list of dictionaries containing the original page number
        and extracted text.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_file)
        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                extracted_pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

    except (
        PdfReadError,
        FileNotDecryptedError,
        OSError,
    ) as exc:
        raise ValueError("Unable to read the PDF file.") from exc

    if not extracted_pages:
        raise ValueError("No readable text found in the PDF.")

    return extracted_pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all readable PDF text into a single string.

    Retained for compatibility with existing callers.
    """
    pages = extract_pages_from_pdf(pdf_path)

    return "\n".join(page["text"] for page in pages)
