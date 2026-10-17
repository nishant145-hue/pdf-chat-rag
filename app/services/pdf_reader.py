from pathlib import Path

from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract readable text page by page.

    Returns:
        A list of dictionaries containing the original page number
        and extracted text.

    Example:
        [
            {"page": 1, "text": "Introduction to Python..."},
            {"page": 2, "text": "Variables and data types..."},
        ]

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF contains no readable text.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

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

    if not extracted_pages:
        raise ValueError(
            "No readable text found in the PDF."
        )

    return extracted_pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all readable PDF text into a single string.

    Retained for compatibility with existing callers.
    """

    pages = extract_pages_from_pdf(pdf_path)

    return "\n".join(
        page["text"]
        for page in pages
    )
