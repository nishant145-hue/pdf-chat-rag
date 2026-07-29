from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string containing all extracted text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no readable text is found.
    """

    pdf_file = Path(pdf_path)

    # Check whether the file exists
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Open the PDF
    reader = PdfReader(pdf_file)

    extracted_pages = []

    # Read every page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        # Skip pages with no readable text
        if text and text.strip():
            extracted_pages.append(text)
        else:
            print(f"Warning: Page {page_number} contains no readable text.")

    # Combine all pages
    final_text = "\n".join(extracted_pages)

    if not final_text.strip():
        raise ValueError("No readable text found in the PDF.")

    return final_text