from app.config import settings
from app.services.chunker import split_text
from app.utils.helpers import clean_text


def chunk_pages(
    pages: list[dict],
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Split extracted PDF pages into text chunks while preserving
    the original source filename and page number.

    Each page must contain:
        {"page": int, "text": str}
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Source filename cannot be empty.")

    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE

    if overlap is None:
        overlap = settings.CHUNK_OVERLAP

    # Validate chunk settings even when pages is empty.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    if not isinstance(pages, list):
        raise TypeError("Pages must be provided as a list.")

    result = []
    seen_pages = set()

    for item in pages:
        if not isinstance(item, dict):
            raise ValueError("Every page must be a dictionary.")

        page_number = item.get("page")
        text = item.get("text")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError("Each page must have a positive integer page number.")

        if page_number in seen_pages:
            raise ValueError("Page numbers must be unique.")

        seen_pages.add(page_number)

        if not isinstance(text, str):
            raise ValueError("Page text must be a string.")

        cleaned_text = clean_text(text)

        if not cleaned_text:
            continue

        chunks = split_text(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            result.append(
                {
                    "text": chunk,
                    "source": source,
                    "page": page_number,
                }
            )

    return result
