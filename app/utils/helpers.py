import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove leading/trailing whitespace from the whole text
    text = text.strip()

    # Remove empty lines
    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)