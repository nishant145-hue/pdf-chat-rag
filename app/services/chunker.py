
def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into chunks without breaking normal words.

    Words longer than chunk_size are split into character segments.
    Overlap is preserved between chunks where possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Split an oversized word into segments that fit.
        if len(word) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            for start in range(0, len(word), chunk_size):
                chunks.append(word[start:start + chunk_size])

            continue

        required_length = len(word) + (
            1 if current_chunk else 0
        )

        if current_length + required_length <= chunk_size:
            current_chunk.append(word)
            current_length += required_length
            continue

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Preserve whole-word overlap where possible.
        overlap_words = []
        overlap_length = 0

        for previous_word in reversed(current_chunk):
            required = len(previous_word) + (
                1 if overlap_words else 0
            )

            if overlap_length + required > overlap:
                break

            overlap_words.insert(0, previous_word)
            overlap_length += required

        current_chunk = overlap_words
        current_length = overlap_length

        required_length = len(word) + (
            1 if current_chunk else 0
        )

        # If overlap prevents the word from fitting, discard it.
        if current_length + required_length > chunk_size:
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += len(word) + (
            1 if len(current_chunk) > 1 else 0
        )

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
