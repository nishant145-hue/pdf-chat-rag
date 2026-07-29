def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:
    """
    Split text into chunks without breaking words.

    Args:
        text: Clean text.
        chunk_size: Maximum characters in one chunk.
        overlap: Number of overlapping characters.

    Returns:
        List of chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:

        word_length = len(word)

        if current_length + word_length + 1 <= chunk_size:

            current_chunk.append(word)
            current_length += word_length + 1

        else:

            chunks.append(" ".join(current_chunk))

            overlap_chunk = []
            overlap_length = 0

            for previous_word in reversed(current_chunk):

                if overlap_length + len(previous_word) + 1 > overlap:
                    break

                overlap_chunk.insert(0, previous_word)
                overlap_length += len(previous_word) + 1

            current_chunk = overlap_chunk
            current_chunk.append(word)

            current_length = (
                sum(len(w) + 1 for w in current_chunk)
            )

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks