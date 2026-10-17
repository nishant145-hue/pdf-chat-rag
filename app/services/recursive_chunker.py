
import re


def split_paragraphs(text: str) -> list[str]:
    return [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]


def split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]


def split_words(text: str) -> list[str]:
    return text.split()


def build_chunks(parts, chunk_size: int) -> list[str]:
    """
    Combine parts into chunks without exceeding chunk_size.
    Split individual oversized parts when necessary.
    """
    chunks = []
    current = ""

    for part in parts:
        if not part:
            continue

        # A single part may be longer than the maximum size.
        if len(part) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(
                part[start:start + chunk_size]
                for start in range(0, len(part), chunk_size)
            )
            continue

        candidate = f"{current}\n{part}" if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = part

    if current:
        chunks.append(current)

    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
) -> list[str]:
    """
    Split text hierarchically using paragraphs, sentences,
    and words while respecting the maximum character count.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if not text or not text.strip():
        return []

    final_chunks = []

    for paragraph in split_paragraphs(text.strip()):
        if len(paragraph) <= chunk_size:
            final_chunks.append(paragraph)
            continue

        sentences = split_sentences(paragraph)

        sentence_chunks = build_chunks(
            sentences,
            chunk_size,
        )

        for sentence_chunk in sentence_chunks:
            if len(sentence_chunk) <= chunk_size:
                final_chunks.append(sentence_chunk)
                continue

            words = split_words(sentence_chunk)

            word_chunks = build_chunks(
                words,
                chunk_size,
            )

            final_chunks.extend(word_chunks)

    return [
        chunk
        for chunk in final_chunks
        if chunk.strip()
    ]
