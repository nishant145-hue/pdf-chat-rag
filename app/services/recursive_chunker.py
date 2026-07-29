import re


def split_paragraphs(text: str):
    return text.split("\n\n")


def split_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text)


def split_words(text: str):
    return text.split()


def build_chunks(parts, chunk_size):

    chunks = []
    current = ""

    for part in parts:

        if len(current) + len(part) <= chunk_size:

            if current:
                current += "\n" + part
            else:
                current = part

        else:

            if current:
                chunks.append(current)

            current = part

    if current:
        chunks.append(current)

    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 500
):
    """
    Recursive text splitter.

    Priority:

    Paragraph
    ↓
    Sentence
    ↓
    Word
    """

    paragraphs = split_paragraphs(text)

    final_chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= chunk_size:

            final_chunks.append(paragraph)

            continue

        sentences = split_sentences(paragraph)

        sentence_chunks = build_chunks(
            sentences,
            chunk_size
        )

        for sentence_chunk in sentence_chunks:

            if len(sentence_chunk) <= chunk_size:

                final_chunks.append(sentence_chunk)

                continue

            words = split_words(sentence_chunk)

            word_chunks = build_chunks(
                words,
                chunk_size
            )

            final_chunks.extend(word_chunks)

    return final_chunks