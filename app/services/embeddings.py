from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model on first use and cache it."""
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def create_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    embedding = get_embedding_model().encode(text)
    return embedding.tolist()


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple text chunks."""
    if not isinstance(chunks, list):
        raise TypeError("Chunks must be provided as a list.")

    if not chunks:
        return []

    if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("Every chunk must be a non-empty string.")

    embeddings = get_embedding_model().encode(chunks)
    return embeddings.tolist()
