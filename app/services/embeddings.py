from sentence_transformers import SentenceTransformer
from app.config import settings

model = SentenceTransformer(settings.EMBEDDING_MODEL)

def create_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text chunk.

    Args:
        text: Input text.

    Returns:
        Embedding vector.
    """

    embedding = model.encode(text)

    return embedding.tolist()


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple chunks.

    Args:
        chunks: List of text chunks.

    Returns:
        List of embedding vectors.
    """

    embeddings = model.encode(chunks)

    return embeddings.tolist()