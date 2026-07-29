import faiss
import numpy as np

from app.services.persistence import PersistenceService


class VectorStore:
    """
    Handles:

    - Building FAISS indexes
    - Searching vectors
    - Saving indexes
    - Loading indexes
    """

    def __init__(self):

        self.index = None

        self.chunks = []

        self.persistence = PersistenceService()

    # ---------------------------------------------------------
    # Build
    # ---------------------------------------------------------

    def build_index(
        self,
        embeddings: list[list[float]],
        chunks: list[str],
    ):

        vectors = np.array(
            embeddings,
            dtype=np.float32,
        )

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)

        self.chunks = chunks

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[str]:

        if self.index is None:
            raise ValueError("Vector index has not been created.")

        query = np.array(
            [query_embedding],
            dtype=np.float32,
        )

        distances, indices = self.index.search(
            query,
            k,
        )

        results = []

        for index in indices[0]:

            if 0 <= index < len(self.chunks):
                results.append(self.chunks[index])

        return results

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(
        self,
        filename: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):

        if self.index is None:
            raise ValueError("Nothing to save.")

        self.persistence.save_index(
            filename,
            self.index,
        )

        self.persistence.save_chunks(
            filename,
            self.chunks,
        )

        self.persistence.save_metadata(
            filename=filename,
            chunk_count=len(self.chunks),
            embedding_model=embedding_model,
        )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(
        self,
        filename: str,
    ) -> bool:

        if not self.persistence.exists(filename):
            return False

        self.index = self.persistence.load_index(filename)

        self.chunks = self.persistence.load_chunks(filename)

        return True

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(
        self,
        filename: str,
    ):

        self.persistence.delete(filename)

        self.index = None

        self.chunks = []

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def total_chunks(self) -> int:

        return len(self.chunks)

    def is_loaded(self) -> bool:

        return self.index is not None