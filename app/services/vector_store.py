import faiss
import numpy as np

from app.services.persistence import PersistenceService


class VectorStore:
    """FAISS vector index with optional source metadata for each chunk."""

    def __init__(self):
        self.index = None
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []
        self.persistence = PersistenceService()

    def build_index(
        self,
        embeddings: list[list[float]],
        chunks: list[str] | list[dict],
    ) -> None:
        """Build a FAISS index from text chunks or metadata records."""

        if len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty.")

        if len(chunks) == 0:
            raise ValueError("Chunks cannot be empty.")

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        texts = []
        metadata = []

        for chunk in chunks:
            if isinstance(chunk, str):
                text = chunk
                details = {"source": None, "page": None}

            elif isinstance(chunk, dict):
                text = chunk.get("text")
                source = chunk.get("source")
                page = chunk.get("page")

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        "Chunk records must contain non-empty text."
                    )

                if source is not None and not isinstance(source, str):
                    raise ValueError("Chunk source must be a string or None.")

                if page is not None and (
                    not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                ):
                    raise ValueError(
                        "Chunk page must be a positive integer or None."
                    )

                details = {"source": source, "page": page}

            else:
                raise ValueError(
                    "Each chunk must be a string or a metadata dictionary."
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunks must contain non-empty strings.")

            texts.append(text)
            metadata.append(details)

        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Embeddings must be numeric vectors.") from exc

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimensions cannot be empty.")

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings must contain only finite values."
            )

        new_index = faiss.IndexFlatL2(vectors.shape[1])
        new_index.add(vectors)

        self.index = new_index
        self.chunks = texts
        self.chunk_metadata = metadata

    def _search_indices(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[int]:
        """Validate a query and return nearest chunk indices."""

        if self.index is None:
            raise ValueError("Vector index has not been created.")

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer.")

        try:
            query = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Query embedding must be numeric.") from exc

        if query.ndim != 1 or query.size == 0:
            raise ValueError(
                "Query embedding must be a non-empty vector."
            )

        if query.shape[0] != self.index.d:
            raise ValueError(
                "Query embedding dimensions must match the index dimensions."
            )

        if not np.isfinite(query).all():
            raise ValueError(
                "Query embedding must contain finite values."
            )

        if self.index.ntotal == 0:
            return []

        query = query.reshape(1, -1)
        search_k = min(k, self.index.ntotal)

        _, indices = self.index.search(query, search_k)

        return [
            int(index)
            for index in indices[0]
            if 0 <= index < len(self.chunks)
        ]

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[str]:
        """Return nearest chunks as plain strings for existing callers."""

        indices = self._search_indices(query_embedding, k)
        return [self.chunks[index] for index in indices]

    def search_with_metadata(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[dict]:
        """Return nearest chunks together with their source metadata."""

        indices = self._search_indices(query_embedding, k)

        return [
            {
                "text": self.chunks[index],
                "source": self.chunk_metadata[index].get("source"),
                "page": self.chunk_metadata[index].get("page"),
            }
            for index in indices
        ]

    def save(
        self,
        filename: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        original_filename: str | None = None,
        stored_filename: str | None = None,
    ) -> None:
        """Persist the index, chunks, and document metadata."""

        if self.index is None:
            raise ValueError("Nothing to save.")

        self.persistence.save_index(filename, self.index)

        records = [
            {
                "text": text,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
            }
            for text, metadata in zip(
                self.chunks,
                self.chunk_metadata,
            )
        ]

        self.persistence.save_chunks(filename, records)

        self.persistence.save_metadata(
            filename=filename,
            chunk_count=len(self.chunks),
            embedding_model=embedding_model,
            original_filename=original_filename,
            stored_filename=stored_filename,
            storage_id=filename,
        )

    def load(self, filename: str) -> bool:
        """Load a saved index and support both legacy and new chunk formats."""

        if not self.persistence.exists(filename):
            return False

        loaded_index = self.persistence.load_index(filename)
        loaded_chunks = self.persistence.load_chunks(filename)

        if loaded_index is None or loaded_chunks is None:
            return False

        if loaded_index.ntotal != len(loaded_chunks):
            raise ValueError(
                "Loaded index and chunk counts do not match."
            )

        texts = []
        metadata = []

        for chunk in loaded_chunks:
            if isinstance(chunk, str):
                # Legacy format: page metadata was not saved.
                texts.append(chunk)
                metadata.append({"source": None, "page": None})

            elif isinstance(chunk, dict):
                text = chunk.get("text")
                source = chunk.get("source")
                page = chunk.get("page")

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        "Persisted chunk records must contain non-empty text."
                    )

                if source is not None and not isinstance(source, str):
                    raise ValueError(
                        "Persisted chunk source must be a string or None."
                    )

                if page is not None and (
                    not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                ):
                    raise ValueError(
                        "Persisted chunk page must be a positive integer or None."
                    )

                texts.append(text)
                metadata.append({"source": source, "page": page})

            else:
                raise ValueError("Unsupported persisted chunk format.")

        self.index = loaded_index
        self.chunks = texts
        self.chunk_metadata = metadata

        return True

    def delete(self, filename: str) -> None:
        """Delete persisted data and clear the in-memory index."""

        self.persistence.delete(filename)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_loaded(self) -> bool:
        return self.index is not None
