from app.config import settings
from app.logger import logger

from app.services.embeddings import create_embedding
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore


class Retriever:
    """Retrieve candidate chunks and optionally rerank them."""

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[str]:
        """Retrieve relevant chunks for a question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self.vector_store.is_loaded():
            raise RuntimeError("Vector store is not initialized.")

        if top_k is None:
            top_k = settings.FINAL_TOP_K

        if candidate_k is None:
            candidate_k = settings.FAISS_TOP_K

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

        if (
            not isinstance(candidate_k, int)
            or isinstance(candidate_k, bool)
            or candidate_k <= 0
        ):
            raise ValueError("candidate_k must be a positive integer.")

        query_embedding = create_embedding(question)

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            k=candidate_k,
        )

        if not candidates:
            return []

        if not settings.ENABLE_RERANKING:
            return candidates[:top_k]

        try:
            if self.reranker is None:
                self.reranker = Reranker()

            return self.reranker.rerank(
                question=question,
                chunks=candidates,
                top_k=top_k,
            )

        except Exception as exc:
            logger.warning(
                "Reranking failed; using FAISS results instead. Error: %s",
                exc,
            )

            return candidates[:top_k]
