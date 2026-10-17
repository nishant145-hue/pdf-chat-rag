from sentence_transformers import CrossEncoder

from app.config import settings
from app.logger import logger


class Reranker:
    """Reorder retrieved chunks by their relevance to a question."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL

        logger.info(
            "Loading reranker model: %s",
            self.model_name,
        )

        self.model = CrossEncoder(self.model_name)

        logger.info("Reranker loaded successfully.")

    def rerank(
        self,
        question: str,
        chunks: list[str],
        top_k: int | None = None,
    ) -> list[str]:
        """Return the highest-scoring chunks for the question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not chunks:
            return []

        if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
            raise ValueError("Chunks must contain non-empty strings.")

        if top_k is None:
            top_k = settings.FINAL_TOP_K

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        pairs = [(question, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)

        if len(scores) != len(chunks):
            raise ValueError("The reranker returned an unexpected number of scores.")

        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        logger.info("Reranked %d chunk(s).", len(chunks))

        return [
            chunk
            for chunk, _ in ranked[:top_k]
        ]
