from sentence_transformers import CrossEncoder

from app.logger import logger


class Reranker:
    """
    Cross-encoder based reranker.

    Reorders retrieved chunks according
    to their relevance to the query.
    """

    def __init__(
        self,
        model_name: str = (
            "CrossEncoder(settings.RERANKER_MODEL)"
        ),
    ):

        logger.info(
            "Loading reranker model: %s",
            model_name,
        )

        self.model = CrossEncoder(model_name)

        logger.info("Reranker loaded successfully.")

    # ---------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------

    def rerank(
        self,
        question: str,
        chunks: list[str],
        top_k: int = 5,
    ) -> list[str]:

        if not chunks:
            return []

        pairs = [
            (question, chunk)
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        logger.info(
            "Reranked %d chunk(s).",
            len(chunks),
        )

        return [
            chunk
            for chunk, score in ranked[:top_k]
        ]