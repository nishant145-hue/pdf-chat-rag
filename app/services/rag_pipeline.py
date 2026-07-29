from pathlib import Path
from app.config import settings
from app.logger import logger

from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.chunker import split_text
from app.services.embeddings import (
    create_embedding,
    create_embeddings,
)
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Responsibilities:
    - Load cached FAISS index if available
    - Otherwise build a new index
    - Answer user questions
    """

    def __init__(self, pdf_path: str):

        self.pdf_path = pdf_path

        self.filename = Path(pdf_path).name

        self.vector_store = VectorStore()

        self.process_document()

    # ---------------------------------------------------------
    # Build / Load
    # ---------------------------------------------------------

    def process_document(self):

        logger.info("=" * 60)
        logger.info(f"Processing: {self.filename}")

        # -------------------------------------------------
        # Load Cached Index
        # -------------------------------------------------

        if self.vector_store.load(self.filename):

            logger.info("Cached FAISS index found.")
            logger.info("Loading index from disk...")
            logger.info("Loading chunks...")
            logger.info("Pipeline Ready!")
            logger.info("=" * 60)

            return

        # -------------------------------------------------
        # Build New Index
        # -------------------------------------------------

        logger.info("No cached index found.")
        logger.info("Reading PDF...")

        text = extract_text_from_pdf(
            self.pdf_path
        )

        logger.info("Cleaning text...")

        text = clean_text(text)

        logger.info("Chunking...")

        chunks = split_text(text)

        logger.info("Generating embeddings...")

        embeddings = create_embeddings(chunks)

        logger.info("Building FAISS index...")

        self.vector_store.build_index(
            embeddings,
            chunks,
        )

        logger.info("Saving FAISS index...")

        self.vector_store.save(
            filename=self.filename,
        )

        logger.info("Pipeline Ready!")

        logger.info("=" * 60)

    # ---------------------------------------------------------
    # Ask Question
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> str:

        if not self.vector_store.is_loaded():
            raise RuntimeError(
                "Vector store is not initialized."
            )

        query_embedding = create_embedding(
            question
        )

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            k=3,
        )

        context = "\n\n".join(
            retrieved_chunks
        )

        answer = generate_answer(
            question=question,
            context=context,
        )

        return answer

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def total_chunks(self):

        return self.vector_store.total_chunks()

    def filename_only(self):

        return self.filename