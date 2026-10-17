from pathlib import Path

from app.config import settings
from app.logger import logger
from app.services.embeddings import create_embedding, create_embeddings
from app.services.llm import generate_answer
from app.services.metadata_chunker import chunk_pages
from app.services.pdf_reader import extract_pages_from_pdf
from app.services.retriever import Retriever
from app.services.vector_store import VectorStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for a single PDF."""

    def __init__(
        self,
        pdf_path: str,
        storage_id: str | None = None,
        original_filename: str | None = None,
    ):
        self.pdf_path = pdf_path

        # Preserve the user-facing filename for citations and display.
        self.filename = original_filename or Path(pdf_path).name

        # Identify persisted index and chunk files independently.
        self.storage_id = storage_id or self.filename

        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)

        self.process_document()

    def process_document(self) -> None:
        """Load a compatible cached index or process the PDF from scratch."""

        logger.info("=" * 60)
        logger.info("Processing: %s", self.filename)

        if self.vector_store.load(self.storage_id):
            metadata = self.vector_store.persistence.load_metadata(
                    self.storage_id
                ) or {}

            # Legacy caches lack page-level source metadata.
            # Reprocess them to create accurate citations.
            has_page_metadata = (
                self.vector_store.total_chunks() > 0
                and all(
                    item.get("source") is not None
                    and item.get("page") is not None
                    for item in self.vector_store.chunk_metadata
                )
            )

            cached_model = metadata.get("embedding_model")
            compatible_model = cached_model == settings.EMBEDDING_MODEL

            if has_page_metadata and compatible_model:
                logger.info("Compatible cached index found.")
                logger.info("Pipeline ready.")
                logger.info("=" * 60)
                return

            logger.info(
                "Cached index is legacy or uses a different embedding model; "
                "rebuilding it."
            )

        logger.info("Extracting PDF pages...")

        pages = extract_pages_from_pdf(self.pdf_path)

        logger.info("Creating metadata-aware chunks...")

        chunks = chunk_pages(
            pages=pages,
            source=self.filename,
        )

        if not chunks:
            raise ValueError("No usable text chunks were created from the PDF.")

        logger.info("Generating embeddings for %d chunks...", len(chunks))

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = create_embeddings(chunk_texts)

        logger.info("Building FAISS index...")

        self.vector_store.build_index(
            embeddings=embeddings,
            chunks=chunks,
        )

        logger.info("Saving index and metadata...")

        self.vector_store.save(
            filename=self.storage_id,
            embedding_model=settings.EMBEDDING_MODEL,
            original_filename=self.filename,
            stored_filename=Path(self.pdf_path).name,
        )

        logger.info("Pipeline ready.")
        logger.info("=" * 60)

    def ask(self, question: str) -> str:
        """Answer a question using the retrieved document context."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self.vector_store.is_loaded():
            raise RuntimeError("Vector store is not initialized.")

        retrieved_chunks = self.retriever.retrieve(question)

        if not retrieved_chunks:
            return "I couldn't find that in the uploaded document."

        context = "\n\n".join(retrieved_chunks)

        return generate_answer(
            question=question,
            context=context,
        )

    def ask_with_citations(self, question: str) -> dict:
        """Return an answer together with its retrieved source citations."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self.vector_store.is_loaded():
            raise RuntimeError("Vector store is not initialized.")

        query_embedding = create_embedding(question)

        candidate_k = settings.FAISS_TOP_K
        top_k = settings.FINAL_TOP_K

        candidates = self.vector_store.search_with_metadata(
            query_embedding=query_embedding,
            k=candidate_k,
        )

        if not candidates:
            return {
                "answer": "I couldn't find that in the uploaded document.",
                "citations": [],
            }

        candidate_texts = [item["text"] for item in candidates]

        if settings.ENABLE_RERANKING:
            try:
                if self.retriever.reranker is None:
                    from app.services.reranker import Reranker
                    self.retriever.reranker = Reranker()

                candidate_texts = self.retriever.reranker.rerank(
                    question=question,
                    chunks=candidate_texts,
                    top_k=top_k,
                )
            except Exception as exc:
                logger.warning(
                    "Reranking failed; using FAISS results instead. Error: %s",
                    exc,
                )
                candidate_texts = candidate_texts[:top_k]
        else:
            candidate_texts = candidate_texts[:top_k]

        # Match reranked text back to its metadata.
        # Track consumed entries so duplicate chunk text does not always
        # resolve to the same citation.
        remaining = list(candidates)
        selected = []

        for text in candidate_texts:
            for index, item in enumerate(remaining):
                if item["text"] == text:
                    selected.append(item)
                    remaining.pop(index)
                    break

        context = "\n\n".join(item["text"] for item in selected)

        answer = generate_answer(
            question=question,
            context=context,
        )

        citations = [
            {
                "source": item["source"],
                "page": item["page"],
                "text": item["text"],
            }
            for item in selected
        ]

        return {
            "answer": answer,
            "citations": citations,
        }

    def total_chunks(self) -> int:
        return self.vector_store.total_chunks()

    def filename_only(self) -> str:
        return self.filename
