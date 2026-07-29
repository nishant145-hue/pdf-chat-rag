"""
Custom exceptions for the PDF Chat RAG application.
"""


class RAGError(Exception):
    """
    Base exception for the entire application.
    """

    default_message = "An unknown RAG error occurred."

    def __init__(self, message: str | None = None):

        self.message = message or self.default_message

        super().__init__(self.message)


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------


class PDFProcessingError(RAGError):

    default_message = "Unable to process the PDF."


class EmptyPDFError(PDFProcessingError):

    default_message = (
        "The PDF contains no extractable text."
    )


class InvalidPDFError(PDFProcessingError):

    default_message = (
        "The uploaded file is not a valid PDF."
    )


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------


class ChunkingError(RAGError):

    default_message = (
        "Text chunking failed."
    )


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------


class EmbeddingError(RAGError):

    default_message = (
        "Embedding generation failed."
    )


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------


class VectorStoreError(RAGError):

    default_message = (
        "Vector store operation failed."
    )


# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------


class PersistenceError(RAGError):

    default_message = (
        "Persistence operation failed."
    )


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------


class LLMError(RAGError):

    default_message = (
        "Language model request failed."
    )


# ---------------------------------------------------------
# Documents
# ---------------------------------------------------------


class DocumentNotFoundError(RAGError):

    default_message = (
        "Document not found."
    )


class DuplicateDocumentError(RAGError):

    default_message = (
        "Document already exists."
    )


# ---------------------------------------------------------
# Cache
# ---------------------------------------------------------


class CacheError(RAGError):

    default_message = (
        "Cache operation failed."
    )