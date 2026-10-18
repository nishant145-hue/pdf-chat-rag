from pathlib import Path
from uuid import uuid4

from app.services.rag_pipeline import RAGPipeline


class DocumentManager:
    """
    Manages all loaded documents.

    Responsibilities:
    -----------------
    - Register documents
    - Store pipelines
    - Return pipelines
    - Delete documents
    - List loaded documents
    """

    def __init__(self):

        self.documents = {}

    # ---------------------------------------------------------
    # Add Document
    # ---------------------------------------------------------


    def add_document(
        self,
        pdf_path: str,
        filename: str,
        owner_user_id: int | None = None,
    ) -> str:
        """Register a document and optionally associate its owner."""
        document_id = str(uuid4())

        pipeline = RAGPipeline(
            pdf_path=pdf_path,
            storage_id=document_id,
            original_filename=filename,
        )

        self.documents[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "path": pdf_path,
            "pipeline": pipeline,
            "chunks": pipeline.total_chunks(),
            "owner_user_id": owner_user_id,
        }

        return document_id

    def add_restored_document(
        self,
        pipeline,
        filename: str,
        pdf_path: str,
        owner_user_id: int | None = None,
        document_id: str | None = None,
    ) -> str:
        """Register a restored document with a stable ID when available."""
        restored_id = document_id or pipeline.storage_id

        self.documents[restored_id] = {
            "document_id": restored_id,
            "filename": filename,
            "path": pdf_path,
            "pipeline": pipeline,
            "chunks": pipeline.total_chunks(),
            "owner_user_id": owner_user_id,
        }

        return restored_id


    # ---------------------------------------------------------
    # Get Pipeline
    # ---------------------------------------------------------

    def get_pipeline(
        self,
        document_id: str,
    ):

        document = self.documents.get(document_id)

        if document is None:
            return None

        return document["pipeline"]

    # ---------------------------------------------------------
    # Document Information
    # ---------------------------------------------------------

    def get_document(
        self,
        document_id: str,
    ):

        return self.documents.get(document_id)

    # ---------------------------------------------------------
    # List Documents
    # ---------------------------------------------------------

    def list_documents(self):

        return [
            {
                "document_id": document["document_id"],
                "filename": document["filename"],
                "chunks": document["chunks"],
            }
            for document in self.documents.values()
        ]

    # ---------------------------------------------------------
    # Delete Document
    # ---------------------------------------------------------

    def delete_document(self, document_id: str) -> bool:
        document = self.documents.get(document_id)

        if document is None:
            return False

        pipeline = document["pipeline"]

        # Delete persisted data using the unique storage ID.
        pipeline.vector_store.delete(pipeline.storage_id)

        pdf_path = Path(document["path"])

        if pdf_path.exists():
            pdf_path.unlink()

        del self.documents[document_id]

        return True

    # ---------------------------------------------------------
    # Exists
    # ---------------------------------------------------------

    def exists(
        self,
        document_id: str,
    ) -> bool:

        return document_id in self.documents

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def total_documents(self) -> int:

        return len(self.documents)

    def total_chunks(self) -> int:

        return sum(
            document["chunks"]
            for document in self.documents.values()
        )

    # ---------------------------------------------------------
    # Clear Everything
    # ---------------------------------------------------------

    def clear(self):
        for document in list(self.documents.values()):
            pipeline = document["pipeline"]

            pipeline.vector_store.delete(pipeline.storage_id)

            pdf_path = Path(document["path"])

            if pdf_path.exists():
                pdf_path.unlink()

        self.documents.clear()