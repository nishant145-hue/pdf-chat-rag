import json
from pathlib import Path

from app.api.auth_routes import auth_service
from app.logger import logger
from app.services.rag_pipeline import RAGPipeline


class StartupLoader:
    """Restore previously processed PDFs when the application starts."""

    def __init__(self, manager):
        self.manager = manager
        self.metadata_dir = Path("vector_store/metadata")

    def load_documents(self):
        if not self.metadata_dir.exists():
            logger.info("No metadata directory found.")
            return

        metadata_files = list(
            self.metadata_dir.glob("*_metadata.json")
        )

        if not metadata_files:
            logger.info("No cached documents found.")
            return

        logger.info("=" * 60)
        logger.info("Loading cached documents...")

        restored = 0

        for metadata_file in metadata_files:
            try:
                with open(
                    metadata_file,
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)

                filename = metadata.get("filename")

                if not filename:
                    logger.warning(
                        "Skipping metadata without a filename: %s",
                        metadata_file.name,
                    )
                    continue

                # New uploads record the actual disk filename.
                # Older metadata falls back to the original filename.
                stored_filename = metadata.get(
                    "stored_filename",
                    filename,
                )

                pdf_path = Path("uploads") / stored_filename

                if not pdf_path.is_file():
                    logger.info(
                        "Skipping %s: PDF file is missing.",
                        filename,
                    )
                    continue


                # Legacy metadata was keyed by the original filename.
                storage_id = metadata.get("storage_id") or filename

                # Preserve the persisted storage ID as the document ID.
                document_id = storage_id

                # Legacy documents remain unassigned unless an ownership
                # record already exists in the authentication database.
                owner_user_id = auth_service.get_document_owner(
                    document_id
                )



                pipeline = RAGPipeline(
                    pdf_path=str(pdf_path),
                    storage_id=storage_id,
                    original_filename=filename,
                )

                self.manager.add_restored_document(
                    pipeline=pipeline,
                    filename=filename,
                    pdf_path=str(pdf_path),
                    document_id=document_id,
                    owner_user_id=owner_user_id,
                )

                restored += 1

                logger.info("Restored: %s", filename)

            except Exception:
                logger.exception(
                    "Failed to restore %s",
                    metadata_file.name,
                )

        logger.info("Restored %d document(s).", restored)
        logger.info("=" * 60)