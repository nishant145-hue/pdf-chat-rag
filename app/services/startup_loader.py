from pathlib import Path
import json

from faiss import logger

from app.services.rag_pipeline import RAGPipeline


class StartupLoader:
    """
    Automatically restores previously processed
    PDF documents when the FastAPI server starts.
    """

    def __init__(self, manager):

        self.manager = manager

        self.metadata_dir = Path(
            "vector_store/metadata"
        )

    # ---------------------------------------------------------
    # Restore All Documents
    # ---------------------------------------------------------

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

                filename = metadata["filename"]

                pdf_path = Path("uploads") / filename

                if not pdf_path.exists():

                    logger.info(
                        f"Skipping {filename} "
                        "(PDF missing)"
                    )

                    continue

                pipeline = RAGPipeline(
                    str(pdf_path)
                )

                document_id = self.manager.add_restored_document(
                    pipeline=pipeline,
                    filename=filename,
                    pdf_path=str(pdf_path),
                )

                restored += 1

                logger.info(
                    f"✓ Restored: {filename}"
                )

            except Exception as error:

                logger.info(
                    f"Failed to restore "
                    f"{metadata_file.name}"
                )

                logger.info(error)

        logger.info(
            f"Restored {restored} document(s)."
        )

        logger.info("=" * 60)