from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.document_manager import DocumentManager


def test_deleting_document_does_not_delete_another_document(tmp_path):
    """Deleting one document must not affect another document."""

    first_pdf = tmp_path / "first_report.pdf"
    second_pdf = tmp_path / "second_report.pdf"

    first_pdf.write_bytes(b"first PDF")
    second_pdf.write_bytes(b"second PDF")

    manager = DocumentManager()

    with patch(
        "app.services.document_manager.RAGPipeline"
    ) as mock_pipeline_class:
        first_pipeline = MagicMock()
        first_pipeline.total_chunks.return_value = 2
        first_pipeline.storage_id = "storage-first"

        second_pipeline = MagicMock()
        second_pipeline.total_chunks.return_value = 3
        second_pipeline.storage_id = "storage-second"

        mock_pipeline_class.side_effect = [
            first_pipeline,
            second_pipeline,
        ]

        first_id = manager.add_document(
            pdf_path=str(first_pdf),
            filename="report.pdf",
        )

        second_id = manager.add_document(
            pdf_path=str(second_pdf),
            filename="report.pdf",
        )

    assert first_id != second_id
    assert manager.total_documents() == 2

    # Delete the first document.
    assert manager.delete_document(first_id) is True

    # Its PDF is removed.
    assert not first_pdf.exists()

    # The second document remains intact.
    assert second_pdf.exists()
    assert second_pdf.read_bytes() == b"second PDF"

    assert manager.get_document(first_id) is None
    assert manager.get_document(second_id) is not None
    assert manager.total_documents() == 1

    # Deletion must use the correct storage ID.
    first_pipeline.vector_store.delete.assert_called_once_with(
        "storage-first"
    )

    second_pipeline.vector_store.delete.assert_not_called()


def test_deleting_unknown_document_returns_false():
    manager = DocumentManager()

    assert manager.delete_document("missing-document") is False
    assert manager.total_documents() == 0