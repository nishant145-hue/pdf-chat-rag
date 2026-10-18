import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.auth_service import AuthService

from app.services.document_manager import DocumentManager
from app.services.startup_loader import StartupLoader


def write_metadata(metadata_dir: Path, name: str, data: dict):
    metadata_path = metadata_dir / f"{name}_metadata.json"

    metadata_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    return metadata_path


def test_restores_document_with_new_metadata_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    metadata_dir = tmp_path / "vector_store" / "metadata"
    uploads_dir = tmp_path / "uploads"

    metadata_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    stored_filename = "unique-id_report.pdf"
    pdf_path = uploads_dir / stored_filename
    pdf_path.write_bytes(b"test PDF")

    write_metadata(
        metadata_dir,
        "storage-123",
        {
            "filename": "report.pdf",
            "storage_id": "storage-123",
            "stored_filename": stored_filename,
            "chunk_count": 2,
            "embedding_model": "all-MiniLM-L6-v2",
        },
    )

    manager = DocumentManager()

    pipeline = MagicMock()
    pipeline.total_chunks.return_value = 2

    with patch(
        "app.services.startup_loader.RAGPipeline",
        return_value=pipeline,
    ) as mock_pipeline:
        loader = StartupLoader(manager)
        loader.load_documents()

    mock_pipeline.assert_called_once_with(
        pdf_path=str(Path("uploads") / stored_filename),
        storage_id="storage-123",
        original_filename="report.pdf",
    )

    assert manager.total_documents() == 1

    document = manager.list_documents()[0]

    assert document["filename"] == "report.pdf"
    assert document["chunks"] == 2


def test_restores_document_with_legacy_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    metadata_dir = tmp_path / "vector_store" / "metadata"
    uploads_dir = tmp_path / "uploads"

    metadata_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    pdf_path = uploads_dir / "legacy.pdf"
    pdf_path.write_bytes(b"legacy PDF")

    write_metadata(
        metadata_dir,
        "legacy",
        {
            "filename": "legacy.pdf",
            "chunk_count": 1,
            "embedding_model": "all-MiniLM-L6-v2",
        },
    )

    manager = DocumentManager()

    pipeline = MagicMock()
    pipeline.total_chunks.return_value = 1

    with patch(
        "app.services.startup_loader.RAGPipeline",
        return_value=pipeline,
    ) as mock_pipeline:
        StartupLoader(manager).load_documents()

    mock_pipeline.assert_called_once_with(
        pdf_path=str(Path("uploads") / "legacy.pdf"),
        storage_id="legacy.pdf",
        original_filename="legacy.pdf",
    )

    assert manager.total_documents() == 1
    assert manager.list_documents()[0]["filename"] == "legacy.pdf"


def test_skips_document_when_pdf_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    metadata_dir = tmp_path / "vector_store" / "metadata"
    metadata_dir.mkdir(parents=True)

    write_metadata(
        metadata_dir,
        "missing",
        {
            "filename": "missing.pdf",
            "storage_id": "missing-storage",
            "stored_filename": "missing-storage_missing.pdf",
        },
    )

    manager = DocumentManager()

    with patch(
        "app.services.startup_loader.RAGPipeline"
    ) as mock_pipeline:
        StartupLoader(manager).load_documents()

    mock_pipeline.assert_not_called()
    assert manager.total_documents() == 0

def test_restored_document_preserves_ownership_after_restart(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    # Create isolated storage directories.
    metadata_dir = tmp_path / "vector_store" / "metadata"
    uploads_dir = tmp_path / "uploads"

    metadata_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    # Create a persisted PDF and its metadata.
    stored_filename = "private-report.pdf"
    pdf_path = uploads_dir / stored_filename
    pdf_path.write_bytes(b"test PDF")

    write_metadata(
        metadata_dir,
        "persistent-document",
        {
            "filename": "private-report.pdf",
            "storage_id": "persistent-document",
            "stored_filename": stored_filename,
            "chunk_count": 1,
            "embedding_model": "all-MiniLM-L6-v2",
        },
    )

    # Create an isolated authentication database.
    auth_database = tmp_path / "auth.db"
    service = AuthService(auth_database)

    owner = service.register_user(
        "persistent-owner@example.com",
        "secure-owner-password-123",
    )

    other_user = service.register_user(
        "other-persistent-user@example.com",
        "secure-other-password-123",
    )

    service.assign_document_owner(
        "persistent-document",
        owner["id"],
    )

    # Simulate an application restart by creating a new manager
    # and restoring the document from persisted metadata.
    manager = DocumentManager()

    pipeline = MagicMock()
    pipeline.total_chunks.return_value = 1

    with (
        patch(
            "app.services.startup_loader.auth_service",
            service,
        ),
        patch(
            "app.services.startup_loader.RAGPipeline",
            return_value=pipeline,
        ),
    ):
        StartupLoader(manager).load_documents()

    # Confirm that the document was restored.
    assert manager.total_documents() == 1

    document = manager.get_document("persistent-document")

    assert document is not None
    assert document["filename"] == "private-report.pdf"

    # Ownership must remain associated with the original owner.
    assert service.user_owns_document(
        "persistent-document",
        owner["id"],
    )

    assert not service.user_owns_document(
        "persistent-document",
        other_user["id"],
    )

    # The restored document should also retain its owner ID in memory.
    assert document["owner_user_id"] == owner["id"]

def test_restored_legacy_document_has_no_owner(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    metadata_dir = tmp_path / "vector_store" / "metadata"
    uploads_dir = tmp_path / "uploads"

    metadata_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    stored_filename = "legacy-report.pdf"

    (uploads_dir / stored_filename).write_bytes(b"legacy PDF")

    write_metadata(
        metadata_dir,
        "legacy-report",
        {
            "filename": "legacy-report.pdf",
            "storage_id": "legacy-report",
            "stored_filename": stored_filename,
            "chunk_count": 1,
        },
    )

    auth_database = tmp_path / "legacy_auth.db"
    service = AuthService(auth_database)

    user = service.register_user(
        "legacy-test-user@example.com",
        "secure-legacy-password-123",
    )

    manager = DocumentManager()

    pipeline = MagicMock()
    pipeline.total_chunks.return_value = 1

    with (
        patch(
            "app.services.startup_loader.auth_service",
            service,
        ),
        patch(
            "app.services.startup_loader.RAGPipeline",
            return_value=pipeline,
        ),
    ):
        StartupLoader(manager).load_documents()

    document = manager.get_document("legacy-report")

    assert document is not None
    assert document["owner_user_id"] is None

    assert service.get_document_owner("legacy-report") is None

    assert not service.user_owns_document(
        "legacy-report",
        user["id"],
    )