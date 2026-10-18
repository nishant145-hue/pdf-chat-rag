
from app.services.auth_service import AuthService


def test_document_owner_can_be_verified(tmp_path):
    service = AuthService(tmp_path / "ownership.db")

    user = service.register_user(
        "owner@example.com",
        "secure-password-123",
    )

    service.assign_document_owner(
        "document-123",
        user["id"],
    )

    assert service.user_owns_document(
        "document-123",
        user["id"],
    )


def test_another_user_does_not_own_document(tmp_path):
    service = AuthService(tmp_path / "ownership.db")

    owner = service.register_user(
        "owner@example.com",
        "secure-password-123",
    )

    another_user = service.register_user(
        "another@example.com",
        "secure-password-456",
    )

    service.assign_document_owner(
        "document-123",
        owner["id"],
    )

    assert not service.user_owns_document(
        "document-123",
        another_user["id"],
    )


def test_unassigned_document_has_no_owner(tmp_path):
    service = AuthService(tmp_path / "ownership.db")

    user = service.register_user(
        "user@example.com",
        "secure-password-123",
    )

    assert not service.user_owns_document(
        "legacy-document",
        user["id"],
    )


def test_delete_document_owner_removes_ownership_record(tmp_path):
    service = AuthService(tmp_path / "ownership_cleanup.db")

    user = service.register_user(
        "cleanup@example.com",
        "secure-password-123",
    )

    service.assign_document_owner(
        "document-to-delete",
        user["id"],
    )

    assert service.user_owns_document(
        "document-to-delete",
        user["id"],
    )

    deleted = service.delete_document_owner(
        "document-to-delete",
    )

    assert deleted is True

    assert service.get_document_owner(
        "document-to-delete",
    ) is None

    assert not service.user_owns_document(
        "document-to-delete",
        user["id"],
    )