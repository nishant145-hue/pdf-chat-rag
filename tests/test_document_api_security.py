from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import jwt
import pytest
from app.api.auth_routes import auth_service

from app.api.routes import manager
from app.config import settings

PROTECTED_ENDPOINTS = [
    ("get", "/documents"),
    ("get", "/status"),
    ("post", "/ask"),
    ("delete", "/documents/private-document"),
]


@pytest.mark.parametrize(
    ("method", "endpoint"),
    PROTECTED_ENDPOINTS,
)
def test_protected_endpoints_reject_missing_token(
    authenticated_client,
    method,
    endpoint,
):
    client = authenticated_client["client"]

    response = getattr(client, method)(endpoint)

    assert response.status_code == 401


def test_users_cannot_list_another_users_documents(
    authenticated_client,
    monkeypatch,
):
    client = authenticated_client["client"]
    current_user = authenticated_client["user"]

    # Create a second user who owns the document.
    other_user = auth_service.register_user(
        email="other-owner@example.com",
        password="secure-other-password-123",
    )

    auth_service.assign_document_owner(
        "private-document",
        other_user["id"],
    )

    # Simulate the document being loaded into the document manager.
    monkeypatch.setattr(
        manager,
        "list_documents",
        lambda: [
            {
                "document_id": "private-document",
                "filename": "private.pdf",
                "chunks": 1,
            }
        ],
    )

    # Confirm the requesting user is not the document owner.
    assert current_user["id"] != other_user["id"]

    response = client.get(
        "/documents",
        headers=authenticated_client["headers"],
    )

    assert response.status_code == 200
    assert response.json()["documents"] == []

def test_users_cannot_ask_questions_about_another_users_document(
    authenticated_client,
    monkeypatch,
):
    client = authenticated_client["client"]
    current_user = authenticated_client["user"]

    # Create another user who owns the document.
    other_user = auth_service.register_user(
        email="other-question-owner@example.com",
        password="secure-other-password-123",
    )

    assert current_user["id"] != other_user["id"]

    # Assign the document to the other user.
    auth_service.assign_document_owner(
        "private-document",
        other_user["id"],
    )

    # Mock the document pipeline to avoid running the actual RAG pipeline.
    pipeline = MagicMock()

    monkeypatch.setattr(
        manager,
        "get_document",
        lambda document_id: {
            "document_id": document_id,
            "pipeline": pipeline,
        },
    )

    # The current user attempts to query another user's document.
    response = client.post(
        "/ask",
        headers=authenticated_client["headers"],
        json={
            "document_id": "private-document",
            "question": "What does this document say?",
        },
    )

    # Access must be denied, and the pipeline must never execute.
    assert response.status_code == 404
    pipeline.ask_with_citations.assert_not_called()

def test_users_cannot_delete_another_users_document(
    authenticated_client,
    monkeypatch,
):
    client = authenticated_client["client"]
    current_user = authenticated_client["user"]

    # Create a second user who owns the document.
    other_user = auth_service.register_user(
        email="other-delete-owner@example.com",
        password="secure-delete-password-123",
    )

    assert current_user["id"] != other_user["id"]

    # Assign the document to the second user.
    auth_service.assign_document_owner(
        "private-document",
        other_user["id"],
    )

    # Simulate the document being loaded.
    monkeypatch.setattr(
        manager,
        "get_document",
        lambda document_id: {
            "document_id": document_id,
            "pipeline": MagicMock(),
        },
    )

    delete_calls = []

    def unexpected_delete(document_id):
        delete_calls.append(document_id)
        return True

    monkeypatch.setattr(
        manager,
        "delete_document",
        unexpected_delete,
    )

    # User A attempts to delete User B's document.
    response = client.delete(
        "/documents/private-document",
        headers=authenticated_client["headers"],
    )

    assert response.status_code == 404
    assert delete_calls == []

    # Confirm that User B's ownership record remains intact.
    assert auth_service.user_owns_document(
        "private-document",
        other_user["id"],
    )

    def unexpected_delete(document_id):
        delete_calls.append(document_id)
        return True

    monkeypatch.setattr(
        manager,
        "delete_document",
        unexpected_delete,
    )

    response = client.delete(
        "/documents/unassigned-document",
        headers=authenticated_client["headers"],
    )

    assert response.status_code == 404
    assert delete_calls == []

def test_successful_deletion_removes_ownership_record(
    authenticated_client,
    monkeypatch,
):
    client = authenticated_client["client"]
    user = authenticated_client["user"]

    document_id = "document-to-delete"

    auth_service.assign_document_owner(
        document_id,
        user["id"],
    )

    monkeypatch.setattr(
        manager,
        "get_document",
        lambda requested_id: {
            "document_id": requested_id,
            "pipeline": MagicMock(),
        },
    )

    monkeypatch.setattr(
        manager,
        "delete_document",
        lambda requested_id: True,
    )

    response = client.delete(
        f"/documents/{document_id}",
        headers=authenticated_client["headers"],
    )

    assert response.status_code == 200
    assert auth_service.get_document_owner(document_id) is None
    assert not auth_service.user_owns_document(
        document_id,
        user["id"],
    )

def test_status_counts_only_current_users_documents(
    authenticated_client,
    monkeypatch,
):
    client = authenticated_client["client"]
    current_user = authenticated_client["user"]

    # Register a second user.
    other_user = auth_service.register_user(
        email="other-status-owner@example.com",
        password="secure-status-password-123",
    )

    # Assign one document to each user.
    auth_service.assign_document_owner(
        "current-user-document",
        current_user["id"],
    )

    auth_service.assign_document_owner(
        "other-user-document",
        other_user["id"],
    )

    # Simulate both documents being loaded.
    monkeypatch.setattr(
        manager,
        "list_documents",
        lambda: [
            {
                "document_id": "current-user-document",
                "filename": "mine.pdf",
                "chunks": 1,
            },
            {
                "document_id": "other-user-document",
                "filename": "other.pdf",
                "chunks": 1,
            },
        ],
    )

    response = client.get(
        "/status",
        headers=authenticated_client["headers"],
    )

    assert response.status_code == 200
    assert response.json()["documents_loaded"] == 1

def test_protected_endpoint_rejects_invalid_signature(
    authenticated_client,
):
    client = authenticated_client["client"]
    user = authenticated_client["user"]

    invalid_token = jwt.encode(
        {
            "sub": str(user["id"]),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "incorrect-signing-secret-that-is-not-the-real-key",
        algorithm=settings.JWT_ALGORITHM,
    )

    response = client.get(
        "/documents",
        headers={
            "Authorization": f"Bearer {invalid_token}",
        },
    )

    assert response.status_code == 401


def test_protected_endpoint_rejects_expired_token(
    authenticated_client,
):
    client = authenticated_client["client"]
    user = authenticated_client["user"]

    expired_time = datetime.now(timezone.utc) - timedelta(minutes=10)

    expired_token = jwt.encode(
        {
            "sub": str(user["id"]),
            "iat": expired_time - timedelta(minutes=5),
            "exp": expired_time,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = client.get(
        "/documents",
        headers={
            "Authorization": f"Bearer {expired_token}",
        },
    )

    assert response.status_code == 401