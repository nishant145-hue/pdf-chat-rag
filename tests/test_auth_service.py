import sqlite3
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.config import settings
from app.services.auth_service import AuthService

@pytest.fixture
def auth_service(tmp_path):
    """Create an isolated authentication database for each test."""
    database_path = tmp_path / "test_auth.db"
    return AuthService(database_path=database_path)

def test_register_user(auth_service):
    user = auth_service.register_user(
    "student@example.com",
    "A-Strong-Password-123!",
    )


    assert user["id"] > 0
    assert user["email"] == "student@example.com"
    assert "password_hash" not in user


def test_email_is_normalized(auth_service):
    user = auth_service.register_user(
    "  student@example.com ",
    "A-Strong-Password-123!",
    )

    assert user["email"] == "student@example.com"


def test_duplicate_email_is_rejected(auth_service):
    auth_service.register_user(
    "student@example.com",
    "A-Strong-Password-123!",
    )


    with pytest.raises(ValueError, match="already exists"):
        auth_service.register_user(
            "STUDENT@example.com",
            "Another-Strong-Password-456!",
        )


def test_short_password_is_rejected(auth_service):
    with pytest.raises(ValueError, match="at least 12 characters"):
        auth_service.register_user(
        "student@example.com",
        "short",
        )

def test_authenticate_user_with_correct_password(auth_service):
    auth_service.register_user(
    "student@example.com",
    "A-Strong-Password-123!",
    )

    user = auth_service.authenticate_user(
        "STUDENT@example.com",
        "A-Strong-Password-123!",
    )

    assert user is not None
    assert user["email"] == "student@example.com"
    assert "password_hash" not in user


def test_wrong_password_is_rejected(auth_service):
    auth_service.register_user(
    "student@example.com",
    "A-Strong-Password-123!",
    )


    user = auth_service.authenticate_user(
        "student@example.com",
        "Wrong-Password-123!",
    )

    assert user is None


def test_unknown_email_is_rejected(auth_service):
    user = auth_service.authenticate_user(
    "[unknown@example.com](mailto:unknown@example.com)",
    "A-Strong-Password-123!",
    )


    assert user is None


def test_user_persists_across_service_instances(tmp_path):
    database_path = tmp_path / "persistent_auth.db"

    first_service = AuthService(database_path)
    first_service.register_user(
        "student@example.com",
        "A-Strong-Password-123!",
    )

    second_service = AuthService(database_path)

    user = second_service.authenticate_user(
        "student@example.com",
        "A-Strong-Password-123!",
    )

    assert user is not None
    assert user["email"] == "student@example.com"


def test_access_token_can_be_decoded(auth_service):
    user = auth_service.register_user(
    "student@example.com",
    "A-Strong-Password-123!",
    )


    token = auth_service.create_access_token(user["id"])

    decoded_user_id = auth_service.decode_access_token(token)

    assert decoded_user_id == user["id"]


def test_invalid_access_token_is_rejected(auth_service):
    assert auth_service.decode_access_token("not-a-valid-token") is None

def test_expired_access_token_is_rejected(auth_service):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)


    token = jwt.encode(
        {
            "sub": "1",
            "iat": expired_time - timedelta(minutes=5),
            "exp": expired_time,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    assert auth_service.decode_access_token(token) is None


def test_missing_user_returns_none(auth_service):
    assert auth_service.get_user_by_id(99999) is None

def test_access_token_with_invalid_signature_is_rejected(auth_service):
    user = auth_service.register_user(
        "signature-test@example.com",
        "A-Strong-Password-123!",
    )

    token = jwt.encode(
        {
            "sub": str(user["id"]),
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "incorrect-signing-secret-that-is-not-the-real-key",
        algorithm=settings.JWT_ALGORITHM,
    )

    assert auth_service.decode_access_token(token) is None


def test_access_token_missing_required_claim_is_rejected(auth_service):
    token = jwt.encode(
        {
            "sub": "1",
            "iat": datetime.now(timezone.utc),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    assert auth_service.decode_access_token(token) is None


def test_access_token_issued_in_future_is_rejected(auth_service):
    future_time = datetime.now(timezone.utc) + timedelta(minutes=10)

    token = jwt.encode(
        {
            "sub": "1",
            "iat": future_time,
            "exp": future_time + timedelta(minutes=30),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    assert auth_service.decode_access_token(token) is None