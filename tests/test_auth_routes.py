from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import auth_routes
from app.services.auth_service import AuthService


def test_auth_routes(tmp_path, monkeypatch):
    """Test registration and login using an isolated SQLite database."""

    service = AuthService(tmp_path / "test_auth.db")
    monkeypatch.setattr(auth_routes, "auth_service", service)

    app = FastAPI()
    app.include_router(auth_routes.router)
    client = TestClient(app)

    # Register a new account.
    registration = client.post(
        "/auth/register",
        json={
            "email": "student@example.com",
            "password": "A-Strong-Password-123!",
        },
    )

    assert registration.status_code == 201
    assert registration.json()["email"] == "student@example.com"
    assert "password_hash" not in registration.json()

    # Reject duplicate registration.
    duplicate = client.post(
        "/auth/register",
        json={
            "email": "STUDENT@example.com",
            "password": "Another-Strong-Password-456!",
        },
    )

    assert duplicate.status_code == 409

    # Log in with correct credentials.
    login = client.post(
        "/auth/login",
        json={
            "email": "student@example.com",
            "password": "A-Strong-Password-123!",
        },
    )

    assert login.status_code == 200

    token_response = login.json()

    assert token_response["token_type"] == "bearer"
    assert token_response["expires_in"] == 1800
    assert service.decode_access_token(token_response["access_token"]) == 1

    # Reject an incorrect password.
    invalid_login = client.post(
        "/auth/login",
        json={
            "email": "student@example.com",
            "password": "Wrong-Password-123!",
        },
    )

    assert invalid_login.status_code == 401
    assert invalid_login.json()["detail"] == "Invalid email or password."


def test_registration_rejects_invalid_email():
    app = FastAPI()
    app.include_router(auth_routes.router)
    client = TestClient(app)

    response = client.post(
        "/auth/register",
        json={
            "email": "not-an-email",
            "password": "A-Strong-Password-123!",
        },
    )

    assert response.status_code == 422