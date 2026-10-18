
import pytest
from app.api.auth_routes import auth_service
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def authenticated_client(tmp_path, monkeypatch):
    """Provide a test client with a real JWT and isolated auth database."""

    database_path = (tmp_path / "test_auth.db").resolve()
    monkeypatch.setattr(auth_service, "database_path", database_path)

    auth_service.initialize_database()

    user = auth_service.register_user(
        email="api-test@example.com",
        password="secure-test-password-123",
    )

    token = auth_service.create_access_token(user["id"])

    with TestClient(app) as client:
        yield {
            "client": client,
            "user": user,
            "headers": {
                "Authorization": f"Bearer {token}",
            },
        }
