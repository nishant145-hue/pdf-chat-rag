from fastapi.testclient import TestClient

from app.main import app

def test_cors_allows_localhost_frontend():
    with TestClient(app) as client:
        response = client.options(
        "/",
        headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        },
        )

    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )


def test_cors_allows_loopback_frontend():
    with TestClient(app) as client:
        response = client.options(
        "/",
        headers={
        "Origin": "http://127.0.0.1:5173",
        "Access-Control-Request-Method": "GET",
        },
        )

    assert response.headers.get("access-control-allow-origin") == (
        "http://127.0.0.1:5173"
    )


def test_cors_rejects_unapproved_origin():
    with TestClient(app) as client:
        response = client.options(
        "/",
        headers={
        "Origin": "https://unapproved.example",
        "Access-Control-Request-Method": "GET",
        },
        )


    assert "access-control-allow-origin" not in response.headers
