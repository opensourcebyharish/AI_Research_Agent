from fastapi.testclient import TestClient

from app.main import app


client = TestClient(
    app,
    raise_server_exceptions=False,
)


def test_unhandled_exception_returns_safe_error():
    @app.get("/test-internal-error")
    def test_internal_error():
        raise RuntimeError("secret internal failure")

    response = client.get(
        "/test-internal-error"
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal server error."
    }


def test_unhandled_exception_does_not_expose_internal_error():
    @app.get("/test-secret-error")
    def test_secret_error():
        raise RuntimeError(
            "database password should not be exposed"
        )

    response = client.get(
        "/test-secret-error"
    )

    assert response.status_code == 500

    assert (
        "database password"
        not in response.text
    )