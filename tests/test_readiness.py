from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_readiness_check_returns_ready():
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["service"] == "AI Research Agent"
    assert "documents" in data
    assert "chunks" in data


def test_readiness_check_reports_store_counts():
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["documents"], int)
    assert isinstance(data["chunks"], int)
    assert data["documents"] >= 0
    assert data["chunks"] >= 0