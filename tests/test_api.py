from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.api import routes


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "AI Research Agent",
    }


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "question must not be empty"
    )


def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "top_k": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "top_k must be greater than 0"
    )


def test_ask_returns_rag_response(monkeypatch):
    mock_rag_service = MagicMock()

    mock_rag_service.answer.return_value = {
        "answer": "RAG combines retrieval and generation.",
        "results": [
            {
                "chunk": "RAG combines retrieval and generation.",
                "distance": 0.5,
                "metadata": {
                    "source": "research.pdf",
                    "page": 1,
                    "chunk_id": 0,
                },
            }
        ],
        "context": (
            "[Source: research.pdf | Page: 1 | Chunk: 0]\n"
            "RAG combines retrieval and generation."
        ),
        "prompt": "test prompt",
        "citations": [
            {
                "id": 1,
                "source": "research.pdf",
                "page": 1,
            }
        ],
    }

    monkeypatch.setattr(
        routes,
        "get_rag_service",
        lambda: mock_rag_service,
    )

    routes._rag_service = None

    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "RAG combines retrieval and generation."
    )

    assert data["citations"] == [
        {
            "id": 1,
            "source": "research.pdf",
            "page": 1,
        }
    ]

    mock_rag_service.answer.assert_called_once_with(
        question="What is RAG?",
        top_k=5,
    )


def test_ask_handles_rag_value_error(monkeypatch):
    mock_rag_service = MagicMock()

    mock_rag_service.answer.side_effect = ValueError(
        "invalid question"
    )

    monkeypatch.setattr(
        routes,
        "get_rag_service",
        lambda: mock_rag_service,
    )

    routes._rag_service = None

    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid question"


def test_upload_rejects_missing_file():
    response = client.post(
        "/documents/upload"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "A file is required."
    )


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.txt",
                b"test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only PDF files are supported."
    )


def test_upload_indexes_pdf(monkeypatch):
    mock_ingestion_service = MagicMock()

    mock_ingestion_service.ingest.return_value = 3

    mock_registry = MagicMock()

    mock_registry.exists.return_value = False

    mock_registry.add.return_value = {
        "source": "research.pdf",
        "pages": 2,
        "chunks": 3,
    }

    monkeypatch.setattr(
        routes,
        "get_ingestion_service",
        lambda: mock_ingestion_service,
    )

    monkeypatch.setattr(
        routes,
        "get_document_registry",
        lambda: mock_registry,
    )

    mock_ingestion_service.document_loader.load.return_value = [
        {
            "page_number": 1,
            "text": "Page one content.",
        },
        {
            "page_number": 2,
            "text": "Page two content.",
        },
    ]

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.pdf",
                b"%PDF-test-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "message": "Document indexed successfully.",
        "filename": "research.pdf",
        "pages": 2,
        "chunks_indexed": 3,
    }

    mock_ingestion_service.ingest.assert_called_once()

    mock_registry.add.assert_called_once_with(
        source="research.pdf",
        pages=2,
        chunks=3,
    )


def test_upload_handles_ingestion_value_error(monkeypatch):
    mock_ingestion_service = MagicMock()

    mock_ingestion_service.ingest.side_effect = ValueError(
        "invalid PDF"
    )

    mock_registry = MagicMock()

    mock_registry.exists.return_value = False

    monkeypatch.setattr(
        routes,
        "get_ingestion_service",
        lambda: mock_ingestion_service,
    )

    monkeypatch.setattr(
        routes,
        "get_document_registry",
        lambda: mock_registry,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.pdf",
                b"%PDF-test-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid PDF"

    mock_ingestion_service.ingest.assert_called_once()