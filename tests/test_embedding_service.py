import numpy as np
from unittest.mock import MagicMock, patch

from app.services.embedding_service import EmbeddingService


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_text_returns_embedding(mock_model):
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3])

    mock_model.return_value = mock_instance

    service = EmbeddingService()

    result = service.embed_text("Artificial intelligence")

    assert result == [0.1, 0.2, 0.3]
    mock_instance.encode.assert_called_once_with("Artificial intelligence")


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_text_handles_empty_text(mock_model):
    service = EmbeddingService()

    assert service.embed_text("") == []
    assert service.embed_text("   ") == []

    mock_model.return_value.encode.assert_not_called()


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_chunks_returns_embeddings(mock_model):
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    mock_model.return_value = mock_instance

    service = EmbeddingService()

    chunks = [
        "Artificial intelligence",
        "Machine learning",
    ]

    result = service.embed_chunks(chunks)

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    mock_instance.encode.assert_called_once_with(chunks)


@patch("app.services.embedding_service.SentenceTransformer")
def test_embed_chunks_handles_empty_chunks(mock_model):
    service = EmbeddingService()

    assert service.embed_chunks([]) == []

    mock_model.return_value.encode.assert_not_called()


@patch("app.services.embedding_service.SentenceTransformer")
def test_default_model_name(mock_model):
    EmbeddingService()

    mock_model.assert_called_once_with("all-MiniLM-L6-v2")