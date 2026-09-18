from unittest.mock import MagicMock

import pytest

from app.services.llm_service import LLMService


def test_llm_service_uses_default_model():
    service = LLMService()

    assert service.model_name == "llama3.2:3b"


def test_llm_service_accepts_custom_model():
    service = LLMService(
        model_name="custom-model"
    )

    assert service.model_name == "custom-model"


def test_generate_rejects_empty_prompt():
    service = LLMService()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        service.generate("")


def test_generate_rejects_whitespace_prompt():
    service = LLMService()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        service.generate("   ")


def test_generate_returns_ollama_response(monkeypatch):
    mock_response = {
        "message": {
            "content": "This is a generated answer."
        }
    }

    mock_chat = MagicMock(
        return_value=mock_response
    )

    monkeypatch.setattr(
        "app.services.llm_service.ollama.chat",
        mock_chat,
    )

    service = LLMService()

    result = service.generate(
        "What is Retrieval-Augmented Generation?"
    )

    assert result == "This is a generated answer."

    mock_chat.assert_called_once_with(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "What is Retrieval-Augmented Generation?"
                ),
            }
        ],
    )


def test_generate_strips_response(monkeypatch):
    mock_response = {
        "message": {
            "content": "  Generated answer.  \n"
        }
    }

    monkeypatch.setattr(
        "app.services.llm_service.ollama.chat",
        MagicMock(return_value=mock_response),
    )

    service = LLMService()

    result = service.generate(
        "Test prompt"
    )

    assert result == "Generated answer."