import importlib


def test_default_settings(monkeypatch):
    """Application settings should have safe production defaults."""

    monkeypatch.delenv(
        "APP_NAME",
        raising=False,
    )

    monkeypatch.delenv(
        "APP_ENV",
        raising=False,
    )

    monkeypatch.delenv(
        "DEBUG",
        raising=False,
    )

    monkeypatch.delenv(
        "LLM_MODEL",
        raising=False,
    )

    monkeypatch.delenv(
        "EMBEDDING_DIMENSION",
        raising=False,
    )

    import app.core.config as config

    config = importlib.reload(config)

    assert config.settings.app_name == (
        "AI Research Agent"
    )

    assert config.settings.app_env == (
        "development"
    )

    assert config.settings.debug is False

    assert config.settings.llm_model == (
        "llama3.2:3b"
    )

    assert config.settings.embedding_dimension == 384


def test_environment_settings(monkeypatch):
    """Environment variables should override defaults."""

    monkeypatch.setenv(
        "APP_NAME",
        "Production Research Agent",
    )

    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
        "DEBUG",
        "true",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "custom-model",
    )

    monkeypatch.setenv(
        "EMBEDDING_DIMENSION",
        "768",
    )

    monkeypatch.setenv(
        "MAX_UPLOAD_SIZE_MB",
        "50",
    )

    monkeypatch.setenv(
        "RETRIEVAL_MAX_DISTANCE",
        "2.5",
    )

    import app.core.config as config

    config = importlib.reload(config)

    assert config.settings.app_name == (
        "Production Research Agent"
    )

    assert config.settings.app_env == (
        "production"
    )

    assert config.settings.debug is True

    assert config.settings.llm_model == (
        "custom-model"
    )

    assert config.settings.embedding_dimension == 768

    assert config.settings.max_upload_size_mb == 50

    assert config.settings.max_upload_size_bytes == (
        50 * 1024 * 1024
    )

    assert config.settings.retrieval_max_distance == 2.5


def test_allowed_extensions_are_normalized(monkeypatch):
    """Allowed upload extensions should be normalized."""

    monkeypatch.setenv(
        "ALLOWED_FILE_EXTENSIONS",
        ".PDF, .pdf",
    )

    import app.core.config as config

    config = importlib.reload(config)

    assert config.settings.allowed_file_extensions == (
        ".pdf",
        ".pdf",
    )


def test_invalid_integer_configuration(monkeypatch):
    """Invalid integer configuration should fail clearly."""

    monkeypatch.setenv(
        "API_PORT",
        "not-a-number",
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="API_PORT must be an integer",
    ):
        import app.core.config as config

        importlib.reload(config)


def test_invalid_float_configuration(monkeypatch):
    """Invalid float configuration should fail clearly."""

    monkeypatch.setenv(
        "RETRIEVAL_MAX_DISTANCE",
        "invalid",
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="RETRIEVAL_MAX_DISTANCE must be a number",
    ):
        import app.core.config as config

        importlib.reload(config)