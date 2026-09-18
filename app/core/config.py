import os
from dataclasses import dataclass
from pathlib import Path


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable safely."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable safely."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc


def _get_float(
    name: str,
    default: float | None,
) -> float | None:
    """Read a floating-point environment variable safely."""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number."
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = os.getenv(
        "APP_NAME",
        "AI Research Agent",
    )

    app_version: str = os.getenv(
        "APP_VERSION",
        "1.0.0",
    )

    app_env: str = os.getenv(
        "APP_ENV",
        "development",
    )

    debug: bool = _get_bool(
        "DEBUG",
        False,
    )

    api_host: str = os.getenv(
        "API_HOST",
        "0.0.0.0",
    )

    api_port: int = _get_int(
        "API_PORT",
        8000,
    )

    llm_provider: str = os.getenv(
        "LLM_PROVIDER",
        "ollama",
    )

    llm_model: str = os.getenv(
        "LLM_MODEL",
        "llama3.2:3b",
    )

    ollama_host: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2",
    )

    embedding_dimension: int = _get_int(
        "EMBEDDING_DIMENSION",
        384,
    )

    retrieval_top_k: int = _get_int(
        "RETRIEVAL_TOP_K",
        5,
    )

    retrieval_max_distance: float | None = _get_float(
        "RETRIEVAL_MAX_DISTANCE",
        1.5,
    )

    vector_store_dir: Path = Path(
        os.getenv(
            "VECTOR_STORE_DIR",
            "data/index",
        )
    )

    document_registry_path: Path = Path(
        os.getenv(
            "DOCUMENT_REGISTRY_PATH",
            "data/documents.json",
        )
    )

    max_upload_size_mb: int = _get_int(
        "MAX_UPLOAD_SIZE_MB",
        25,
    )

    allowed_file_extensions: tuple[str, ...] = tuple(
        extension.strip().lower()
        for extension in os.getenv(
            "ALLOWED_FILE_EXTENSIONS",
            ".pdf",
        ).split(",")
        if extension.strip()
    )

    log_level: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )

    log_file: Path = Path(
        os.getenv(
            "LOG_FILE",
            "data/logs/app.log",
        )
    )

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the maximum upload size in bytes."""

        return (
            self.max_upload_size_mb
            * 1024
            * 1024
        )


settings = Settings()