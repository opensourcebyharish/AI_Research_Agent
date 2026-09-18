from pathlib import Path
import logging

from app.core.logging import configure_logging


def test_configure_logging_creates_log_directory(tmp_path):
    log_file = tmp_path / "logs" / "app.log"

    configure_logging(
        level="INFO",
        log_file=log_file,
    )

    assert log_file.parent.exists()


def test_configure_logging_creates_log_file(tmp_path):
    log_file = tmp_path / "logs" / "app.log"

    configure_logging(
        level="INFO",
        log_file=log_file,
    )

    logger = logging.getLogger("ai_research_agent.test")
    logger.info("test log message")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "test log message" in log_file.read_text(
        encoding="utf-8"
    )


def test_configure_logging_accepts_string_path(tmp_path):
    log_file = tmp_path / "logs" / "app.log"

    configure_logging(
        level="INFO",
        log_file=str(log_file),
    )

    assert Path(log_file).parent.exists()