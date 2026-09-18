import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging(
    level: str = "INFO",
    log_file: str | Path = "data/logs/app.log",
) -> None:
    """
    Configure application-wide logging.

    Logs are written both to the console and to a persistent
    application log file.
    """

    numeric_level = getattr(
        logging,
        level.upper(),
        None,
    )

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: {level}"
        )

    log_path = Path(log_file)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatter = logging.Formatter(
        DEFAULT_LOG_FORMAT
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(numeric_level)

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    has_console_handler = any(
        isinstance(
            handler,
            logging.StreamHandler,
        )
        and not isinstance(
            handler,
            logging.FileHandler,
        )
        for handler in root_logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    resolved_log_path = log_path.resolve()

    file_handler = None

    for handler in root_logger.handlers:
        if isinstance(
            handler,
            logging.FileHandler,
        ):
            if (
                Path(handler.baseFilename).resolve()
                == resolved_log_path
            ):
                file_handler = handler
                break

    if file_handler is None:
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)


logger = logging.getLogger(
    "ai_research_agent"
)