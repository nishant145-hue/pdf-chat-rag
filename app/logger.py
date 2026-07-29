import logging
from logging.handlers import RotatingFileHandler

from app.config import settings


def setup_logger() -> logging.Logger:
    """
    Configure the application logger.

    Logs are written to:
    - Console
    - Rotating log file
    """

    settings.LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("pdf_rag")

    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------------------------------------------------------
    # Console Handler
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # File Handler
    # ---------------------------------------------------------

    file_handler = RotatingFileHandler(
        settings.LOG_DIR / settings.LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # Register Handlers
    # ---------------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = setup_logger()