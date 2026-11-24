"""Structured console and rotating file logging configuration for ATLAS AI."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from atlas_ai.core.config import AppSettings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3


def setup_logging(settings: AppSettings) -> logging.Logger:
    """Configure the root 'atlas_ai' logger with console and optional rotating file handlers.

    Args:
        settings: Validated AppSettings instance.

    Returns:
        Configured top-level 'atlas_ai' logger.
    """
    logger = logging.getLogger("atlas_ai")
    logger.setLevel(getattr(logging, settings.log_level))
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs on repeated initialization
    if logger.handlers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if settings.log_to_file:
        log_dir: Path = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "atlas_ai.log"

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, settings.log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped under the 'atlas_ai' namespace."""
    if name.startswith("atlas_ai"):
        return logging.getLogger(name)
    return logging.getLogger(f"atlas_ai.{name}")
