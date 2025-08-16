import logging
import os
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger once per module.

    - Level comes from LOG_LEVEL env var (default INFO)
    - Avoids duplicate handlers when re-imported in reload mode
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = _LEVEL_MAP.get(level_str, logging.INFO)

    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
