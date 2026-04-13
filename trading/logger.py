"""
trading/logger.py
─────────────────
Centralised logging factory.

Usage:
    from trading.logger import get_logger
    logger = get_logger(__name__)

Each unique name gets:
  • A StreamHandler → stdout
  • A RotatingFileHandler → logs/{name}.log  (10 MB × 5 backups)

Calling get_logger() with the same name is idempotent — handlers are never
duplicated even if the function is called multiple times.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from trading.config import get_settings

_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name*, adding handlers exactly once.
    name is typically ``__name__`` from the calling module.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — idempotent

    settings = get_settings()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

    # ── Console ───────────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── Rotating file ─────────────────────────────────────────────────────────
    # Derive filename from the logger name: "trading.strategies.wheel" → "trading.strategies.wheel.log"
    log_file = settings.log_dir / f"{name}.log"
    rotating = RotatingFileHandler(
        log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)

    # Prevent messages from bubbling up to the root logger (avoids duplicates
    # when the root logger also has a StreamHandler configured externally).
    logger.propagate = False

    return logger
