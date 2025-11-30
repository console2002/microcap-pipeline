"""Utility helpers for configuring module logging.

This keeps the CSV runlog/progress logging untouched while ensuring
edgar_core and WEEKLY modules emit to a file handler for diagnostics.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from app.config import load_config

_LOGGING_CONFIGURED = False


def _has_file_handler(handlers: Iterable[logging.Handler], target_path: str) -> bool:
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == os.path.abspath(target_path):
                    return True
            except Exception:
                continue
    return False


def setup_logging() -> None:
    """Attach a simple file handler for app and edgar_core loggers.

    The handler is idempotent and will not be re-added across repeated calls
    (e.g., during tests). Existing CSV logging remains unchanged.
    """

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "app.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not _has_file_handler(root_logger.handlers, log_path):
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    _LOGGING_CONFIGURED = True


__all__ = ["setup_logging"]
