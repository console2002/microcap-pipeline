"""Utility helpers for configuring module logging.

This keeps the CSV runlog/progress logging untouched while ensuring
edgar_core and WEEKLY modules emit to a file handler for diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
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


__all__ = ["setup_logging", "log_diag"]


_DIAG_CONFIG: dict | None = None


def _diagnostics_settings(cfg: dict | None = None) -> dict:
    global _DIAG_CONFIG

    if _DIAG_CONFIG is not None:
        return _DIAG_CONFIG

    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            cfg = {}

    diag_cfg = cfg.get("Diagnostics") or {}
    enabled = bool(diag_cfg.get("Enabled"))
    path = diag_cfg.get("Path") or os.path.join(
        cfg.get("Paths", {}).get("data", "data"), "run_diagnostics.jsonl"
    )

    _DIAG_CONFIG = {"enabled": enabled, "path": path}
    return _DIAG_CONFIG


def log_diag(
    *,
    stage: str,
    ticker: str,
    cik: str | None,
    decision: str,
    details: str,
    fields: dict | None = None,
    cfg: dict | None = None,
) -> None:
    settings = _diagnostics_settings(cfg)
    if not settings.get("enabled"):
        return

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "ticker": ticker,
        "cik": cik or "",
        "decision": decision,
        "details": details,
        "fields": fields or {},
    }

    try:
        path = settings.get("path") or "run_diagnostics.jsonl"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("diagnostics logging failed", exc_info=True)
