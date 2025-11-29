"""Shared runway helpers for Weekly pipeline stages."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Tuple

from app.edgar_adapter import get_adapter
from edgar_core.runway import extract_runway_from_filing

logger = logging.getLogger(__name__)


def _extract_numeric(text: str) -> float | None:
    cleaned = text.replace(",", "")
    cleaned = cleaned.replace("(", "-").replace(")", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def compute_runway_from_html(html_text: str) -> float | None:
    if not html_text:
        return None
    cash_match = re.search(
        r"cash and cash equivalents[:\s]*\$?([\d,\.\(\)-]+)", html_text, re.IGNORECASE
    )
    burn_match = re.search(
        r"operating activities[:\s]*\$?([\d,\.\(\)-]+)", html_text, re.IGNORECASE
    )
    if not cash_match or not burn_match:
        return None
    cash_val = _extract_numeric(cash_match.group(1))
    burn_val = _extract_numeric(burn_match.group(1))
    if cash_val is None or burn_val is None or burn_val == 0:
        return None
    quarterly_burn = abs(burn_val)
    return round(cash_val / quarterly_burn, 2)


def _runway_from_filing(url: str, adapter=None) -> float | None:
    if not url:
        return None
    path = url
    if url.startswith("file://"):
        path = url.replace("file://", "")
    candidate = Path(path)
    if candidate.exists():
        html_text = candidate.read_text(encoding="utf-8", errors="ignore")
        return compute_runway_from_html(html_text)
    if str(url).startswith("http"):
        try:
            edgar_adapter = adapter or get_adapter()
            html_text = edgar_adapter.download_filing_text(str(url))
            if html_text:
                return compute_runway_from_html(html_text)
        except Exception:
            return None
    return None


def compute_runway_quarters(url: str, adapter=None) -> Tuple[float | None, bool]:
    """Return (runway_quarters, used_primary_parser)."""

    if not url:
        return None, False

    adapter = adapter or get_adapter()

    try:
        filing = adapter._resolve_filing(url)  # type: ignore[attr-defined]
    except Exception:
        filing = None
        logger.debug("runway_utils: _resolve_filing failed", exc_info=True)

    if filing is not None:
        try:
            result = extract_runway_from_filing(filing)
            quarters = result.get("runway_quarters")
            if quarters is not None and quarters > 0:
                return round(float(quarters), 2), True
        except Exception:
            logger.debug("runway_utils: extract_runway_from_filing failed", exc_info=True)

    fallback = _runway_from_filing(str(url), adapter=adapter)
    return fallback, False


__all__ = ["compute_runway_from_html", "compute_runway_quarters"]
