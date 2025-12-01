"""Shared runway helpers for Weekly pipeline stages."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

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
    """Legacy regex-based runway fallback for HTML fragments."""

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


def compute_runway_quarters(
    url: str, adapter=None, return_reason: bool = False
) -> Tuple[float | None, bool] | Tuple[float | None, bool, str, str]:
    """Return (runway_quarters, used_primary_parser[, reason_code, reason_detail]).

    The primary path uses ``EdgarAdapter.runway_from_financials`` so gating
    logic aligns with the EDGAR-first pipeline. HTML parsing is only used as a
    last-resort fallback.
    """

    if not url:
        result_tuple = (None, False)
        if return_reason:
            return (*result_tuple, "", "")
        return result_tuple

    adapter = adapter or get_adapter()
    reason_code = "PARSER_ERROR"
    reason_detail = ""

    try:
        primary_result = adapter.runway_from_financials(url, None)
    except Exception:
        primary_result = None
        logger.debug("runway_utils: runway_from_financials failed", exc_info=True)
    else:
        if primary_result:
            quarters = primary_result.get("runway_quarters")
            reason_code = primary_result.get("reason_code", "")
            reason_detail = primary_result.get("reason_detail", "")
            if quarters is not None and quarters > 0:
                result_tuple = (round(float(quarters), 2), True)
                if return_reason:
                    return (*result_tuple, reason_code or "OK", reason_detail or "")
                return result_tuple

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
                result_tuple = (round(float(quarters), 2), True)
                if return_reason:
                    return (*result_tuple, "OK", "")
                return result_tuple
        except Exception:
            logger.debug("runway_utils: extract_runway_from_filing failed", exc_info=True)

    fallback = _runway_from_filing(str(url), adapter=adapter)
    result_tuple = (fallback, False)
    if return_reason:
        fallback_code = reason_code or ("OK" if fallback else "")
        return (*result_tuple, fallback_code, reason_detail)
    return result_tuple


def write_runway_diagnostics(records, path: str) -> None:
    """Write a lightweight diagnostics CSV for runway computation issues."""

    df = pd.DataFrame(records)
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    columns = [
        "Ticker",
        "CIK",
        "Form",
        "FiledAt",
        "Accession",
        "RunwayQuarters",
        "HasRunway",
        "RunwaySourceURL",
        "RunwayReasonCode",
        "RunwayReasonDetail",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    mask = df["RunwayReasonCode"].fillna("").astype(str).str.upper().ne("OK")
    missing_quarters = df["RunwayQuarters"].isna()
    subset = df[mask | missing_quarters].copy()

    subset.to_csv(path, index=False)


__all__ = [
    "compute_runway_from_html",
    "compute_runway_quarters",
    "write_runway_diagnostics",
]
