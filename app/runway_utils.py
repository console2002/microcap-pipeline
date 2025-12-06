"""Shared runway helpers for Weekly pipeline stages."""

from __future__ import annotations

import logging
import re
from typing import Tuple

import pandas as pd

from app.edgar_adapter import get_adapter

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


def compute_runway_quarters(
    url: str,
    adapter=None,
    return_reason: bool = False,
    include_reason_meta: bool = False,
) -> (
    Tuple[float | None, bool]
    | Tuple[float | None, bool, str, str]
    | Tuple[float | None, bool, str, str, dict]
):
    """Return (runway_quarters, used_primary_parser[, reason_code, reason_detail, meta]).

    The primary path uses ``EdgarAdapter.runway_from_financials`` so gating
    logic aligns with the EDGAR-first pipeline.
    """

    if not url:
        result_tuple = (None, False)
        if return_reason:
            return (*result_tuple, "", "")
        return result_tuple

    adapter = adapter or get_adapter()
    reason_code = "PARSER_ERROR"
    reason_detail = ""
    reason_meta: dict[str, str] = {"error_type": "", "error_message": "", "error_stage": ""}

    try:
        primary_result = adapter.runway_from_financials(url, None)
    except Exception as exc:
        primary_result = None
        reason_meta = {
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "error_stage": "adapter.runway_from_financials",
        }
        logger.debug("runway_utils: runway_from_financials failed", exc_info=True)

    if primary_result:
        quarters = primary_result.get("runway_quarters")
        reason_code = primary_result.get("reason_code", "") or ""
        reason_detail = primary_result.get("reason_detail", "") or ""
        reason_meta = {
            "error_type": primary_result.get("error_type", "") or "",
            "error_message": primary_result.get("error_message", "") or "",
            "error_stage": primary_result.get("error_stage", "") or "",
        }

        if reason_code == "OK" and quarters is not None and quarters > 0:
            result_tuple = (round(float(quarters), 2), True)
            if return_reason:
                if include_reason_meta:
                    return (*result_tuple, reason_code, reason_detail, reason_meta)
                return (*result_tuple, reason_code, reason_detail)
            return result_tuple

    result_tuple = (None, False)
    if return_reason:
        if include_reason_meta:
            return (*result_tuple, reason_code or "", reason_detail or "", reason_meta)
        return (*result_tuple, reason_code or "", reason_detail or "")
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
        "FilingForm",
        "FiledAt",
        "Accession",
        "FilingAccession",
        "RunwayQuarters",
        "HasRunway",
        "RunwaySourceURL",
        "FilingURL",
        "RunwayReasonCode",
        "RunwayReasonDetail",
        "RunwayErrorType",
        "RunwayErrorMessage",
        "RunwayErrorStage",
        "Status",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    df["Status"] = df.get("Status", df["RunwayReasonCode"])
    df["FilingForm"] = df.get("FilingForm", df.get("Form"))
    df["FilingAccession"] = df.get("FilingAccession", df.get("Accession"))
    df["FilingURL"] = df.get("FilingURL", df.get("RunwaySourceURL"))

    mask = df["RunwayReasonCode"].fillna("").astype(str).str.upper().ne("OK")
    missing_quarters = df["RunwayQuarters"].isna()
    subset = df[mask | missing_quarters].copy()

    subset.to_csv(path, index=False)


__all__ = [
    "compute_runway_from_html",
    "compute_runway_quarters",
    "write_runway_diagnostics",
]
