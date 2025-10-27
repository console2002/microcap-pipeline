"""Parser for extracting cash runway metrics from 10-Q/10-K filings."""
from __future__ import annotations

import re
from typing import Iterable, Optional
from urllib import request


_NUMERIC_TOKEN_PATTERN = re.compile(r"[\d\$\(\)\-]")
_NUMBER_SEARCH_PATTERN = re.compile(r"[\$\(]*[-+]?\d[\d,]*(?:\.\d+)?\)?")


def _to_float(num_str: str) -> float:
    """Convert a numeric string that may contain formatting into a float."""
    text = num_str.strip()
    match = _NUMBER_SEARCH_PATTERN.search(text)
    if not match:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    value = match.group(0)
    negative = False

    if "(" in value and ")" in value:
        negative = True

    cleaned = value.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    cleaned = cleaned.replace("−", "-").strip()

    if cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:]
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:]

    if not cleaned:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    number = float(cleaned)
    if negative:
        number = -number
    return number


def _extract_number_after_keyword(text: str, keywords: Iterable[str]) -> Optional[float]:
    """Find the first numeric value that appears after any keyword."""
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue

        remainder = text[match.end():match.end() + 500]
        tokens = remainder.split()
        for token in tokens[:6]:
            if not _NUMERIC_TOKEN_PATTERN.search(token):
                continue
            try:
                return _to_float(token)
            except ValueError:
                continue
    return None


def _strip_html(html_text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    without_tags = re.sub(r"<[^>]+>", " ", html_text)
    normalized = " ".join(without_tags.split())
    return normalized


def get_runway_from_filing(filing_url: str) -> dict:
    """Fetch a filing URL and estimate the cash runway metrics."""
    with request.urlopen(filing_url) as response:
        raw_bytes = response.read()
    html_text = raw_bytes.decode("utf-8", errors="ignore")
    text = _strip_html(html_text)

    cash_keywords = [
        "Cash and cash equivalents",
        "Cash and cash equivalents, end of period",
        "Cash and cash equivalents at end of period",
    ]
    cash_value = _extract_number_after_keyword(text, cash_keywords)

    burn_value = _extract_number_after_keyword(
        text,
        [
            "Net cash used in operating activities",
        ],
    )

    if burn_value is None:
        provided_value = _extract_number_after_keyword(
            text,
            [
                "Net cash provided by operating activities",
            ],
        )
        if provided_value is not None:
            quarterly_burn = 0.0
        else:
            quarterly_burn = None
    else:
        quarterly_burn = abs(burn_value)

    cash = float(cash_value) if cash_value is not None else None

    if quarterly_burn is not None and quarterly_burn <= 0:
        quarterly_burn = None

    runway_quarters: Optional[float]
    if cash is not None and quarterly_burn is not None and quarterly_burn > 0:
        runway_quarters = cash / quarterly_burn
    else:
        runway_quarters = None

    note = f"values parsed from 10-Q/10-K: {filing_url}"

    return {
        "cash": cash,
        "quarterly_burn": quarterly_burn,
        "runway_quarters": runway_quarters,
        "note": note,
    }


__all__ = ["get_runway_from_filing"]
