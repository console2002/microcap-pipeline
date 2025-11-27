"""Ownership and holdings extraction."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from edgar import Filing

logger = logging.getLogger(__name__)

BENEFICIAL_PATTERN = re.compile(r"(\d+\.\d+)%", re.IGNORECASE)
SHARES_PATTERN = re.compile(r"(\d[\d,]+)\s+shares", re.IGNORECASE)
GROUP_PATTERN = re.compile(r"group", re.IGNORECASE)


def extract_13f_holdings(filing: Filing) -> Optional[List[Dict]]:
    logger.debug(
        "extract_13f_holdings entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed", exc_info=True)

    if obj is None:
        return None

    infotable = getattr(obj, "infotable", None)
    if infotable is None:
        return None

    holdings: List[Dict] = []
    try:
        df = infotable
        for _, row in df.iterrows():
            holdings.append(
                {
                    "issuer": row.get("nameOfIssuer"),
                    "ticker": row.get("sshPrnamtType"),
                    "cusip": row.get("cusip"),
                    "shares": row.get("sshPrnamt"),
                    "value": row.get("value"),
                    "type": row.get("sshPrnamtType"),
                    "put_call": row.get("putCall"),
                }
            )
    except Exception:
        logger.debug("failed to iterate infotable", exc_info=True)
        return None

    return holdings


def extract_13d_g_positions(filing: Filing) -> Optional[Dict]:
    logger.debug(
        "extract_13d_g_positions entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    try:
        text_blob = filing.text()
    except Exception:
        logger.debug("filing.text() failed", exc_info=True)
        return None

    if not text_blob:
        return None

    percent = None
    shares = None
    group = False

    percent_match = BENEFICIAL_PATTERN.search(text_blob)
    if percent_match:
        try:
            percent = float(percent_match.group(1))
        except Exception:
            percent = None

    shares_match = SHARES_PATTERN.search(text_blob)
    if shares_match:
        try:
            shares = int(shares_match.group(1).replace(",", ""))
        except Exception:
            shares = None

    group = bool(GROUP_PATTERN.search(text_blob))

    if percent is None and shares is None:
        return None

    return {"beneficial_percent": percent, "total_shares": shares, "group_flag": group}
