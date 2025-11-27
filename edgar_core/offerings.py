"""Offering and dilution helper built on top of edgartools filings."""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from edgar import Filing

logger = logging.getLogger(__name__)

ATM_PATTERN = re.compile(r"at[- ]the[- ]market", re.IGNORECASE)


def extract_offering(filing: Filing) -> Optional[Dict]:
    logger.debug(
        "extract_offering entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    text_blob = None
    try:
        text_blob = filing.text()
    except Exception:
        logger.debug("filing.text() failed", exc_info=True)

    if not text_blob:
        try:
            text_blob = filing.html()
        except Exception:
            logger.debug("filing.html() failed", exc_info=True)

    if not text_blob:
        logger.debug(
            "extract_offering exit", extra={"form": filing.form, "filing_date": filing.filing_date, "parse_status": "no_data"}
        )
        return None

    is_atm = bool(ATM_PATTERN.search(text_blob))

    result = {
        "security_type": None,
        "total_shares": None,
        "price_per_share": None,
        "gross_proceeds": None,
        "is_primary": None,
        "is_resale": None,
        "is_atm": is_atm,
        "source_form": getattr(filing, "form", ""),
        "source_date": getattr(filing, "filing_date", ""),
        "source_url": getattr(filing, "text_url", ""),
        "parse_status": "text_scanned",
        "note": "heuristic scan only",
    }

    logger.debug(
        "extract_offering exit", extra={"form": filing.form, "filing_date": filing.filing_date, "parse_status": result["parse_status"]}
    )
    return result
