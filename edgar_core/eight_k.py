"""8-K extraction helpers using edgartools data objects."""
from __future__ import annotations

import logging
from typing import Dict

from edgar import Filing

logger = logging.getLogger(__name__)


def extract_8k_event(filing: Filing) -> Dict:
    logger.debug(
        "extract_8k_event entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    parse_status = "no_data"
    items_raw = []
    items_normalized = []
    has_press_release = False
    has_exhibits = False
    event_date = getattr(filing, "filing_date", None)
    event_text_summary = ""

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed", exc_info=True)

    if obj is not None:
        parse_status = "edgar_obj_ok"
        items_raw = list(getattr(obj, "items", []) or [])
        items_normalized = [str(item) for item in items_raw]
        has_press_release = bool(getattr(obj, "has_press_release", False))
        has_exhibits = bool(getattr(obj, "press_releases", []))
        event_date = getattr(obj, "date_of_report", event_date)
        first_item = items_raw[0] if items_raw else None
        if isinstance(first_item, str):
            event_text_summary = first_item

    result = {
        "form": getattr(filing, "form", "8-K"),
        "cik": getattr(filing, "cik", ""),
        "ticker": getattr(filing, "ticker", ""),
        "filing_date": getattr(filing, "filing_date", ""),
        "event_date": event_date,
        "items_raw": items_raw,
        "items_normalized": items_normalized,
        "has_press_release": has_press_release,
        "has_exhibits": has_exhibits,
        "event_text_summary": event_text_summary,
        "parse_status": parse_status,
    }

    logger.debug(
        "extract_8k_event exit", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", ""), "parse_status": parse_status}
    )
    return result
