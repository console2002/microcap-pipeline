"""Governance signal extraction using edgartools filings."""
from __future__ import annotations

import logging
import re
from typing import Dict

from edgar import Filing

logger = logging.getLogger(__name__)

WEAKNESS_PATTERN = re.compile(r"material weakness", re.IGNORECASE)
GOING_CONCERN_PATTERN = re.compile(r"going concern", re.IGNORECASE)


def extract_governance(filing: Filing) -> Dict:
    logger.debug(
        "extract_governance entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    parse_status = "no_data"
    auditor = None
    has_going_concern = False
    has_material_weakness = False
    governance_notes = ""
    text_blob = None

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed", exc_info=True)

    if obj is not None:
        parse_status = "edgar_obj_ok"
        auditor = getattr(obj, "auditor", None)
        governance_notes = getattr(obj, "management_discussion", "") or ""
        text_blob = str(governance_notes)

    if not text_blob:
        try:
            text_blob = filing.text()
        except Exception:
            logger.debug("filing.text() failed", exc_info=True)
            text_blob = None

    if text_blob:
        has_going_concern = bool(GOING_CONCERN_PATTERN.search(text_blob))
        has_material_weakness = bool(WEAKNESS_PATTERN.search(text_blob))

    result = {
        "auditor_name": auditor,
        "has_going_concern": has_going_concern,
        "has_material_weakness": has_material_weakness,
        "governance_notes": governance_notes,
        "governance_source_form": getattr(filing, "form", ""),
        "governance_source_date": getattr(filing, "filing_date", ""),
        "governance_source_url": getattr(filing, "text_url", ""),
        "parse_status": parse_status,
    }

    logger.debug(
        "extract_governance exit", extra={"form": filing.form, "filing_date": filing.filing_date, "parse_status": parse_status}
    )
    return result
