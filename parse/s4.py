"""Placeholder parser for not-yet-supported forms."""
from __future__ import annotations

import logging

from .logging import log_parse_event
from .postproc import finalize_runway_result

FORM_LABEL = "S-4"

def parse(url: str, html: str | None = None, form_hint: str | None = None) -> dict:
    note = f"Parsing for {FORM_LABEL} filings is not implemented: {url}"
    log_parse_event(logging.INFO, "runway form not implemented", url=url, form=FORM_LABEL)
    result = finalize_runway_result(
        cash=None,
        ocf_raw=None,
        ocf_quarterly=None,
        period_months=None,
        assumption=None,
        note=note,
        form_type=form_hint or FORM_LABEL,
        units_scale=1,
        status="NotImplemented",
        source_tags=None,
    )
    result['complete'] = False
    return result

__all__ = ['parse']
