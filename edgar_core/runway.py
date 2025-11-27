"""Runway extraction helpers using edgartools data objects when available."""
from __future__ import annotations

import logging
from datetime import timedelta
from decimal import Decimal
from typing import Dict, Optional

from edgar import Filing

logger = logging.getLogger(__name__)


def _decimal_value(value) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _extract_cash_and_burn_from_financials(financials) -> tuple[Optional[Decimal], Optional[Decimal]]:
    cash = None
    burn = None

    try:
        balance = financials.balance_sheet()
        if balance:
            rendered = balance.render(standard=True)
            if hasattr(rendered, "to_dataframe"):
                df = rendered.to_dataframe()
                if not df.empty:
                    for column in df.columns:
                        if column.lower().startswith("three") or column.lower().startswith("six") or column.lower().startswith("twelve"):
                            candidate = df[column].iloc[0]
                            cash = _decimal_value(candidate)
                            break
    except Exception:
        logger.debug("failed to read balance_sheet from financials", exc_info=True)

    try:
        cashflow = financials.cashflow_statement()
        if cashflow:
            rendered = cashflow.render(standard=True)
            if hasattr(rendered, "to_dataframe"):
                df = rendered.to_dataframe()
                if not df.empty:
                    for column in df.columns:
                        if column.lower().startswith("three") or column.lower().startswith("six") or column.lower().startswith("twelve"):
                            candidate = df[column].iloc[0]
                            burn = _decimal_value(candidate)
                            break
    except Exception:
        logger.debug("failed to read cashflow_statement from financials", exc_info=True)

    return cash, burn


def extract_runway_from_filing(filing: Filing) -> Dict:
    """Return a dictionary compatible with runway_extract expectations.

    The helper favors edgartools data objects and falls back to minimal
    placeholders so that existing CSV schemas remain intact.
    """

    logger.debug(
        "extract_runway_from_filing entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    cash = None
    burn = None
    parse_status = "no_data"
    note = ""

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed; falling back", exc_info=True)

    if obj is not None:
        financials = getattr(obj, "financials", None)
        if financials is not None:
            parse_status = "edgar_obj_ok"
            cash, burn = _extract_cash_and_burn_from_financials(financials)

    source_form = getattr(filing, "form", "")
    source_date = getattr(filing, "filing_date", "")
    source_url = getattr(filing, "text_url", "")

    runway_quarters = None
    runway_days = None
    quarterly_burn = burn

    if cash is not None and burn is not None and burn != 0:
        try:
            # burn is usually negative for cash outflow
            normalized_burn = burn.copy_abs()
            runway_quarters = float(cash / normalized_burn)
            runway_days = int(runway_quarters * 90)
        except Exception:
            logger.debug("failed to compute runway from cash/burn", exc_info=True)

    result = {
        "cash": cash,  # expected to align with RunwayCash* columns
        "quarterly_burn": quarterly_burn,
        "runway_quarters": runway_quarters,
        "runway_days": runway_days,
        "source_form": source_form,
        "source_date": source_date,
        "source_url": source_url,
        "note": note,
        "parse_status": parse_status,
    }

    logger.debug(
        "extract_runway_from_filing exit", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", ""), "parse_status": parse_status}
    )
    return result
