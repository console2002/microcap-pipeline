"""Runway extraction helpers using edgartools data objects when available."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Optional

from edgar import Filing

logger = logging.getLogger(__name__)


def _decimal_value(value) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _infer_months_from_label(label: str) -> Optional[int]:
    lower = label.lower()
    if lower.startswith("three") or lower.startswith("3"):
        return 3
    if lower.startswith("six") or lower.startswith("6"):
        return 6
    if lower.startswith("nine") or lower.startswith("9"):
        return 9
    if lower.startswith("twelve") or lower.startswith("12") or "year" in lower:
        return 12
    return None


def _extract_cash_and_burn_from_financials(
    financials,
) -> tuple[Optional[Decimal], Optional[Decimal], Optional[int]]:
    cash = None
    burn = None
    period_months = None

    try:
        balance = financials.balance_sheet()
        if balance:
            rendered = balance.render(standard=True)
            if hasattr(rendered, "to_dataframe"):
                df = rendered.to_dataframe()
                if not df.empty:
                    for column in df.columns:
                        inferred = _infer_months_from_label(str(column))
                        if inferred:
                            period_months = period_months or inferred
                        if inferred in {3, 6, 9, 12}:
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
                        inferred = _infer_months_from_label(str(column))
                        if inferred:
                            period_months = period_months or inferred
                        if inferred in {3, 6, 9, 12}:
                            candidate = df[column].iloc[0]
                            burn = _decimal_value(candidate)
                            break
    except Exception:
        logger.debug("failed to read cashflow_statement from financials", exc_info=True)

    return cash, burn, period_months


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
    period_months = None

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed; falling back", exc_info=True)

    if obj is not None:
        financials = getattr(obj, "financials", None)
        if financials is not None:
            parse_status = "edgar_obj_ok"
            cash, burn, period_months = _extract_cash_and_burn_from_financials(financials)

    source_form = getattr(filing, "form", "")
    source_date = getattr(filing, "filing_date", "")
    source_url = getattr(filing, "text_url", "")

    cash_value = float(cash) if cash is not None else None
    ocf_value = float(burn) if burn is not None else None

    from parse.units import normalize_ocf_value
    from parse.postproc import finalize_runway_result

    ocf_quarterly, period_months, assumption = normalize_ocf_value(ocf_value, period_months)

    result = finalize_runway_result(
        cash=cash_value,
        ocf_raw=ocf_value,
        ocf_quarterly=ocf_quarterly,
        period_months=period_months,
        assumption=assumption,
        note="values parsed from edgartools financials (legacy helper)",
        form_type=source_form,
        units_scale=1,
        status=None,
        source_tags=["XBRL"],
    )

    result.update(
        {
            "source_form": source_form,
            "source_date": source_date,
            "source_url": source_url,
            "parse_status": parse_status,
        }
    )

    logger.debug(
        "extract_runway_from_filing exit", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", ""), "parse_status": parse_status}
    )
    return result
