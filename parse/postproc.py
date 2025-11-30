"""Post-processing helpers for runway parsing."""
from __future__ import annotations

from typing import Iterable, Optional

from .units import round_half_up


def infer_runway_estimate(form_type: Optional[str]) -> str:
    if not form_type:
        return ""
    normalized = form_type.upper()
    if normalized.startswith(("10-K", "20-F", "40-F")):
        return "annual_div4"
    return "interim"


def finalize_runway_result(
    *,
    cash: Optional[float],
    ocf_raw: Optional[float],
    ocf_quarterly: Optional[float] = None,
    period_months: Optional[int],
    assumption: Optional[str],
    note: str,
    form_type: Optional[str],
    units_scale: Optional[int],
    status: Optional[str],
    source_tags: Optional[Iterable[str]] = None,
) -> dict:
    estimate = infer_runway_estimate(form_type)

    status_final = status or ""

    quarterly_burn_raw: Optional[float] = None
    runway_quarters_raw: Optional[float] = None

    ocf_quarterly_value: Optional[float] = None
    if ocf_quarterly is not None:
        ocf_quarterly_value = float(ocf_quarterly)
    elif ocf_raw is not None:
        ocf_quarterly_value = float(ocf_raw)

    if ocf_quarterly_value is None:
        status_final = status_final or "Missing OCF"
    elif ocf_quarterly_value >= 0:
        quarterly_burn_raw = 0.0
        status_final = status_final or "OCF positive (self-funding)"
    else:
        quarterly_burn_raw = abs(ocf_quarterly_value)
        if cash is None:
            status_final = status_final or "Missing cash"
        elif quarterly_burn_raw > 0:
            runway_quarters_raw = float(cash) / float(quarterly_burn_raw)
            status_final = status_final or "OK"
        else:
            status_final = status_final or "Missing OCF"

    cash_value = float(cash) if cash is not None else None

    runway_quarters_display: Optional[float] = None
    runway_months_display: Optional[float] = None
    if runway_quarters_raw is not None:
        runway_quarters_display = round(float(runway_quarters_raw), 2)
        runway_months_display = round(float(runway_quarters_raw) * 3, 2)

    result: dict = {
        "cash_raw": cash_value,
        "ocf_raw": ocf_raw,
        "ocf_quarterly_raw": ocf_quarterly_value,
        "quarterly_burn_raw": quarterly_burn_raw,
        "runway_quarters_raw": runway_quarters_raw,
        "runway_quarters_display": runway_quarters_display,
        "runway_months_display": runway_months_display,
        "note": note,
        "estimate": estimate,
        "status": status_final,
        "assumption": assumption or "",
        "period_months": period_months,
        "units_scale": units_scale or 1,
    }

    result["cash"] = round_half_up(cash_value)
    result["quarterly_burn"] = (
        round_half_up(quarterly_burn_raw) if quarterly_burn_raw is not None else None
    )
    result["runway_quarters"] = (
        round_half_up(runway_quarters_raw) if runway_quarters_raw is not None else None
    )

    if form_type:
        result["form_type"] = form_type

    months_valid = period_months in {3, 6, 9, 12}
    complete = cash_value is not None and ocf_quarterly_value is not None and months_valid
    result["complete"] = complete

    if not complete and status_final in {"", "OK"}:
        status_final = "Incomplete"

    result["status"] = status_final

    if source_tags:
        unique_tags = []
        for tag in source_tags:
            if tag and tag not in unique_tags:
                unique_tags.append(tag)
        if unique_tags:
            result["source_tags"] = unique_tags

    return result


__all__ = ["finalize_runway_result", "infer_runway_estimate"]
