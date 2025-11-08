"""Shared logging helpers for runway parsing."""
from __future__ import annotations

import logging
from typing import Optional

_LOGGER = logging.getLogger("parser_10q")


def log_parse_event(level: int, message: str, **fields: object) -> None:
    """Log a structured parsing event with optional contextual fields."""
    details: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        details.append(f"{key}={value}")
    if details:
        message = f"{message} ({', '.join(details)})"
    _LOGGER.log(level, message)


def log_runway_outcome(
    canonical_url: str,
    form_type: Optional[str],
    result: dict,
    *,
    html_info: Optional[dict] = None,
    xbrl_result: Optional[dict] = None,
    xbrl_error: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Log the final runway parsing outcome with normalized fields."""
    sources = None
    if "source_tags" in result:
        tags = result.get("source_tags") or []
        if isinstance(tags, (list, tuple)):
            sources = ",".join(str(tag) for tag in tags if tag)
        else:
            sources = str(tags)

    fields: dict[str, object] = {
        "url": canonical_url,
        "form_type": form_type or result.get("form_type"),
        "status": result.get("status"),
        "complete": result.get("complete"),
        "cash_raw": result.get("cash_raw"),
        "ocf_raw": result.get("ocf_raw"),
        "period_months": result.get("period_months"),
        "sources": sources,
    }

    if html_info is not None:
        fields.update(
            {
                "html_header": html_info.get("found_cashflow_header"),
                "html_cash": html_info.get("cash_value"),
                "html_ocf": html_info.get("ocf_value"),
                "html_period_inferred": html_info.get("period_months_inferred"),
                "html_units_scale": html_info.get("units_scale"),
                "html_evidence": html_info.get("evidence"),
            }
        )

    if xbrl_result is not None:
        fields.update(
            {
                "xbrl_cash": xbrl_result.get("cash_raw"),
                "xbrl_ocf": xbrl_result.get("ocf_raw"),
                "xbrl_period": xbrl_result.get("period_months"),
            }
        )

    if xbrl_error:
        fields["xbrl_error"] = xbrl_error

    if extra:
        fields.update({k: v for k, v in extra.items()})

    note = result.get("note")
    if isinstance(note, str):
        sanitized = note.replace("\n", " ")
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."
        fields["note"] = sanitized

    level = logging.INFO if result.get("complete") else logging.WARNING
    log_parse_event(level, "runway parse outcome", **fields)


__all__ = ["log_parse_event", "log_runway_outcome"]
