"""Shared logging helpers for runway parsing."""
from __future__ import annotations

import csv
import logging
import os
from datetime import UTC, datetime
from typing import Mapping, Optional

_LOGGER = logging.getLogger("parser_10q")

_DEBUG_OCF_PATH: Optional[str] = None
_DEBUG_OCF_INIT_FAILED = False
_DEBUG_OCF_HEADER = [
    "url",
    "form_type",
    "status",
    "note",
    "exhibit_href",
    "exhibit_doc_type",
    "html_source",
    "extra",
    "timestamp",
]


def _resolve_debug_ocf_path() -> Optional[str]:
    """Return the path for the debug OCF CSV, creating it on first use."""

    global _DEBUG_OCF_PATH, _DEBUG_OCF_INIT_FAILED

    if _DEBUG_OCF_INIT_FAILED:
        return None

    if _DEBUG_OCF_PATH is not None:
        return _DEBUG_OCF_PATH

    logs_dir: Optional[str] = None
    try:
        from app.config import load_config  # type: ignore

        cfg = load_config()
        logs_dir = str(cfg.get("Paths", {}).get("logs") or "./logs")
    except Exception:
        logs_dir = "./logs"

    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, "debugOCF.csv")
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(_DEBUG_OCF_HEADER)
        _DEBUG_OCF_PATH = path
        return path
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        _DEBUG_OCF_INIT_FAILED = True
        _LOGGER.debug("failed to initialise debug OCF log (%s)", exc)
        return None


def _serialize_mapping(mapping: Mapping[str, object]) -> str:
    parts = []
    for key, value in mapping.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _record_debug_ocf(
    *,
    url: str,
    form_type: Optional[str],
    status: str,
    note: str,
    exhibit_href: Optional[str],
    exhibit_doc_type: Optional[str],
    html_source: Optional[str],
    extra: Optional[Mapping[str, object]],
) -> None:
    path = _resolve_debug_ocf_path()
    if not path:
        return

    note_clean = " ".join((note or "").split())
    html_source_value = html_source or ""
    extra_serialized = _serialize_mapping(extra) if isinstance(extra, Mapping) else ""

    try:
        with open(path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    url,
                    form_type or "",
                    status,
                    note_clean,
                    exhibit_href or "",
                    exhibit_doc_type or "",
                    html_source_value,
                    extra_serialized,
                    datetime.now(UTC).isoformat(),
                ]
            )
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        _LOGGER.debug("failed to append debug OCF entry (%s)", exc)


def _maybe_record_debug_ocf(
    *,
    url: str,
    form_type: Optional[str],
    result: Mapping[str, object],
    html_info: Optional[Mapping[str, object]],
    extra: Optional[Mapping[str, object]],
) -> None:
    status_raw = str(result.get("status") or "").strip()
    status_upper = status_raw.upper()
    ocf_missing = (
        result.get("ocf_raw") is None
        and result.get("ocf_quarterly_raw") is None
        and result.get("ocf_quarterly") is None
    )

    if (
        not status_upper
        and not ocf_missing
        and "NOTIMPLEMENTED" not in status_upper
    ):
        return

    if (
        "MISSING OCF" not in status_upper
        and "NOTIMPLEMENTED" not in status_upper
        and not ocf_missing
    ):
        return

    note = str(result.get("note") or "")

    exhibit_href = result.get("exhibit_href")
    exhibit_doc_type = result.get("exhibit_doc_type")
    html_source = None

    if isinstance(html_info, Mapping):
        if not exhibit_href:
            exhibit_href = html_info.get("exhibit_href")
        if not exhibit_doc_type:
            exhibit_doc_type = html_info.get("exhibit_doc_type")
        html_source = html_info.get("source") or html_info.get("html_source")

    _record_debug_ocf(
        url=url,
        form_type=form_type or result.get("form_type"),
        status=status_raw,
        note=note,
        exhibit_href=str(exhibit_href or "") or None,
        exhibit_doc_type=str(exhibit_doc_type or "") or None,
        html_source=str(html_source or "") or None,
        extra=extra,
    )


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

    _maybe_record_debug_ocf(
        url=canonical_url,
        form_type=form_type,
        result=result,
        html_info=html_info,
        extra=extra,
    )


__all__ = ["log_parse_event", "log_runway_outcome"]
