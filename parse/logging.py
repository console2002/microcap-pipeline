"""Shared logging helpers for runway parsing."""
from __future__ import annotations

import csv
import logging
import os
from typing import Mapping, Optional
from urllib.parse import urlparse

_LOGGER = logging.getLogger("parser_10q")

_DEBUG_OCF_PATH: Optional[str] = None
_DEBUG_OCF_INIT_FAILED = False
_DEBUG_OCF_HEADER = [
    "url",
    "form_type",
    "status",
    "note",
    "source",
    "extra",
]

_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".svg",
)


def _is_image_href(href: Optional[str]) -> bool:
    """Return True when the supplied href points to an image asset."""

    if not href:
        return False

    try:
        parsed = urlparse(href)
        path = parsed.path or href
    except Exception:
        path = href

    lowered = path.lower()
    return any(lowered.endswith(ext) for ext in _IMAGE_EXTENSIONS)


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
        if os.path.exists(path):
            try:
                with open(path, "r", newline="", encoding="utf-8") as handle:
                    reader_rows = list(csv.reader(handle))
            except Exception:
                reader_rows = []
            if reader_rows:
                header = reader_rows[0]
                if header and header[-1].strip().lower() == "timestamp":
                    trimmed_rows = [
                        row[: len(_DEBUG_OCF_HEADER)] if row else []
                        for row in reader_rows[1:]
                    ]
                    with open(path, "w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(_DEBUG_OCF_HEADER)
                        writer.writerows(trimmed_rows)
            else:
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(_DEBUG_OCF_HEADER)
        else:
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
    source: Optional[str],
    extra: Optional[Mapping[str, object]],
) -> None:
    path = _resolve_debug_ocf_path()
    if not path:
        return

    note_clean = " ".join((note or "").split())
    extra_serialized = _serialize_mapping(extra) if isinstance(extra, Mapping) else ""
    normalized_url = (url or "").strip()
    source_value = (source or "").strip()
    form_value = (form_type or "").strip()

    try:
        with open(path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    normalized_url,
                    form_value,
                    status,
                    note_clean,
                    source_value,
                    extra_serialized,
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

    incomplete = not bool(result.get("complete"))
    success_status = False
    if status_upper:
        if status_upper.startswith("OK"):
            success_status = True
        elif status_upper == "OCF POSITIVE (SELF-FUNDING)":
            success_status = True
    failure_status = bool(status_raw) and not success_status
    should_record = ocf_missing or incomplete or failure_status
    if not should_record:
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

    if _is_image_href(exhibit_href):
        exhibit_href = None
        exhibit_doc_type = None
        html_source = None if html_source == "exhibit" else html_source

    combined_extra: dict[str, object] = {}
    if isinstance(extra, Mapping):
        combined_extra.update({k: v for k, v in extra.items() if v is not None})
    if exhibit_href:
        combined_extra.setdefault("exhibit_href", exhibit_href)
    if exhibit_doc_type:
        combined_extra.setdefault("exhibit_doc_type", exhibit_doc_type)
    if html_source:
        combined_extra.setdefault("html_source", html_source)

    source_label = None
    if exhibit_doc_type:
        source_label = str(exhibit_doc_type)
    elif html_source:
        source_label = str(html_source)

    _record_debug_ocf(
        url=url,
        form_type=form_type or result.get("form_type"),
        status=status_raw,
        note=note,
        source=source_label,
        extra=combined_extra,
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


def log_exhibit_attempt(
    *,
    filing_url: str,
    form_type: Optional[str],
    exhibit_url: str,
    exhibit_doc_type: Optional[str],
    status: str,
    note: str = "",
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Record a single exhibit attempt in the debug OCF log."""

    if _is_image_href(exhibit_url):
        return

    display_form = "Exhibit"
    if form_type:
        display_form = f"Exhibit ({form_type})"

    combined_extra: dict[str, object] = {"filing_url": filing_url}
    if isinstance(extra, Mapping):
        combined_extra.update({k: v for k, v in extra.items() if v is not None})

    _record_debug_ocf(
        url=exhibit_url,
        form_type=display_form,
        status=status,
        note=note,
        source=exhibit_doc_type or "exhibit",
        extra=combined_extra,
    )


__all__ = ["log_parse_event", "log_runway_outcome", "log_exhibit_attempt"]
