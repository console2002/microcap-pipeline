"""Parser for 6-K filings."""
from __future__ import annotations

import logging
from typing import Optional
from urllib.error import HTTPError, URLError

from .exhibits import follow_exhibits_and_parse
from .htmlutil import parse_html_cashflow_sections, strip_html, unescape_html_entities
from .ixbrl import derive_from_xbrl
from .logging import log_parse_event, log_runway_outcome
from .postproc import finalize_runway_result
from .router import (
    _extract_form_hint_from_url,  # type: ignore[attr-defined]
    _form_defaults,  # type: ignore[attr-defined]
    _infer_form_type_from_text,  # type: ignore[attr-defined]
    _infer_form_type_from_url,  # type: ignore[attr-defined]
    _normalize_form_type,  # type: ignore[attr-defined]
    _extract_note_suffix,  # type: ignore[attr-defined]
    _fetch_url,  # type: ignore[attr-defined]
)
from .units import normalize_ocf_value


def _merge_results(
    *,
    canonical_url: str,
    form_type: Optional[str],
    html_info: dict,
    html_cash: Optional[float],
    html_ocf_raw_value: Optional[float],
    html_period_inferred: Optional[int],
    html_units_scale: int,
    html_assumption: str,
    html_source: str,
    exhibit_status: Optional[str],
    exhibit_href: Optional[str],
    exhibit_doc_type: Optional[str],
    default_period: Optional[int],
    xbrl_result: Optional[dict],
    xbrl_error: Optional[str],
) -> dict:
    xbrl_cash_value = xbrl_result.get("cash_raw") if xbrl_result else None
    xbrl_ocf_raw_value = xbrl_result.get("ocf_raw") if xbrl_result else None
    xbrl_units_scale = xbrl_result.get("units_scale") if xbrl_result else 1
    xbrl_period = xbrl_result.get("period_months") if xbrl_result else None
    xbrl_assumption = xbrl_result.get("assumption") if xbrl_result else ""

    final_cash = None
    cash_source = None
    if html_cash is not None:
        final_cash = float(html_cash)
        cash_source = "HTML"
    elif xbrl_cash_value is not None:
        final_cash = float(xbrl_cash_value)
        cash_source = "XBRL"

    xbrl_ocf_quarterly = None
    if xbrl_result and xbrl_ocf_raw_value is not None:
        normalized = xbrl_result.get("ocf_quarterly_raw")
        if normalized is not None:
            xbrl_ocf_quarterly = float(normalized)

    html_ocf_normalized = None
    normalized_period = html_period_inferred
    if html_ocf_raw_value is not None:
        html_ocf_normalized, normalized_period, html_assumption = normalize_ocf_value(
            html_ocf_raw_value, html_period_inferred
        )
        if normalized_period not in {3, 6, 9, 12} and html_period_inferred in {3, 6, 9, 12}:
            normalized_period = html_period_inferred

    final_period = xbrl_period if xbrl_period in {3, 6, 9, 12} else None
    if final_period is None and html_period_inferred in {3, 6, 9, 12}:
        final_period = html_period_inferred
    if final_period is None and normalized_period in {3, 6, 9, 12}:
        final_period = normalized_period
    if final_period is None and default_period in {3, 6, 9, 12}:
        final_period = default_period

    final_ocf_raw = None
    final_ocf_quarterly = None
    final_assumption = ""
    ocf_source: Optional[str] = None
    if xbrl_ocf_quarterly is not None:
        final_ocf_raw = xbrl_ocf_raw_value
        final_ocf_quarterly = xbrl_ocf_quarterly
        final_assumption = xbrl_assumption
        ocf_source = "XBRL"
    elif html_ocf_normalized is not None:
        final_ocf_raw = html_ocf_raw_value
        final_ocf_quarterly = html_ocf_normalized
        final_assumption = html_assumption
        ocf_source = "HTML"
    else:
        final_assumption = html_assumption or xbrl_assumption or ""

    if ocf_source == "HTML" and html_period_inferred in {3, 6, 9, 12}:
        final_period = html_period_inferred

    used_html_for_period = False
    if (
        final_period in {3, 6, 9, 12}
        and html_period_inferred in {3, 6, 9, 12}
        and final_period == html_period_inferred
    ):
        used_html_for_period = True

    used_xbrl = False
    if cash_source == "XBRL" or ocf_source == "XBRL":
        used_xbrl = True
    if xbrl_period in {3, 6, 9, 12} and final_period == xbrl_period:
        used_xbrl = True

    used_html = False
    if cash_source == "HTML" or ocf_source == "HTML" or used_html_for_period:
        used_html = True

    unit_candidates: list[int] = []
    if cash_source == "HTML" or ocf_source == "HTML":
        unit_candidates.append(html_units_scale)
    if cash_source == "XBRL" or ocf_source == "XBRL":
        unit_candidates.append(xbrl_units_scale)

    final_units_scale = 1
    for value in unit_candidates:
        if value and value != 1:
            final_units_scale = value
            break
    if final_units_scale == 1 and unit_candidates:
        final_units_scale = unit_candidates[0] or 1

    source_tags: list[str] = []
    if used_xbrl:
        source_tags.append("XBRL")
    if used_html:
        source_tags.append("HTML")

    merged_sources = used_xbrl and used_html
    note_parts: list[str] = []
    if merged_sources:
        note_parts.append(f"values parsed from XBRL and HTML (merged): {canonical_url}")
    elif used_xbrl:
        note_parts.append(f"values parsed from XBRL: {canonical_url}")
    else:
        note_parts.append(f"values parsed from filing HTML: {canonical_url}")

    html_evidence = html_info.get("evidence") if isinstance(html_info, dict) else ""
    if used_html and html_evidence:
        note_parts.append(str(html_evidence))

    if html_source == "exhibit" and exhibit_href:
        if exhibit_doc_type:
            note_parts.append(f"exhibit source: {exhibit_doc_type} {exhibit_href}")
        else:
            note_parts.append(f"exhibit source: {exhibit_href}")

    if xbrl_result:
        for suffix in _extract_note_suffix(xbrl_result.get("note")):
            cleaned_suffix = suffix.replace(canonical_url, "").strip()
            if cleaned_suffix:
                note_parts.append(cleaned_suffix)
    elif xbrl_error:
        note_parts.append(xbrl_error)

    note_text = "; ".join(part for part in note_parts if part)

    result = finalize_runway_result(
        cash=final_cash,
        ocf_raw=final_ocf_raw,
        ocf_quarterly=final_ocf_quarterly,
        period_months=final_period,
        assumption=final_assumption,
        note=note_text,
        form_type=form_type,
        units_scale=final_units_scale,
        status=exhibit_status,
        source_tags=source_tags or None,
    )

    if html_source == "exhibit" and exhibit_href:
        result["exhibit_href"] = exhibit_href
        if exhibit_doc_type:
            result["exhibit_doc_type"] = exhibit_doc_type

    log_runway_outcome(
        canonical_url,
        form_type,
        result,
        html_info=html_info,
        xbrl_result=xbrl_result,
        xbrl_error=xbrl_error,
        extra={"path": "html_merge"},
    )

    log_parse_event(
        logging.DEBUG,
        f"runway: DONE status={result.get('status')} ocf_raw={result.get('ocf_raw')} cash_raw={result.get('cash_raw')} period_months={result.get('period_months')} runway_q={result.get('runway_quarters_raw')}",
        url=canonical_url,
    )

    return result


def parse(url: str, html: str | None = None, form_hint: str | None = None) -> dict:
    canonical_url = url
    parsed_form_hint = _normalize_form_type(form_hint)

    form_hint_query = _extract_form_hint_from_url(canonical_url)
    form_type_hint = _infer_form_type_from_url(canonical_url)

    xbrl_result: Optional[dict] = None
    xbrl_error: Optional[str] = None
    detected_form_type: Optional[str] = None

    is_local_file = canonical_url.startswith("file://")

    if not is_local_file:
        try:
            xbrl_result, detected_form_type = derive_from_xbrl(
                canonical_url, form_type_hint
            )
        except Exception as exc:
            xbrl_error = f"XBRL parse failed ({exc.__class__.__name__}: {exc})"
            log_parse_event(
                logging.WARNING,
                "runway parse XBRL failure",
                url=canonical_url,
                error=xbrl_error,
            )

    if xbrl_result and xbrl_error:
        xbrl_result = dict(xbrl_result)
        xbrl_result["note"] = f"{xbrl_result['note']} (with warning: {xbrl_error})"

    if xbrl_result and xbrl_result.get("complete"):
        result_copy = dict(xbrl_result)
        if not result_copy.get("source_tags"):
            result_copy["source_tags"] = ["XBRL"]
        log_runway_outcome(
            canonical_url,
            detected_form_type or form_type_hint,
            result_copy,
            xbrl_result=xbrl_result,
            xbrl_error=xbrl_error,
            extra={"path": "xbrl_complete"},
        )
        return result_copy

    html_text = html
    if html_text is None:
        try:
            raw_bytes = _fetch_url(canonical_url)
        except HTTPError as exc:
            log_parse_event(
                logging.ERROR,
                "runway fetch HTTP error",
                url=canonical_url,
                status_code=exc.code,
            )
            raise RuntimeError(
                f"HTTP error fetching filing ({exc.code}): {canonical_url}"
            ) from exc
        except URLError as exc:
            log_parse_event(
                logging.ERROR,
                "runway fetch URL error",
                url=canonical_url,
                reason=getattr(exc, "reason", exc),
            )
            raise RuntimeError(
                f"URL error fetching filing ({exc.reason}): {canonical_url}"
            ) from exc
        except Exception as exc:
            log_parse_event(
                logging.ERROR,
                "runway fetch unexpected error",
                url=canonical_url,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise RuntimeError(
                f"Unexpected error fetching filing: {canonical_url} ({exc.__class__.__name__}: {exc})"
            ) from exc

        html_text = raw_bytes.decode("utf-8", errors="ignore")

    html_text = unescape_html_entities(html_text, context=canonical_url)
    text = strip_html(html_text)

    text_form = _infer_form_type_from_text(text)

    form_type_candidates = [
        parsed_form_hint,
        form_hint_query,
        form_type_hint,
        text_form,
        detected_form_type,
        xbrl_result.get("form_type") if xbrl_result else None,
    ]
    form_type: Optional[str] = None
    for candidate in form_type_candidates:
        normalized_candidate = _normalize_form_type(candidate)
        if normalized_candidate:
            form_type = normalized_candidate
            break

    log_parse_event(
        logging.DEBUG,
        "form inference",
        url=canonical_url,
        form_hint=form_hint_query,
        url_form=form_type_hint,
        text_form=text_form,
        final_form=form_type,
    )

    defaults = _form_defaults(form_type)
    cashflow_headers = defaults["cashflow_headers"]
    cashflow_header_patterns = defaults.get("cashflow_header_patterns", [])

    exhibit_override: Optional[dict] = None
    if form_type == "6-K":
        lower_html = html_text.lower()
        header_present = any(header.lower() in lower_html for header in cashflow_headers)
        if not header_present:
            for pattern in cashflow_header_patterns:
                try:
                    if pattern.search(html_text):
                        header_present = True
                        break
                except Exception:
                    continue
        if not header_present:
            exhibit_parse = follow_exhibits_and_parse(canonical_url, html_text)
            if exhibit_parse.get("source") == "exhibit":
                exhibit_override = exhibit_parse
            else:
                note_parts = [
                    f"6-K missing operating cash flow statement headers: {canonical_url}"
                ]
                if xbrl_result:
                    for suffix in _extract_note_suffix(xbrl_result.get("note")):
                        cleaned_suffix = suffix.replace(canonical_url, "").strip()
                        if cleaned_suffix:
                            note_parts.append(cleaned_suffix)
                elif xbrl_error:
                    note_parts.append(xbrl_error)

                period_value = (
                    xbrl_result.get("period_months")
                    if (xbrl_result and xbrl_result.get("period_months") in {3, 6, 9, 12})
                    else defaults.get("period_months_default")
                )
                assumption_value = (xbrl_result.get("assumption") if xbrl_result else "") or ""
                units_scale_value = (xbrl_result.get("units_scale") if xbrl_result else None) or 1
                source_tags = ["XBRL"] if (xbrl_result and xbrl_result.get("cash_raw") is not None) else None

                result = finalize_runway_result(
                    cash=xbrl_result.get("cash_raw") if xbrl_result else None,
                    ocf_raw=None,
                    ocf_quarterly=None,
                    period_months=period_value,
                    assumption=assumption_value,
                    note="; ".join(part for part in note_parts if part),
                    form_type=form_type,
                    units_scale=units_scale_value,
                    status="6-K missing OCF (exhibits tried)",
                    source_tags=source_tags,
                )

                log_runway_outcome(
                    canonical_url,
                    form_type,
                    result,
                    xbrl_result=xbrl_result,
                    xbrl_error=xbrl_error,
                    extra={
                        "path": "6-K header+exhibit check",
                        "header_present": header_present,
                    },
                )

                return result

    html_info = parse_html_cashflow_sections(html_text, context_url=canonical_url)
    html_cash = html_info.get("cash_value")
    html_ocf_raw_value = html_info.get("ocf_value")
    html_period_inferred = html_info.get("period_months_inferred")
    html_units_scale = html_info.get("units_scale") or 1
    html_assumption = ""
    html_source = "main"
    exhibit_status: Optional[str] = None
    exhibit_href: Optional[str] = None
    exhibit_doc_type: Optional[str] = None

    if html_cash is None or html_ocf_raw_value is None:
        exhibit_parse = exhibit_override or follow_exhibits_and_parse(canonical_url, html_text)
        if exhibit_parse.get("source") == "exhibit":
            html_source = "exhibit"
            html_cash = exhibit_parse.get("cash_value")
            html_ocf_raw_value = exhibit_parse.get("ocf_value")
            html_period_inferred = exhibit_parse.get("period_months")
            html_units_scale = exhibit_parse.get("units_scale") or 1
            html_assumption = ""
            exhibit_status = exhibit_parse.get("status")
            exhibit_href = exhibit_parse.get("exhibit_href")
            exhibit_doc_type = exhibit_parse.get("exhibit_doc_type")
            source_html_info = exhibit_parse.get("html_info")
            if isinstance(source_html_info, dict):
                html_info = dict(source_html_info)
            else:
                html_info = {
                    "found_cashflow_header": exhibit_parse.get("found_cashflow_header"),
                    "cash_value": html_cash,
                    "ocf_value": html_ocf_raw_value,
                    "period_months_inferred": html_period_inferred,
                    "units_scale": html_units_scale,
                    "evidence": exhibit_parse.get("evidence") or "",
                    "html_header": exhibit_parse.get("found_cashflow_header"),
                    "source": "exhibit",
                }
            html_info["source"] = "exhibit"
            if exhibit_href:
                html_info["exhibit_href"] = exhibit_href

    defaults = _form_defaults(form_type)

    result = _merge_results(
        canonical_url=canonical_url,
        form_type=form_type,
        html_info=html_info,
        html_cash=html_cash,
        html_ocf_raw_value=html_ocf_raw_value,
        html_period_inferred=html_period_inferred,
        html_units_scale=html_units_scale,
        html_assumption=html_assumption,
        html_source=html_source,
        exhibit_status=exhibit_status,
        exhibit_href=exhibit_href,
        exhibit_doc_type=exhibit_doc_type,
        default_period=defaults.get("period_months_default"),
        xbrl_result=xbrl_result,
        xbrl_error=xbrl_error,
    )

    if html_source == "exhibit" and exhibit_href:
        result["exhibit_href"] = exhibit_href
        if exhibit_doc_type:
            result["exhibit_doc_type"] = exhibit_doc_type

    return result


__all__ = ["parse"]
