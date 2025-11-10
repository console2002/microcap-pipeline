"""Routing entry point for runway parsing."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

from importlib import import_module

from .htmlutil import strip_html, unescape_html_entities
from .logging import log_parse_event, log_runway_outcome
from .postproc import finalize_runway_result

_USER_AGENT: Optional[str] = None

_FORM_TYPE_PATTERN = re.compile(
    r"""
    (
        10[-\s]?QT?|
        10[-\s]?KT?|
        20[-\s]?F|
        40[-\s]?F|
        6[-\s]?K|
        8[-\s]?K|
        S[-\s]?3|
        S[-\s]?4|
        S[-\s]?8|
        8[-\s]?A12B|
        424B[1-8]|
        DEF\s*14A|
        13D|
        13G|
        13F[-\s]?HR
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FORM_SELECTION_PRIORITY: dict[str, int] = {
    "6-K": 0,
    "10-Q": 1,
    "10-K": 2,
    "20-F": 3,
    "40-F": 4,
    "8-K": 5,
    "S-3": 6,
    "S-4": 7,
    "S-8": 8,
    "8-A12B": 9,
    "424B1": 10,
    "424B2": 10,
    "424B3": 10,
    "424B4": 10,
    "424B5": 10,
    "424B7": 10,
    "424B8": 10,
    "DEF 14A": 11,
    "13D": 12,
    "13G": 13,
    "13F-HR": 14,
}

_FORM_ADAPTERS = {
    "10-Q": "US_Quarterly",
    "10-K": "US_Annual",
    "20-F": "Foreign_Annual",
    "40-F": "Foreign_Annual",
    "6-K": "SixK",
}

_BALANCE_HEADERS = [
    "CONSOLIDATED BALANCE SHEETS",
    "CONDENSED CONSOLIDATED BALANCE SHEETS",
    "BALANCE SHEETS",
]

_CASHFLOW_HEADERS_BASE = [
    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONSOLIDATED STATEMENT OF CASH FLOWS",
    "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS",
    "STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS",
]

_CASHFLOW_HEADER_PATTERNS = [
    re.compile(
        r"CONSOLIDATED\s*(?:CONDENSED\s*)?(?:INTERIM\s*)?STATEMENTS?\s*OF\s*CASH\s*FLOWS",
        re.IGNORECASE,
    ),
]

_OCF_KEYWORDS_BURN_BASE = [
    "Net cash used in operating activities - continuing operations",
    "Net cash used in operating activities — continuing operations",
    "Net cash used in operating activities from continuing operations",
    "Net cash used for operating activities - continuing operations",
    "Net cash (used in) operating activities - continuing operations",
    "Net cash (used in) operating activities — continuing operations",
    "Net cash used in operating activities, continuing operations",
    "Net cash (used in) operating activities, continuing operations",
    "Net cash used for operating activities, continuing operations",
    "Net cash used in operating activities",
    "Net cash used for operating activities",
    "Net cash (used in) operating activities",
    "Net cash flows used in operating activities",
]

_OCF_KEYWORDS_PROVIDED_BASE = [
    "Net cash provided by operating activities - continuing operations",
    "Net cash provided by operating activities — continuing operations",
    "Net cash provided by operating activities from continuing operations",
    "Net cash provided by (used in) operating activities - continuing operations",
    "Net cash provided by (used in) operating activities — continuing operations",
    "Net cash from operating activities - continuing operations",
    "Net cash from operating activities — continuing operations",
    "Net cash provided by operating activities, continuing operations",
    "Net cash provided by (used in) operating activities, continuing operations",
    "Net cash flows from operating activities, continuing operations",
    "Net cash from operating activities, continuing operations",
    "Net cash provided by operating activities",
    "Net cash provided by (used in) operating activities",
    "Net cash flows from operating activities",
    "Net cash from operating activities",
]

_OCF_KEYWORDS_BURN_EXTRA = [
    "Net cash used in operating activities — continuing operations",
    "Net cash (used in) operating activities — continuing operations",
    "Net cash used for operating activities — continuing operations",
    "Net cash used in operating activities",
    "Net cash used for operating activities",
    "Net cash (used in) operating activities",
    "Net cash flows used in operating activities",
    "Net cash flows (used in) operating activities",
    "Net cash used in operations",
    "Cash used in operations",
    "Cash used in operating activities",
    "Cash flows used in operating activities",
    "Cash flows (used in) operating activities",
    "Cash flows provided by (used in) operating activities",
    "Net cash outflow from operating activities",
    "Net cash (outflow) from operating activities",
    "Net cash used by operating activities",
]

_OCF_KEYWORDS_PROVIDED_EXTRA = [
    "Net cash provided by operating activities — continuing operations",
    "Net cash provided by (used in) operating activities — continuing operations",
    "Net cash provided from operating activities — continuing operations",
    "Net cash from operating activities — continuing operations",
    "Net cash flows from operating activities — continuing operations",
    "Net cash provided by operating activities",
    "Net cash provided from operating activities",
    "Net cash provided by (used in) operating activities",
    "Net cash from operating activities",
    "Net cash flows from operating activities",
    "Net cash flow from operating activities",
    "Net cash generated from operating activities",
    "Net cash generated by operating activities",
    "Net cash flows generated by operating activities",
    "Net cash flows generated from operating activities",
    "Cash generated from operations",
    "Cash generated by operations",
    "Net cash inflow from operating activities",
    "Cash flows provided by operating activities",
    "Cash flows provided by (used in) operating activities",
    "Cash flows from operating activities",
]

_PARSER_MODULES: dict[str, str] = {
    "10-Q": "parse.q10_k10",
    "10-K": "parse.q10_k10",
    "20-F": "parse.f20_f40",
    "40-F": "parse.f20_f40",
    "6-K": "parse.k6",
    "8-K": "parse.k8",
    "S-3": "parse.s3",
    "424B1": "parse.f424b",
    "424B2": "parse.f424b",
    "424B3": "parse.f424b",
    "424B4": "parse.f424b",
    "424B5": "parse.f424b",
    "424B7": "parse.f424b",
    "424B8": "parse.f424b",
    "S-8": "parse.s8",
    "S-4": "parse.s4",
    "8-A12B": "parse.f8a12b",
    "DEF 14A": "parse.def14a",
    "13D": "parse.f13d_g",
    "13G": "parse.f13d_g",
    "13F-HR": "parse.f13f_hr",
}


def _normalize_form_type(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    matches = list(_FORM_TYPE_PATTERN.finditer(text))
    if not matches:
        return None

    mapping = {
        "10Q": "10-Q",
        "10K": "10-K",
        "20F": "20-F",
        "40F": "40-F",
        "6K": "6-K",
        "8K": "8-K",
        "S3": "S-3",
        "S4": "S-4",
        "S8": "S-8",
        "8A12B": "8-A12B",
        "DEF14A": "DEF 14A",
        "13D": "13D",
        "13G": "13G",
        "13FHR": "13F-HR",
    }

    candidates: list[tuple[int, int, str]] = []

    for match in matches:
        normalized_raw = match.group(1).upper().replace(" ", "")
        normalized = normalized_raw.replace("QT", "Q").replace("KT", "K").replace("-", "")
        if normalized.endswith("/A"):
            normalized = normalized[:-2]
        elif normalized.endswith("A") and normalized not in {"DEF14A"}:
            normalized = normalized[:-1]
        if normalized.startswith("424B"):
            canonical = normalized
        else:
            canonical = mapping.get(normalized) or normalized

        priority = _FORM_SELECTION_PRIORITY.get(canonical, 100)
        candidates.append((match.start(), priority, canonical))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _extract_form_hint_from_url(filing_url: str) -> Optional[str]:
    try:
        parsed = urlparse(filing_url)
    except Exception:
        return None
    query = parse_qs(parsed.query)
    for key, values in query.items():
        if key.lower() != "form":
            continue
        for value in values:
            normalized = _normalize_form_type(value)
            if normalized:
                return normalized
    return None


def _infer_form_type_from_url(filing_url: str) -> Optional[str]:
    hint = _extract_form_hint_from_url(filing_url)
    if hint:
        return hint
    try:
        parsed = urlparse(filing_url)
    except Exception:
        return None
    components = [parsed.path or ""]
    query = parse_qs(parsed.query)
    for values in query.values():
        components.extend(values)
    combined = " ".join(components)
    return _normalize_form_type(combined)


def _infer_form_type_from_text(text: str) -> Optional[str]:
    return _normalize_form_type(text)


def _form_defaults(form_type: Optional[str]) -> dict:
    normalized = _normalize_form_type(form_type)
    adapter = _FORM_ADAPTERS.get(normalized) if normalized else None

    period_default: Optional[int] = None
    if adapter in {"US_Annual", "Foreign_Annual"}:
        period_default = 12
    elif adapter in {"US_Quarterly", "SixK"}:
        period_default = 3

    burn_keywords = list(_OCF_KEYWORDS_BURN_BASE)
    burn_keywords.extend(_OCF_KEYWORDS_BURN_EXTRA)

    provided_keywords = list(_OCF_KEYWORDS_PROVIDED_BASE)
    provided_keywords.extend(_OCF_KEYWORDS_PROVIDED_EXTRA)

    return {
        "period_months_default": period_default,
        "ocf_keywords_burn": burn_keywords,
        "ocf_keywords_provided": provided_keywords,
        "cashflow_headers": list(_CASHFLOW_HEADERS_BASE),
        "cashflow_header_patterns": list(_CASHFLOW_HEADER_PATTERNS),
    }


def _extract_note_suffix(note: Optional[str]) -> list[str]:
    if not note:
        return []
    text = str(note)
    prefix = "values parsed from XBRL"
    lowered = text.lower()
    prefix_lower = prefix.lower()
    if lowered.startswith(prefix_lower):
        colon_idx = text.find(":")
        if colon_idx != -1:
            remainder = text[colon_idx + 1 :].strip()
        else:
            remainder = text[len(prefix) :].strip()
        return [remainder] if remainder else []
    return [text]


def _canonicalize_sec_filing_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    if parsed.scheme == "file":
        query_params = parse_qs(parsed.query)
        doc_values = query_params.get("doc") or []
        viewer_path = Path(unquote(parsed.path or "")) if parsed.path else None
        viewer_dir = viewer_path.parent if (viewer_path and viewer_path.exists()) else None

        def _resolve_candidate(doc_value: str) -> Optional[Path]:
            cleaned = unquote(str(doc_value or "")).strip()
            if not cleaned:
                return None
            if cleaned.lower().startswith("file://"):
                try:
                    nested_parsed = urlparse(cleaned)
                    nested_path = Path(unquote(nested_parsed.path or ""))
                except Exception:
                    return None
                return nested_path.resolve() if nested_path.exists() else None

            candidate_path = Path(cleaned)
            if candidate_path.exists():
                return candidate_path.resolve()

            if viewer_dir is not None:
                if cleaned.startswith("/"):
                    candidate_from_root = (viewer_dir / cleaned.lstrip("/")).resolve()
                    if candidate_from_root.exists():
                        return candidate_from_root
                relative_candidate = (viewer_dir / cleaned).resolve()
                if relative_candidate.exists():
                    return relative_candidate
                last_segment = Path(cleaned).name
                if last_segment:
                    sibling_candidate = (viewer_dir / last_segment).resolve()
                    if sibling_candidate.exists():
                        return sibling_candidate
            return None

        for doc_value in doc_values:
            target_path = _resolve_candidate(doc_value)
            if target_path is None:
                continue

            remaining_params: list[tuple[str, str]] = []
            for key, values in query_params.items():
                if key.lower() == "doc":
                    continue
                for value in values:
                    if value:
                        remaining_params.append((key, value))

            query_string = urlencode(remaining_params) if remaining_params else ""
            canonical_base = target_path.as_uri()
            return f"{canonical_base}?{query_string}" if query_string else canonical_base

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if not host.endswith("sec.gov"):
        return url

    if not any(path.startswith(prefix) for prefix in ("/ix", "/ixviewer", "/cgi-bin/viewer")):
        return url

    query = parse_qs(parsed.query)
    doc_values = query.get("doc") or []
    if not doc_values:
        return url

    doc_value = unquote(str(doc_values[0] or "")).strip()
    if not doc_value:
        return url

    doc_parsed = urlparse(doc_value)
    fragmentless = doc_parsed._replace(fragment="")
    if fragmentless.scheme and fragmentless.netloc:
        canonical = fragmentless.geturl()
    else:
        doc_path = fragmentless.path or doc_value
        if not doc_path.startswith("/"):
            doc_path = f"/{doc_path.lstrip('/')}"
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or "www.sec.gov"
        canonical = urlunparse((scheme, netloc, doc_path, "", fragmentless.query, ""))

    return canonical


def _user_agent() -> str:
    global _USER_AGENT
    if _USER_AGENT is None:
        try:
            from app.config import load_config

            cfg = load_config()
            user_agent = str(cfg.get("UserAgent", "")).strip()
        except Exception:
            user_agent = ""
        if not user_agent:
            user_agent = "microcap-pipeline/1.0"
        _USER_AGENT = user_agent
    return _USER_AGENT


def _fetch_url(url: str) -> bytes:
    req = request.Request(url, headers={"User-Agent": _user_agent()})
    with request.urlopen(req) as response:
        return response.read()


def _fetch_json(url: str) -> dict:
    try:
        raw = _fetch_url(url)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error fetching JSON ({exc.code}): {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error fetching JSON ({exc.reason}): {url}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error fetching JSON: {url} ({exc.__class__.__name__}: {exc})"
        ) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


def url_matches_form(url: str, form: str | None) -> bool:
    normalized = _normalize_form_type(form)
    if not normalized:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    components = [parsed.path or ""]
    query = parse_qs(parsed.query)
    for values in query.values():
        components.extend(values)
    combined = " ".join(components).upper().replace("-", "")
    target = normalized.upper().replace("-", "")
    if target in combined:
        return True

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if host.endswith("sec.gov"):
        if any(path.startswith(prefix) for prefix in ("/ix", "/ixviewer", "/cgi-bin/viewer")):
            return True

    return False


def _resolve_parser(form_type: Optional[str]) -> Optional[Callable[[str, Optional[str], Optional[str]], dict]]:
    normalized = _normalize_form_type(form_type)
    module_name = _PARSER_MODULES.get(normalized or "")
    if not module_name:
        return None
    module = import_module(module_name)
    return getattr(module, "parse", None)


def _not_implemented_result(url: str, form_type: Optional[str]) -> dict:
    label = form_type or "Unknown"
    note = f"Parsing for {label} filings is not implemented: {url}"
    result = finalize_runway_result(
        cash=None,
        ocf_raw=None,
        ocf_quarterly=None,
        period_months=None,
        assumption=None,
        note=note,
        form_type=form_type,
        units_scale=1,
        status="NotImplemented",
        source_tags=None,
    )
    log_runway_outcome(
        url,
        form_type,
        result,
        extra={"path": "router_not_implemented"},
    )
    result["complete"] = False
    return result


def get_runway_from_filing(filing_url: str) -> dict:
    original_input = filing_url
    original_form_hint = _extract_form_hint_from_url(filing_url)
    canonical_url = _canonicalize_sec_filing_url(filing_url)

    html_text: Optional[str] = None
    is_local_file = False
    local_file_path: Optional[Path] = None

    parsed_initial = urlparse(canonical_url)
    if parsed_initial.scheme == "file":
        file_path = unquote(parsed_initial.path or "")
        if parsed_initial.netloc:
            file_path = f"//{parsed_initial.netloc}{file_path}"
        candidate = Path(file_path)
        if candidate.exists():
            resolved = candidate.resolve()
            is_local_file = True
            local_file_path = resolved
            base_uri = resolved.as_uri()
            if parsed_initial.query:
                canonical_url = f"{base_uri}?{parsed_initial.query}"
            else:
                canonical_url = base_uri
            parsed_initial = urlparse(canonical_url)
    elif not parsed_initial.scheme and not parsed_initial.netloc:
        path_only = parsed_initial.path or canonical_url
        query_only = parsed_initial.query
        candidate = Path(unquote(path_only))
        if candidate.exists():
            resolved = candidate.resolve()
            is_local_file = True
            local_file_path = resolved
            base_uri = resolved.as_uri()
            canonical_url = f"{base_uri}?{query_only}" if query_only else base_uri
            parsed_initial = urlparse(canonical_url)

    if canonical_url != original_input:
        log_parse_event(
            logging.DEBUG,
            "canonical_url",
            original=original_input,
            canonical=canonical_url,
        )

    if is_local_file and local_file_path is not None:
        log_parse_event(
            logging.DEBUG,
            "offline html detected",
            offline_html=str(local_file_path),
        )
        try:
            html_text = local_file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read local filing HTML: {local_file_path} ({exc})"
            ) from exc

    form_hint_query = _extract_form_hint_from_url(canonical_url)
    combined_form_hint = form_hint_query or original_form_hint
    form_type_hint = _infer_form_type_from_url(canonical_url)
    if not form_type_hint and original_form_hint:
        form_type_hint = original_form_hint

    parser = _resolve_parser(form_type_hint or combined_form_hint)
    if parser is None:
        text_form_hint: Optional[str] = None
        fetch_error: Optional[str] = None

        if html_text is None:
            try:
                raw_bytes = _fetch_url(canonical_url)
            except Exception as exc:  # pragma: no cover - network errors
                fetch_error = f"{exc.__class__.__name__}: {exc}"
                log_parse_event(
                    logging.DEBUG,
                    "form inference fetch failed",
                    url=canonical_url,
                    error=fetch_error,
                )
            else:
                html_text = raw_bytes.decode("utf-8", errors="ignore")

        if html_text:
            unescaped = unescape_html_entities(html_text, context=canonical_url)
            text_form_hint = _infer_form_type_from_text(strip_html(unescaped))

        parser = _resolve_parser(text_form_hint or combined_form_hint)
        if parser is not None:
            log_parse_event(
                logging.DEBUG,
                "form inferred from text",
                url=canonical_url,
                form_hint=form_hint_query,
                url_form=form_type_hint,
                text_form=text_form_hint,
            )
        else:
            log_parse_event(
                logging.DEBUG,
                "parser not implemented",
                url=canonical_url,
                form_hint=form_hint_query,
                url_form=form_type_hint,
                text_form=text_form_hint,
                fetch_error=fetch_error,
            )
            fallback_form = text_form_hint or form_type_hint or form_hint_query
            return _not_implemented_result(canonical_url, fallback_form)

    return parser(canonical_url, html_text, combined_form_hint)


__all__ = ["get_runway_from_filing", "url_matches_form", "_fetch_url", "_fetch_json", "_form_defaults", "_normalize_form_type", "_extract_form_hint_from_url", "_infer_form_type_from_url", "_infer_form_type_from_text", "_extract_note_suffix"]
