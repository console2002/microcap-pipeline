"""Parser for extracting cash runway metrics from 10-Q/10-K filings."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse


_USER_AGENT: str | None = None

_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)
_FORM_TYPE_PATTERN = re.compile(r"10-[A-Za-z0-9]+", re.IGNORECASE)


_LOGGER = logging.getLogger(__name__)


def _log_debug(message: str) -> None:
    """Log debug messages without requiring callers to handle logging setup."""
    _LOGGER.debug(message)


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


_NUMERIC_TOKEN_PATTERN = re.compile(r"[\d\$\(\)\-]")
_NUMBER_SEARCH_PATTERN = re.compile(r"[\$\(]*[-+]?\d[\d,]*(?:\.\d+)?\)?")
_FACT_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
]


def _to_float(num_str: str) -> float:
    """Convert a numeric string that may contain formatting into a float."""
    text = num_str.strip()
    match = _NUMBER_SEARCH_PATTERN.search(text)
    if not match:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    value = match.group(0)
    negative = False

    if "(" in value and ")" in value:
        negative = True

    cleaned = value.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    cleaned = cleaned.replace("−", "-").strip()

    if cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:]
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:]

    if not cleaned:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    number = float(cleaned)
    if negative:
        number = -number
    return number


def _extract_number_after_keyword(text: str, keywords: Iterable[str]) -> Optional[float]:
    """Find the first numeric value that appears after any keyword."""
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue

        remainder = text[match.end():match.end() + 1000]
        tokens = remainder.split()
        for token in tokens[:12]:
            if not _NUMERIC_TOKEN_PATTERN.search(token):
                continue
            try:
                return _to_float(token)
            except ValueError:
                continue
    return None


def _strip_html(html_text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    without_tags = re.sub(r"<[^>]+>", " ", html_text)
    normalized = " ".join(without_tags.split())
    return normalized


def _normalize_digits(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits


def _extract_filing_identifiers(filing_url: str) -> tuple[Optional[str], Optional[str]]:
    parsed = urlparse(filing_url)
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    doc_path = path
    if "doc" in query and query["doc"]:
        doc_path = query["doc"][0]
    doc_path = unquote(doc_path)

    match = _ACCESSION_RE.search(doc_path)
    if not match:
        match = _ACCESSION_RE.search(path)
    if not match:
        return None, None

    cik_digits = _normalize_digits(match.group(1))
    accession_digits = _normalize_digits(match.group(2))
    if not cik_digits or not accession_digits:
        return None, None

    cik_padded = cik_digits.zfill(10)
    if len(accession_digits) < 18:
        accession_digits = accession_digits.zfill(18)
    accession = f"{accession_digits[:10]}-{accession_digits[10:12]}-{accession_digits[12:]}"
    return cik_padded, accession


def _normalize_form_type(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    match = _FORM_TYPE_PATTERN.search(text)
    if not match:
        return None
    normalized = match.group(0).upper()
    if normalized.endswith("/A"):
        normalized = normalized[:-2]
    if normalized.startswith("10-K"):
        return "10-K"
    if normalized.startswith("10-Q"):
        return "10-Q"
    return normalized


def _infer_form_type_from_url(filing_url: str) -> Optional[str]:
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


def _parse_fact_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        pass
    else:
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    for fmt in _FACT_DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _select_best_fact(
    facts: Iterable[dict], accession: str
) -> tuple[Optional[float], Optional[str]]:
    best_value: Optional[float] = None
    best_form: Optional[str] = None
    best_key: tuple[int, datetime, int] | None = None
    for fact in facts:
        val = fact.get("val")
        if not isinstance(val, (int, float)):
            continue
        fact_accn = (fact.get("accn") or "").strip()
        match_score = 1 if accession and fact_accn == accession else 0
        end_dt = _parse_fact_timestamp(
            fact.get("end") or fact.get("filed") or fact.get("instant")
        )
        if end_dt is None:
            end_dt = datetime.min.replace(tzinfo=timezone.utc)
        fy_value = fact.get("fy")
        try:
            fy_score = int(fy_value)
        except (TypeError, ValueError):
            fy_score = 0
        key = (match_score, end_dt, fy_score)
        if best_key is None or key > best_key:
            best_key = key
            best_value = float(val)
            best_form = _normalize_form_type(fact.get("form"))
    return best_value, best_form


def _extract_fact_value(
    units: dict, accession: str
) -> tuple[Optional[float], Optional[str]]:
    preferred_units = ["USD", "USD ", "USD millions", "USDm", "USD $", "USD $ millions"]
    for unit_name in preferred_units:
        facts = units.get(unit_name)
        if not facts:
            continue
        value, form_type = _select_best_fact(facts, accession)
        if value is not None:
            return value, form_type

    # fallback to any available facts, preferring matching accession / newest data
    all_facts: list[dict] = []
    for facts in units.values():
        all_facts.extend(facts)
    if not all_facts:
        return None, None
    return _select_best_fact(all_facts, accession)


def _fetch_xbrl_value(
    cik: str, accession: str, concept: str
) -> tuple[Optional[float], Optional[str]]:
    base_url = "https://data.sec.gov/api/xbrl/companyconcept"
    url = f"{base_url}/CIK{cik}/us-gaap/{concept}.json"
    data = _fetch_json(url)
    units = data.get("units") or {}
    return _extract_fact_value(units, accession)


def _derive_from_xbrl(
    filing_url: str, form_type_hint: Optional[str] = None
) -> tuple[Optional[dict], Optional[str]]:
    cik, accession = _extract_filing_identifiers(filing_url)
    if not cik or not accession:
        return None, form_type_hint

    cash = None
    form_type = form_type_hint
    cash_concepts = [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsIncludingRestrictedCashCurrent",
    ]
    errors: list[str] = []

    for concept in cash_concepts:
        try:
            cash_value, value_form = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if cash_value is not None:
            cash = cash_value
            if form_type is None and value_form:
                form_type = value_form
            break

    burn = None
    burn_concepts = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivitiesExcludingDiscontinuedOperations",
        "NetCashProvidedByUsedInOperatingActivitiesConverted",
    ]
    for concept in burn_concepts:
        try:
            burn_value, value_form = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if burn_value is not None:
            burn = burn_value
            if form_type is None and value_form:
                form_type = value_form
            break

    if cash is None and burn is None:
        if errors:
            raise RuntimeError("; ".join(errors))
        return None, form_type

    quarterly_burn: Optional[float]
    if burn is None:
        quarterly_burn = None
    else:
        quarterly_burn = abs(float(burn))
        if quarterly_burn == 0:
            quarterly_burn = 0.0

    if form_type and form_type.upper().startswith("10-K") and quarterly_burn is not None:
        quarterly_burn = quarterly_burn / 4.0

    runway_quarters: Optional[float]
    if cash is not None and quarterly_burn not in (None, 0.0):
        runway_quarters = float(cash) / float(quarterly_burn)
    else:
        runway_quarters = None

    cash_value = float(cash) if cash is not None else None
    result = {
        "cash": round(cash_value, 2) if cash_value is not None else None,
        "quarterly_burn": round(quarterly_burn, 2) if quarterly_burn is not None else None,
        "runway_quarters": round(runway_quarters, 2) if runway_quarters is not None else None,
        "note": f"values parsed from XBRL: {filing_url}",
    }
    if cash is None or burn is None:
        result["note"] += " (partial XBRL data)"
    return result, form_type


def get_runway_from_filing(filing_url: str) -> dict:
    """Fetch a filing URL and estimate the cash runway metrics."""

    form_type = _infer_form_type_from_url(filing_url)

    xbrl_result: Optional[dict] = None
    xbrl_error: Optional[str] = None
    detected_form_type: Optional[str] = None
    try:
        xbrl_result, detected_form_type = _derive_from_xbrl(
            filing_url, form_type_hint=form_type
        )
    except Exception as exc:
        xbrl_error = f"XBRL parse failed ({exc.__class__.__name__}: {exc})"

    if xbrl_result:
        if xbrl_error:
            xbrl_result["note"] = f"{xbrl_result['note']} (with warning: {xbrl_error})"
        return xbrl_result

    try:
        raw_bytes = _fetch_url(filing_url)
    except HTTPError as exc:
        raise RuntimeError(
            f"HTTP error fetching filing ({exc.code}): {filing_url}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"URL error fetching filing ({exc.reason}): {filing_url}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error fetching filing: {filing_url} ({exc.__class__.__name__}: {exc})"
        ) from exc
    html_text = raw_bytes.decode("utf-8", errors="ignore")
    text = _strip_html(html_text)

    form_type = detected_form_type or form_type
    if form_type is None:
        form_type = _infer_form_type_from_text(text)

    cash_keywords = [
        "Cash and cash equivalents, end of period",
        "Cash and cash equivalents at end of period",
        "Cash, cash equivalents and restricted cash, end of period",
        "Cash and cash equivalents, including restricted cash, end of period",
        "Cash and cash equivalents",
    ]
    cash_value = _extract_number_after_keyword(text, cash_keywords)

    provided_value: Optional[float] = None
    burn_value = _extract_number_after_keyword(
        text,
        [
            "Net cash used in operating activities",
            "Net cash (used in) provided by operating activities",
            "Net cash flows from operating activities",
            "Net cash used in operating activities — continuing operations",
            "Net cash used in operating activities - continuing operations",
        ],
    )

    if burn_value is None:
        provided_value = _extract_number_after_keyword(
            text,
            [
                "Net cash provided by operating activities",
                "Net cash provided by operating activities — continuing operations",
                "Net cash provided by operating activities - continuing operations",
            ],
        )
        if provided_value is not None:
            quarterly_burn = 0.0
        else:
            quarterly_burn = None
    else:
        if burn_value > 0:
            provided_value = burn_value
            quarterly_burn = 0.0
        else:
            quarterly_burn = abs(burn_value)

    _log_debug(f"runway_html: cash_keywords matched -> {cash_value}")
    _log_debug(
        "runway_html: burn_keywords matched -> "
        f"{burn_value if burn_value is not None else provided_value}"
    )

    cash = float(cash_value) if cash_value is not None else None

    if quarterly_burn is not None and quarterly_burn <= 0:
        quarterly_burn = None

    if form_type and form_type.upper().startswith("10-K") and quarterly_burn is not None:
        quarterly_burn = quarterly_burn / 4.0

    runway_quarters: Optional[float]
    if cash is not None and quarterly_burn is not None and quarterly_burn > 0:
        runway_quarters = cash / quarterly_burn
    else:
        runway_quarters = None

    note_parts = [f"values parsed from 10-Q/10-K HTML: {filing_url}"]
    if xbrl_error:
        note_parts.append(xbrl_error)

    return {
        "cash": round(cash, 2) if cash is not None else None,
        "quarterly_burn": round(quarterly_burn, 2) if quarterly_burn is not None else None,
        "runway_quarters": round(runway_quarters, 2)
        if runway_quarters is not None
        else None,
        "note": "; ".join(note_parts),
    }


__all__ = ["get_runway_from_filing"]
