"""Parser for extracting cash runway metrics from 10-Q/10-K filings."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse


_USER_AGENT: str | None = None

_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)


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

        remainder = text[match.end():match.end() + 500]
        tokens = remainder.split()
        for token in tokens[:6]:
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


def _select_best_fact(facts: Iterable[dict], accession: str) -> Optional[float]:
    best_value: Optional[float] = None
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
    return best_value


def _extract_fact_value(units: dict, accession: str) -> Optional[float]:
    preferred_units = ["USD", "USD ", "USD millions", "USDm", "USD $", "USD $ millions"]
    for unit_name in preferred_units:
        facts = units.get(unit_name)
        if not facts:
            continue
        value = _select_best_fact(facts, accession)
        if value is not None:
            return value

    # fallback to any available facts, preferring matching accession / newest data
    all_facts: list[dict] = []
    for facts in units.values():
        all_facts.extend(facts)
    if not all_facts:
        return None
    return _select_best_fact(all_facts, accession)


def _fetch_xbrl_value(cik: str, accession: str, concept: str) -> Optional[float]:
    base_url = "https://data.sec.gov/api/xbrl/companyconcept"
    url = f"{base_url}/CIK{cik}/us-gaap/{concept}.json"
    data = _fetch_json(url)
    units = data.get("units") or {}
    value = _extract_fact_value(units, accession)
    return value


def _derive_from_xbrl(filing_url: str) -> Optional[dict]:
    cik, accession = _extract_filing_identifiers(filing_url)
    if not cik or not accession:
        return None

    cash = None
    cash_concepts = [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsIncludingRestrictedCashCurrent",
    ]
    errors: list[str] = []

    for concept in cash_concepts:
        try:
            cash = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if cash is not None:
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
            burn = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if burn is not None:
            break

    if cash is None and burn is None:
        if errors:
            raise RuntimeError("; ".join(errors))
        return None

    quarterly_burn: Optional[float]
    if burn is None:
        quarterly_burn = None
    else:
        quarterly_burn = abs(float(burn))
        if quarterly_burn == 0:
            quarterly_burn = 0.0

    runway_quarters: Optional[float]
    if cash is not None and quarterly_burn not in (None, 0.0):
        runway_quarters = float(cash) / float(quarterly_burn)
    else:
        runway_quarters = None

    result = {
        "cash": float(cash) if cash is not None else None,
        "quarterly_burn": quarterly_burn,
        "runway_quarters": runway_quarters,
        "note": f"values parsed from XBRL: {filing_url}",
    }
    if cash is None or burn is None:
        result["note"] += " (partial XBRL data)"
    return result


def get_runway_from_filing(filing_url: str) -> dict:
    """Fetch a filing URL and estimate the cash runway metrics."""

    xbrl_result: Optional[dict] = None
    xbrl_error: Optional[str] = None
    try:
        xbrl_result = _derive_from_xbrl(filing_url)
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

    cash_keywords = [
        "Cash and cash equivalents",
        "Cash and cash equivalents, end of period",
        "Cash and cash equivalents at end of period",
    ]
    cash_value = _extract_number_after_keyword(text, cash_keywords)

    burn_value = _extract_number_after_keyword(
        text,
        [
            "Net cash used in operating activities",
        ],
    )

    if burn_value is None:
        provided_value = _extract_number_after_keyword(
            text,
            [
                "Net cash provided by operating activities",
            ],
        )
        if provided_value is not None:
            quarterly_burn = 0.0
        else:
            quarterly_burn = None
    else:
        quarterly_burn = abs(burn_value)

    cash = float(cash_value) if cash_value is not None else None

    if quarterly_burn is not None and quarterly_burn <= 0:
        quarterly_burn = None

    runway_quarters: Optional[float]
    if cash is not None and quarterly_burn is not None and quarterly_burn > 0:
        runway_quarters = cash / quarterly_burn
    else:
        runway_quarters = None

    note_parts = [f"values parsed from 10-Q/10-K HTML: {filing_url}"]
    if xbrl_error:
        note_parts.append(xbrl_error)

    return {
        "cash": cash,
        "quarterly_burn": quarterly_burn,
        "runway_quarters": runway_quarters,
        "note": "; ".join(note_parts),
    }


__all__ = ["get_runway_from_filing"]
