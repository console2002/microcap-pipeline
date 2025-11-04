"""Parser for extracting cash runway metrics from 10-Q/10-K filings."""
from __future__ import annotations

import json
import logging
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse


_USER_AGENT: str | None = None

_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)
_FORM_TYPE_PATTERN = re.compile(
    r"(10-QT?|10-KT?|20-F|40-F|6-K)(?:/A)?",
    re.IGNORECASE,
)

_LOGGER = logging.getLogger(__name__)


def _log_debug(message: str) -> None:
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
_SCALE_PATTERN = re.compile(r"\([^)]*?\bin\s+(thousands|millions)\b[^)]*\)", re.IGNORECASE)

FORM_ADAPTERS = {
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
    "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS",
]

_CASH_KEYWORDS_BALANCE = [
    "Cash and cash equivalents",
    "Cash and cash equivalents, current",
    "Cash",
    "Cash, cash equivalents and restricted cash",
    "Cash and cash equivalents including restricted cash",
]

_CASH_KEYWORDS_FLOW = [
    "Cash and cash equivalents, end of period",
    "Cash and cash equivalents at end of period",
    "Cash, cash equivalents and restricted cash, end of period",
    "Cash and cash equivalents, including restricted cash, end of period",
    "Cash, end of period",
    "Cash - End of period",
    "Cash – End of period",
    "Cash end of period",
    "Cash at end of period",
    "Cash at period end",
]

_OCF_KEYWORDS_BURN_BASE = [
    "Net cash used in operating activities - continuing operations",
    "Net cash used in operating activities — continuing operations",
    "Net cash used in operating activities from continuing operations",
    "Net cash used for operating activities - continuing operations",
    "Net cash (used in) operating activities - continuing operations",
    "Net cash (used in) operating activities — continuing operations",
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
    "Cash generated from operations",
    "Cash generated by operations",
    "Net cash inflow from operating activities",
]


_FACT_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
]
_PERIOD_TEXT_PATTERNS = [
    (re.compile(r"three\s+months\s+ended", re.IGNORECASE), 3),
    (re.compile(r"six\s+months\s+ended", re.IGNORECASE), 6),
    (re.compile(r"nine\s+months\s+ended", re.IGNORECASE), 9),
    (re.compile(r"twelve\s+months\s+ended", re.IGNORECASE), 12),
    (re.compile(r"twelve\s+month\s+period", re.IGNORECASE), 12),
    (re.compile(r"year\s+ended", re.IGNORECASE), 12),
]

_PERIOD_MONTH_MAP = {
    "Q1": 3,
    "Q2": 6,
    "Q3": 9,
    "Q4": 12,
    "FY": 12,
    "HY": 6,
    "H1": 6,
    "H2": 6,
    "YTD": 12,
}

_SCALE_MULTIPLIER = {
    "thousands": 1_000,
    "thousand": 1_000,
    "millions": 1_000_000,
    "million": 1_000_000,
}


def _round_half_up(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    try:
        quant = Decimal("1").scaleb(-digits)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return float(value)
    return float(rounded)


def _months_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
    if not start or not end:
        return None
    delta_days = (end - start).days
    if delta_days <= 0:
        return None
    approx_months = delta_days / 30.4375
    if approx_months <= 0:
        return None
    rounded = int(round(approx_months))
    for target in (3, 6, 9, 12):
        if abs(rounded - target) <= 1:
            return target
    return rounded


def _infer_period_months(fact: Optional[dict], form_type: Optional[str]) -> Optional[int]:
    if fact:
        fp_value = str(fact.get("fp") or "").upper()
        if fp_value in _PERIOD_MONTH_MAP:
            months = _PERIOD_MONTH_MAP[fp_value]
            if months in {3, 6, 9, 12}:
                return months
        start = _parse_fact_timestamp(fact.get("start"))
        end = _parse_fact_timestamp(fact.get("end") or fact.get("instant"))
        months = _months_between(start, end)
        if months:
            return months

    if not form_type:
        return None

    defaults = _form_defaults(form_type)
    return defaults.get("period_months_default")


def _normalize_ocf_value(
    value: Optional[float],
    period_months: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[str]]:
    if value is None:
        return None, period_months, None

    if period_months in {3, 6, 9, 12}:
        divisor_map = {3: 1, 6: 2, 9: 3, 12: 4}
        assumption_map = {3: "", 6: "6m/2", 9: "9m/3", 12: "annual/4"}
        divisor = divisor_map[period_months]
        normalized_value = float(value) / float(divisor)
        return normalized_value, period_months, assumption_map[period_months]

    return float(value) if value is not None else None, period_months, None


def _detect_scale_multiplier(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = _SCALE_PATTERN.search(text)
    if not match:
        return None
    keyword = match.group(1).lower()
    return _SCALE_MULTIPLIER.get(keyword)


def _to_float(num_str: str) -> float:
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


def _normalize_for_match(text: str) -> str:
    if not text:
        return text
    # NBSP -> space
    text = text.replace("\u00A0", " ")
    # unify unicode dashes to ASCII hyphen
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    # collapse "- / -" -> "-"
    text = re.sub(r"-\s*/\s*-", "-", text)
    # squeeze whitespace
    return re.sub(r"\s+", " ", text)


def _extract_number_after_keyword(text: str, keywords: Iterable[str]) -> Optional[float]:
    if not text:
        return None
    norm_text = _normalize_for_match(text)
    for keyword in keywords:
        k = _normalize_for_match(keyword)
        pattern = re.compile(re.escape(k), re.IGNORECASE)
        m = pattern.search(norm_text)
        if not m:
            continue
        remainder = norm_text[m.end() : m.end() + 1000]
        for token in remainder.split()[:12]:
            if _NUMERIC_TOKEN_PATTERN.search(token):
                try:
                    return _to_float(token)
                except ValueError:
                    continue
    return None


def _strip_html(html_text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_text or "")
    return " ".join(without_tags.split())


def _extract_html_section(html_text: str, headers: Iterable[str], window: int = 20000) -> Optional[str]:
    if not html_text:
        return None
    lower = html_text.lower()
    for header in headers:
        idx = lower.find(header.lower())
        if idx != -1:
            end = min(len(html_text), idx + window)
            return html_text[idx:end]
    return None


def _parse_html_cashflow_sections(html_text: str) -> dict:
    text = _strip_html(html_text)
    defaults = _form_defaults(None)

    balance_section_html = _extract_html_section(html_text, _BALANCE_HEADERS)
    cashflow_section_html = _extract_html_section(html_text, defaults["cashflow_headers"])

    balance_section_text = _strip_html(balance_section_html) if balance_section_html else None
    cashflow_section_text = _strip_html(cashflow_section_html) if cashflow_section_html else None

    scale_multiplier = 1
    for candidate in (balance_section_html, cashflow_section_html, html_text):
        detected = _detect_scale_multiplier(candidate)
        if detected:
            scale_multiplier = detected
            break

    cash_value = None
    if balance_section_text:
        cash_value = _extract_number_after_keyword(balance_section_text, _CASH_KEYWORDS_BALANCE)
    if cash_value is None and cashflow_section_text:
        cash_value = _extract_number_after_keyword(cashflow_section_text, _CASH_KEYWORDS_FLOW)
    if cash_value is None:
        cash_value = _extract_number_after_keyword(text, _CASH_KEYWORDS_FLOW)

    burn_keywords = defaults["ocf_keywords_burn"]
    provided_keywords = defaults["ocf_keywords_provided"]

    ocf_text_source = cashflow_section_text or text
    burn_value = _extract_number_after_keyword(ocf_text_source, burn_keywords)
    ocf_value = burn_value
    if ocf_value is None:
        ocf_value = _extract_number_after_keyword(ocf_text_source, provided_keywords)
    elif burn_value is not None and burn_value > 0:
        ocf_value = -abs(burn_value)

    if ocf_value is None:
        provided_value = _extract_number_after_keyword(ocf_text_source, provided_keywords)
        if provided_value is not None:
            ocf_value = provided_value

    if cash_value is not None:
        cash_value = float(cash_value) * float(scale_multiplier)
    if burn_value is not None and burn_value > 0:
        burn_value = -abs(float(burn_value))
    if ocf_value is not None and burn_value is None:
        ocf_value = float(ocf_value) * float(scale_multiplier)
    elif burn_value is not None:
        ocf_value = float(burn_value) * float(scale_multiplier)

    period_months_inferred = _infer_months_from_text(cashflow_section_text) or _infer_months_from_text(text)

    evidence_parts = [
        f"cashflow_header={'yes' if cashflow_section_html else 'no'}",
        f"cash_found={'yes' if cash_value is not None else 'no'}",
        f"ocf_found={'yes' if ocf_value is not None else 'no'}",
    ]
    if scale_multiplier != 1:
        evidence_parts.append(f"scale={scale_multiplier}")

    return {
        "found_cashflow_header": bool(cashflow_section_html),
        "cash_value": cash_value,
        "ocf_value": ocf_value,
        "period_months_inferred": period_months_inferred,
        "units_scale": scale_multiplier,
        "evidence": "; ".join(evidence_parts),
    }


def _infer_months_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for pattern, months in _PERIOD_TEXT_PATTERNS:
        if pattern.search(text):
            return months
    return None


def _normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _extract_filing_identifiers(filing_url: str) -> tuple[Optional[str], Optional[str]]:
    parsed = urlparse(filing_url)
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    doc_path = path
    if "doc" in query and query["doc"]:
        doc_path = query["doc"][0]
    doc_path = unquote(doc_path)

    match = _ACCESSION_RE.search(doc_path) or _ACCESSION_RE.search(path)
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
    normalized = match.group(1).upper()
    canonical_map = {
        "10-Q": "10-Q",
        "10-QT": "10-Q",
        "10-K": "10-K",
        "10-KT": "10-K",
        "20-F": "20-F",
        "40-F": "40-F",
        "6-K": "6-K",
    }
    return canonical_map.get(normalized)


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


def _form_defaults(form_type: Optional[str]) -> dict:
    normalized = _normalize_form_type(form_type)
    adapter = FORM_ADAPTERS.get(normalized) if normalized else None

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
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    best_value: Optional[float] = None
    best_form: Optional[str] = None
    best_fact: Optional[dict] = None
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
            best_fact = fact
    return best_value, best_form, best_fact


def _extract_fact_value(
    units: dict, accession: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    preferred_units = ["USD", "USD ", "USD millions", "USDm", "USD $", "USD $ millions"]
    for unit_name in preferred_units:
        facts = units.get(unit_name)
        if not facts:
            continue
        value, form_type, fact = _select_best_fact(facts, accession)
        if value is not None:
            return value, form_type, fact

    all_facts: list[dict] = []
    for facts in units.values():
        all_facts.extend(facts)
    if not all_facts:
        return None, None, None
    return _select_best_fact(all_facts, accession)


def _fetch_xbrl_value(
    cik: str, accession: str, concept: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    base_url = "https://data.sec.gov/api/xbrl/companyconcept"
    taxonomies = ["us-gaap", "ifrs-full"]
    for taxonomy in taxonomies:
        url = f"{base_url}/CIK{cik}/{taxonomy}/{concept}.json"
        try:
            data = _fetch_json(url)
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP error fetching JSON (404)" in message:
                continue
            raise

        units = data.get("units") or {}
        value, form_type, fact = _extract_fact_value(units, accession)
        if value is not None:
            return value, form_type, fact

    return None, None, None


def _derive_from_xbrl(
    filing_url: str, form_type_hint: Optional[str] = None
) -> tuple[Optional[dict], Optional[str]]:
    cik, accession = _extract_filing_identifiers(filing_url)
    if not cik or not accession:
        return None, form_type_hint

    cash = None
    form_type = form_type_hint
    cash_fact: Optional[dict] = None
    cash_concepts = [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsIncludingRestrictedCashCurrent",
        "CashAndCashEquivalentsAndOtherShortTermInvestments",
    ]
    errors: list[str] = []

    for concept in cash_concepts:
        try:
            cash_value, value_form, fact = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if cash_value is not None:
            cash = cash_value
            if form_type is None and value_form:
                form_type = value_form
            cash_fact = fact
            break

    burn = None
    burn_fact: Optional[dict] = None
    burn_concepts = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivitiesExcludingDiscontinuedOperations",
        "NetCashProvidedByUsedInOperatingActivitiesConverted",
        "NetCashFlowsFromUsedInOperatingActivities",
        "NetCashFlowsFromOperatingActivities",
        "NetCashFlowsFromOperations",
    ]
    for concept in burn_concepts:
        try:
            burn_value, value_form, fact = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if burn_value is not None:
            burn = burn_value
            if form_type is None and value_form:
                form_type = value_form
            burn_fact = fact
            break

    if cash is None and burn is None:
        if errors:
            raise RuntimeError("; ".join(errors))
        return None, form_type

    period_months = _infer_period_months(burn_fact, form_type)
    ocf_normalized, period_months, assumption = _normalize_ocf_value(burn, period_months)

    status = "OK"
    if ocf_normalized is None:
        status = "Missing OCF"
    elif period_months not in {3, 6, 9, 12}:
        status = "Missing OCF period"

    result = _finalize_runway_result(
        cash=float(cash) if cash is not None else None,
        ocf_raw=ocf_normalized,
        period_months=period_months,
        assumption=assumption,
        note=f"values parsed from XBRL: {filing_url}" + (" (partial XBRL data)" if cash is None or burn is None else ""),
        form_type=form_type,
        units_scale=1,
        status=status,
        source_tags=["XBRL"],
    )
    return result, form_type


def _infer_runway_estimate(form_type: Optional[str]) -> str:
    if not form_type:
        return ""
    normalized = form_type.upper()
    if normalized.startswith(("10-K", "20-F", "40-F")):
        return "annual_div4"
    return "interim"


def _finalize_runway_result(
    *,
    cash: Optional[float],
    ocf_raw: Optional[float],
    period_months: Optional[int],
    assumption: Optional[str],
    note: str,
    form_type: Optional[str],
    units_scale: Optional[int],
    status: Optional[str],
    source_tags: Optional[Iterable[str]] = None,
) -> dict:
    estimate = _infer_runway_estimate(form_type)

    status_final = status or ""

    quarterly_burn_raw: Optional[float] = None
    runway_quarters_raw: Optional[float] = None
    show_quarters = True

    if ocf_raw is None:
        show_quarters = False
        if not status_final:
            status_final = "Missing OCF"
    else:
        ocf_value = float(ocf_raw)
        if ocf_value >= 0:
            quarterly_burn_raw = 0.0
            show_quarters = False
            status_final = "OCF positive (self-funding)"
        else:
            quarterly_burn_raw = abs(ocf_value)
            if cash is not None and quarterly_burn_raw > 0:
                runway_quarters_raw = float(cash) / float(quarterly_burn_raw)
            if not status_final:
                status_final = "OK"

    cash_value = float(cash) if cash is not None else None

    result: dict = {
        "cash_raw": cash_value,
        "ocf_raw": ocf_raw,
        "quarterly_burn_raw": quarterly_burn_raw,
        "runway_quarters_raw": runway_quarters_raw,
        "note": note,
        "estimate": estimate,
        "status": status_final,
        "assumption": assumption or "",
        "period_months": period_months,
        "units_scale": units_scale or 1,
    }

    result["cash"] = _round_half_up(cash_value)
    result["quarterly_burn"] = (
        _round_half_up(quarterly_burn_raw) if quarterly_burn_raw is not None else None
    )
    result["runway_quarters"] = (
        _round_half_up(runway_quarters_raw)
        if show_quarters and runway_quarters_raw is not None
        else None
    )

    if form_type:
        result["form_type"] = form_type

    months_valid = period_months in {3, 6, 9, 12}
    complete = (cash_value is not None and ocf_raw is not None and months_valid)
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


def get_runway_from_filing(filing_url: str) -> dict:
    form_type_hint = _infer_form_type_from_url(filing_url)

    xbrl_result: Optional[dict] = None
    xbrl_error: Optional[str] = None
    detected_form_type: Optional[str] = None
    try:
        xbrl_result, detected_form_type = _derive_from_xbrl(
            filing_url, form_type_hint=form_type_hint
        )
    except Exception as exc:
        xbrl_error = f"XBRL parse failed ({exc.__class__.__name__}: {exc})"

    if xbrl_result and xbrl_error:
        xbrl_result = dict(xbrl_result)
        xbrl_result["note"] = f"{xbrl_result['note']} (with warning: {xbrl_error})"

    if xbrl_result and xbrl_result.get("complete"):
        result_copy = dict(xbrl_result)
        if not result_copy.get("source_tags"):
            result_copy["source_tags"] = ["XBRL"]
        return result_copy

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

    form_type_candidates = [
        detected_form_type,
        xbrl_result.get("form_type") if xbrl_result else None,
        form_type_hint,
        _infer_form_type_from_text(text),
    ]
    form_type: Optional[str] = None
    for candidate in form_type_candidates:
        normalized_candidate = _normalize_form_type(candidate)
        if normalized_candidate:
            form_type = normalized_candidate
            break

    defaults = _form_defaults(form_type)
    cashflow_headers = defaults["cashflow_headers"]

    if form_type == "6-K":
        lower_html = html_text.lower()
        header_present = any(header.lower() in lower_html for header in cashflow_headers)
        if not header_present:
            note_parts = [f"6-K missing operating cash flow statement headers: {filing_url}"]
            if xbrl_result:
                for suffix in _extract_note_suffix(xbrl_result.get("note")):
                    cleaned_suffix = suffix.replace(filing_url, "").strip()
                    if cleaned_suffix:
                        note_parts.append(cleaned_suffix)
            elif xbrl_error:
                note_parts.append(xbrl_error)

            period_value = None
            if xbrl_result and xbrl_result.get("period_months") in {3, 6, 9, 12}:
                period_value = xbrl_result.get("period_months")
            else:
                period_value = defaults.get("period_months_default")

            assumption_value = (xbrl_result.get("assumption") if xbrl_result else "") or ""
            units_scale_value = (xbrl_result.get("units_scale") if xbrl_result else None) or 1

            source_tags: list[str] = []
            if xbrl_result and xbrl_result.get("cash_raw") is not None:
                source_tags.append("XBRL")

            return _finalize_runway_result(
                cash=xbrl_result.get("cash_raw") if xbrl_result else None,
                ocf_raw=None,
                period_months=period_value,
                assumption=assumption_value,
                note="; ".join(part for part in note_parts if part),
                form_type=form_type,
                units_scale=units_scale_value,
                status="6-K missing OCF",
                source_tags=source_tags or None,
            )

    html_info = _parse_html_cashflow_sections(html_text)
    html_cash = html_info.get("cash_value")
    html_ocf_value = html_info.get("ocf_value")
    html_period_inferred = html_info.get("period_months_inferred")
    html_units_scale = html_info.get("units_scale") or 1
    html_evidence = html_info.get("evidence") or ""

    period_from_xbrl = None
    if xbrl_result and xbrl_result.get("period_months") in {3, 6, 9, 12}:
        period_from_xbrl = xbrl_result.get("period_months")

    period_from_html = (
        html_period_inferred if html_period_inferred in {3, 6, 9, 12} else None
    )

    final_period = period_from_xbrl or period_from_html or defaults.get("period_months_default")

    period_for_html_normalization = (
        html_period_inferred
        if html_period_inferred in {3, 6, 9, 12}
        else final_period
    )
    if period_for_html_normalization is None and html_period_inferred is not None:
        period_for_html_normalization = html_period_inferred

    html_ocf_normalized = None
    html_assumption = ""
    if html_ocf_value is not None:
        html_ocf_normalized, normalized_period, html_assumption = _normalize_ocf_value(
            html_ocf_value, period_for_html_normalization
        )
        if final_period is None and normalized_period is not None:
            final_period = normalized_period
        html_assumption = html_assumption or ""

    xbrl_cash = xbrl_result.get("cash_raw") if xbrl_result else None
    xbrl_ocf = xbrl_result.get("ocf_raw") if xbrl_result else None
    xbrl_assumption = (xbrl_result.get("assumption") if xbrl_result else "") or ""
    xbrl_units_scale = (xbrl_result.get("units_scale") if xbrl_result else None) or 1

    final_cash = None
    cash_source: Optional[str] = None
    if xbrl_cash is not None:
        final_cash = xbrl_cash
        cash_source = "XBRL"
    elif html_cash is not None:
        final_cash = html_cash
        cash_source = "HTML"

    final_ocf_raw = None
    final_assumption = ""
    ocf_source: Optional[str] = None
    if xbrl_ocf is not None:
        final_ocf_raw = xbrl_ocf
        final_assumption = xbrl_assumption
        ocf_source = "XBRL"
    elif html_ocf_normalized is not None:
        final_ocf_raw = html_ocf_normalized
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
    if period_from_xbrl in {3, 6, 9, 12} and final_period == period_from_xbrl:
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
        note_parts.append(f"values parsed from XBRL and HTML (merged): {filing_url}")
    elif used_xbrl:
        note_parts.append(f"values parsed from XBRL: {filing_url}")
    else:
        note_parts.append(f"values parsed from filing HTML: {filing_url}")

    if used_html and html_evidence:
        note_parts.append(html_evidence)

    if xbrl_result:
        for suffix in _extract_note_suffix(xbrl_result.get("note")):
            cleaned_suffix = suffix.replace(filing_url, "").strip()
            if cleaned_suffix:
                note_parts.append(cleaned_suffix)
    elif xbrl_error:
        note_parts.append(xbrl_error)

    note_text = "; ".join(part for part in note_parts if part)

    return _finalize_runway_result(
        cash=final_cash,
        ocf_raw=final_ocf_raw,
        period_months=final_period,
        assumption=final_assumption,
        note=note_text,
        form_type=form_type,
        units_scale=final_units_scale,
        status=None,
        source_tags=source_tags or None,
    )


__all__ = ["get_runway_from_filing", "url_matches_form"]
