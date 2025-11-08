"""iXBRL helpers for runway parsing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

from .router import _normalize_form_type  # type: ignore[attr-defined]
from .units import normalize_ocf_value
from .postproc import finalize_runway_result

from urllib.parse import parse_qs, unquote, urlparse
import re

from .router import _fetch_json  # type: ignore[attr-defined]


_FACT_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
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


_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)


def normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def parse_fact_timestamp(value: object) -> Optional[datetime]:
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


def months_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
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


def infer_period_months(fact: Optional[dict], form_type: Optional[str]) -> Optional[int]:
    if fact:
        fp_value = str(fact.get("fp") or "").upper()
        if fp_value in _PERIOD_MONTH_MAP:
            months = _PERIOD_MONTH_MAP[fp_value]
            if months in {3, 6, 9, 12}:
                return months
        start = parse_fact_timestamp(fact.get("start"))
        end = parse_fact_timestamp(fact.get("end") or fact.get("instant"))
        months = months_between(start, end)
        if months:
            return months

    if not form_type:
        return None

    from .router import _form_defaults  # local import to avoid cycles

    defaults = _form_defaults(form_type)
    return defaults.get("period_months_default")


def extract_filing_identifiers(filing_url: str) -> Tuple[Optional[str], Optional[str]]:
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

    cik_digits = normalize_digits(match.group(1))
    accession_digits = normalize_digits(match.group(2))
    if not cik_digits or not accession_digits:
        return None, None

    cik_padded = cik_digits.zfill(10)
    if len(accession_digits) < 18:
        accession_digits = accession_digits.zfill(18)
    accession = f"{accession_digits[:10]}-{accession_digits[10:12]}-{accession_digits[12:]}"
    return cik_padded, accession


def select_best_fact(
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
        end_dt = parse_fact_timestamp(
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


def extract_fact_value(
    units: dict, accession: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    preferred_units = ["USD", "USD ", "USD millions", "USDm", "USD $", "USD $ millions"]
    for unit_name in preferred_units:
        facts = units.get(unit_name)
        if not facts:
            continue
        value, form_type, fact = select_best_fact(facts, accession)
        if value is not None:
            return value, form_type, fact

    all_facts: list[dict] = []
    for facts in units.values():
        all_facts.extend(facts)
    if not all_facts:
        return None, None, None
    return select_best_fact(all_facts, accession)


def fetch_xbrl_value(
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
        value, form_type, fact = extract_fact_value(units, accession)
        if value is not None:
            return value, form_type, fact

    return None, None, None


def derive_from_xbrl(
    filing_url: str,
    form_type_hint: Optional[str],
) -> tuple[Optional[dict], Optional[str]]:
    cik, accession = extract_filing_identifiers(filing_url)
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
            cash_value, value_form, fact = fetch_xbrl_value(cik, accession, concept)
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
            burn_value, value_form, fact = fetch_xbrl_value(cik, accession, concept)
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

    period_months = infer_period_months(burn_fact, form_type)
    ocf_normalized, period_months, assumption = normalize_ocf_value(burn, period_months)

    status = "OK"
    if ocf_normalized is None:
        status = "Missing OCF"
    elif period_months not in {3, 6, 9, 12}:
        status = "Missing OCF period"

    result = finalize_runway_result(
        cash=float(cash) if cash is not None else None,
        ocf_raw=float(burn) if burn is not None else None,
        ocf_quarterly=ocf_normalized,
        period_months=period_months,
        assumption=assumption,
        note=f"values parsed from XBRL: {filing_url}" + (" (partial XBRL data)" if cash is None or burn is None else ""),
        form_type=form_type,
        units_scale=1,
        status=status,
        source_tags=["XBRL"],
    )
    return result, form_type


__all__ = [
    "derive_from_xbrl",
    "fetch_xbrl_value",
    "extract_filing_identifiers",
    "infer_period_months",
    "months_between",
    "parse_fact_timestamp",
]
