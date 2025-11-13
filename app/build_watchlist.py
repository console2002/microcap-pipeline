"""Build a validated watchlist from deep research results and market data."""
from __future__ import annotations

import csv
import os
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

import pandas as pd

from app.config import load_config
from app.utils import ensure_csv, log_line, utc_now_iso


ProgressFn = Optional[Callable[[str], None]]


_OUTPUT_COLUMNS = [
    "Ticker",
    "Company",
    "Sector",
    "CIK",
    "Price",
    "MarketCap",
    "ADV20",
    "LatestForm",
    "LatestFiledAt",
    "LatestFiledAgeDays",
    "CatalystType",
    "Catalyst",
    "CatalystStatus",
    "CatalystPrimaryForm",
    "CatalystPrimaryDate",
    "CatalystPrimaryDaysAgo",
    "CatalystPrimaryURL",
    "RunwayQuartersRaw",
    "RunwayQuarters",
    "RunwayCashRaw",
    "RunwayCash",
    "RunwayQuarterlyBurnRaw",
    "RunwayQuarterlyBurn",
    "RunwayEstimate",
    "RunwayNotes",
    "RunwayDays",
    "RunwayBucket",
    "RunwaySourceForm",
    "RunwaySourceDate",
    "RunwaySourceDaysAgo",
    "RunwaySourceURL",
    "RunwayStatus",
    "DilutionFlag",
    "Dilution",
    "DilutionStatus",
    "DilutionFormsHit",
    "DilutionKeywordsHit",
    "DilutionPrimaryForm",
    "DilutionPrimaryDate",
    "DilutionPrimaryDaysAgo",
    "DilutionPrimaryURL",
    "DilutionEvidenceAll",
    "Governance",
    "GovernanceStatus",
    "GovernancePrimaryForm",
    "GovernancePrimaryDate",
    "GovernancePrimaryDaysAgo",
    "GovernancePrimaryURL",
    "GovernanceEvidenceAll",
    "GovernanceNotes",
    "HasGoingConcern",
    "HasMaterialWeakness",
    "Auditor",
    "Insider",
    "InsiderStatus",
    "InsiderForms345Links",
    "InsiderBuyCount",
    "HasInsiderCluster",
    "Ownership",
    "OwnershipStatus",
    "OwnershipLinks",
    "Materiality",
    "SubscoresEvidenced",
    "SubscoresEvidencedCount",
    "Status",
    "ChecklistPassed",
    "ChecklistCatalyst",
    "ChecklistDilution",
    "ChecklistRunway",
    "ChecklistGovernance",
    "ChecklistInsider",
    "ChecklistOwnership",
    "RiskFlag",
    "Tier1Type",
    "Tier1Trigger",
]

_NEGATIVE_DILUTION_TERMS = (
    "high risk",
    "toxic",
    "at-the-market",
)

_SUBSCORE_SPLIT_RE = re.compile(r"[;\n]+")
_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_URL_RE = re.compile(r"https?://\S+")

_MANAGER_FORM_PREFIXES = (
    "13F",
    "13F-HR",
    "SC 13D",
    "SC 13G",
)

_DILUTION_FORM_PREFIXES = (
    "S-1",
    "S-3",
    "F-1",
    "F-3",
    "424B",
    "S-8",
)

_DILUTION_OFFERING_PREFIXES = (
    "S-1",
    "S-3",
    "F-1",
    "F-3",
    "424B",
)

_DILUTION_KEYWORDS = [
    "equity distribution agreement",
    "atm",
    "at-the-market",
    "sales agreement",
    "registered direct",
    "underwritten",
    "follow-on",
    "prospectus supplement",
    "rule 415",
    "shelf",
    "pipe",
    "subscription agreement",
    "convertible note",
    "convertible debenture",
    "convertible",
    "warrant",
    "pre-funded",
]

_FORM_URL_PATTERNS = {
    "10q": re.compile(r"/[^/]*10-?q"),
    "10k": re.compile(r"/[^/]*10-?k"),
    "8k": re.compile(r"/[^/]*8-?k"),
    "6k": re.compile(r"/[^/]*6-?k"),
    "20f": re.compile(r"/[^/]*20-?f"),
    "40f": re.compile(r"/[^/]*40-?f"),
    "def14a": re.compile(r"/[^/]*def14a"),
    "424b": re.compile(r"/[^/]*424b"),
    "s1": re.compile(r"/[^/]*s-?1"),
    "s3": re.compile(r"/[^/]*s-?3"),
    "f1": re.compile(r"/[^/]*f-?1"),
    "f3": re.compile(r"/[^/]*f-?3"),
    "8a12b": re.compile(r"/[^/]*8a12b"),
}

_GOVERNANCE_VALID_FORMS = {"DEF 14A", "10-K", "20-F", "40-F"}
_INSIDER_LINK_RE = re.compile(r"(form[345]|xslf345)", re.IGNORECASE)
_OWNERSHIP_LINK_RE = re.compile(r"(13f|sc13d|sc13g)", re.IGNORECASE)

_CATALYST_ITEM_LABELS = {
    "1.01": "Material Agreement",
    "2.02": "Earnings Results",
    "3.02": "Unregistered Sales",
    "7.01": "Reg FD",
    "8.01": "Other Events",
}

_CATALYST_KEYWORD_LABELS = [
    (re.compile(r"guidance"), "Guidance"),
    (re.compile(r"contract|award|partnership"), "Commercial Update"),
    (re.compile(r"regulatory|approval|clearance|adcom|pdufa"), "Regulatory"),
    (re.compile(r"bankruptcy|restructuring"), "Restructuring"),
    (re.compile(r"buyback"), "Buyback"),
    (re.compile(r"uplist|downlist"), "Listing Change"),
]


def _progress_log_path() -> str:
    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "progress.csv")
    ensure_csv(path, ["timestamp", "status", "message"])
    return path


def _emit(status: str, message: str, progress_fn: ProgressFn) -> None:
    path = _progress_log_path()
    timestamp = utc_now_iso()
    log_line(path, [timestamp, status, message])
    if progress_fn is not None:
        try:
            progress_fn(f"{status} {message}")
        except Exception:
            pass


def _resolve_path(filename: str, base_dir: str | None) -> str:
    candidates: list[str] = []
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    candidates.extend([filename, os.path.join("data", filename)])
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, filename) if base_dir else filename


def _normalize_ticker(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return text


def _normalize_cik(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(10)


def _count_subscores(raw: object) -> int:
    if raw is None:
        return 0
    text = str(raw).strip()
    if not text:
        return 0
    parts = [segment.strip() for segment in _SUBSCORE_SPLIT_RE.split(text)]
    return sum(1 for part in parts if part)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _csv_cell_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    return value


def _round_half_up_number(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        quant = Decimal("1").scaleb(-digits)
        return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return float(value)


def _runway_bucket(quarters: Optional[float]) -> str:
    if quarters is None:
        return ""
    if quarters < 1:
        return "<1q"
    if quarters < 3:
        return "1-3q"
    if quarters < 6:
        return "3-6q"
    if quarters < 12:
        return "6-12q"
    return ">12q"


def _normalize_form_name(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"\s+", " ", text)
    return text


def _form_guard_key(value: str) -> str:
    text = _normalize_form_name(value)
    if text.startswith("FORM "):
        text = text[5:]
    text = text.replace(" ", "")
    if text.endswith("/A"):
        text = text[:-2]
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_manager_form(value: str) -> bool:
    text = _normalize_form_name(value)
    return any(text.startswith(prefix) for prefix in _MANAGER_FORM_PREFIXES)


def _url_matches_form(form: str, url: str) -> bool:
    if not form or not url:
        return True

    key = _form_guard_key(form)
    pattern = next(
        (regex for prefix, regex in _FORM_URL_PATTERNS.items() if key.startswith(prefix)),
        None,
    )
    if pattern is None:
        return True

    parsed = urlsplit(url)
    lowered_url = url.lower()

    doc_candidates: list[str] = []
    if parsed.query:
        query = parse_qs(parsed.query)
        for candidate_key in ("doc", "filename", "file", "document"):
            for value in query.get(candidate_key, []):
                if value:
                    doc_candidates.append(unquote(value))

    text_to_check = " ".join([lowered_url, *[candidate.lower() for candidate in doc_candidates]])
    if pattern.search(text_to_check):
        return True

    for other_prefix, other_pattern in _FORM_URL_PATTERNS.items():
        if other_prefix == key:
            continue
        if other_pattern.search(text_to_check):
            return False

    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    doc_paths = [candidate.lower() for candidate in doc_candidates]

    if netloc.endswith("sec.gov"):
        if "/archives/edgar/data/" in path:
            return True
        if path.startswith(("/ix", "/cgi-bin/ix", "/cgi-bin/viewer")):
            return True
        if any("/archives/edgar/data/" in candidate for candidate in doc_paths):
            return True

    return False


def _parse_date_value(value: object) -> Optional[pd.Timestamp]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        if _is_iso_date_string(text):
            ts = pd.to_datetime(text, utc=True, errors="coerce", format="%Y-%m-%d")
        else:
            ts = pd.to_datetime(text, utc=True, errors="coerce", dayfirst=True)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        if ts.empty:
            return None
        ts = ts.iloc[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.normalize()


def _format_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d")


def _is_iso_date_string(value: str) -> bool:
    return bool(value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _compute_days_ago(ts: Optional[pd.Timestamp], now: pd.Timestamp) -> str:
    if ts is None:
        return ""
    delta = now - ts
    if pd.isna(delta):
        return ""
    try:
        return str(int(delta.days))
    except Exception:
        return ""


def _select_primary_entry(entries: list[dict]) -> Optional[dict]:
    if not entries:
        return None

    def sort_key(entry: dict) -> tuple[int, pd.Timestamp, int]:
        date_val = entry.get("date_value")
        has_date = 1 if isinstance(date_val, pd.Timestamp) else 0
        sortable = date_val if isinstance(date_val, pd.Timestamp) else pd.Timestamp.min.tz_localize("UTC")
        has_url = 1 if entry.get("url") else 0
        return (has_date, sortable, has_url)

    ordered = sorted(entries, key=sort_key, reverse=True)
    return ordered[0] if ordered else None


def _entry_core(entry: dict) -> Optional[tuple[str, str, str]]:
    url = _clean_text(entry.get("url"))
    if not url:
        return None

    date_value = entry.get("date_value")
    if isinstance(date_value, pd.Timestamp):
        date_str = _format_date(date_value)
    else:
        date_str = entry.get("date") or ""
    if date_str and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        parsed = _parse_date_value(date_str)
        date_str = _format_date(parsed)

    form = _normalize_form_name(entry.get("form", ""))

    if not (date_str and form):
        return None

    return date_str, form, url


def _extract_entry_description(entry: dict, url: str) -> str:
    raw_text = entry.get("raw") or ""
    if not raw_text:
        return ""

    idx = raw_text.find(url) if url else -1
    if idx != -1:
        remainder = raw_text[idx + len(url) :]
    else:
        match = _URL_RE.search(raw_text)
        if not match:
            return ""
        remainder = raw_text[match.end() :]

    desc_text = remainder.strip(" |")
    if not desc_text:
        return ""
    desc_text = re.sub(r"\s+", " ", desc_text)
    return desc_text


def _entry_to_text(entry: dict) -> str:
    core = _entry_core(entry)
    if core is None:
        return ""

    date_str, form, url = core
    base_text = f"{date_str} {form} {url}"
    desc_text = _extract_entry_description(entry, url)
    if desc_text:
        return f"{base_text} {desc_text}"
    return base_text


def _format_evidence_entries(entries: list[dict]) -> str:
    formatted: list[str] = []
    seen_core: set[str] = set()
    for entry in entries:
        core = _entry_core(entry)
        if core is None:
            continue
        core_text = f"{core[0]} {core[1]} {core[2]}"
        if core_text in seen_core:
            continue
        seen_core.add(core_text)
        text = _entry_to_text(entry)
        if text:
            formatted.append(text)
    return "; ".join(formatted)


def detect_tier1(evidence_text: str) -> tuple[str, bool]:
    text = (evidence_text or "").lower()
    patterns = [
        ("FDA", r"(510\(k\)|de ?novo|pdufa|approval|cleared|clearance|crl|complete response letter|adcom vote|ind (accepted|cleared))"),
        ("GuidanceUp", r"(raises|increases|hikes)\s+guidance|guidance.*(raised|increased)"),
        ("FundedAward", r"(award|contract).*(obligated|obligation).*\$[0-9]"),
        ("OverhangRemoval", r"(terminate(d|s)?|withdraw(n|s)?)\s.*(atm|shelf|sales agreement|equity distribution)"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            return label, True
    return "None", False


def _enforce_form_url_guard(
    ticker: str,
    subscore: str,
    summary: dict,
    entries: list[dict],
    now: pd.Timestamp,
    progress_fn: ProgressFn,
) -> dict:
    form = summary.get("primary_form", "")
    url = summary.get("primary_url", "")
    if not form or not url:
        return summary

    if _url_matches_form(form, url):
        return summary

    message = f"Fix(FormURLMismatch): {ticker} {subscore} expected={form} gotURL={url}"

    for entry in entries:
        candidate_form = entry.get("form") or ""
        candidate_url = entry.get("url") or ""
        if candidate_form and candidate_url and _url_matches_form(candidate_form, candidate_url):
            summary["primary_entry"] = entry
            summary["primary_form"] = candidate_form
            summary["primary_date"] = entry.get("date", "")
            summary["primary_days_ago"] = _compute_days_ago(entry.get("date_value"), now)
            summary["primary_url"] = candidate_url
            summary["primary_text"] = _entry_to_text(entry)
            summary["primary_raw"] = entry.get("raw", "")
            summary["status"] = "Pass"
            _emit("WARN", message, progress_fn)
            return summary

    _emit("WARN", message, progress_fn)
    return summary


def _extract_urls_list(text: str) -> list[str]:
    urls = _URL_RE.findall(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _infer_catalyst_type(entries: list[dict], fallback_form: str, fallback_text: str) -> str:
    for entry in entries:
        lower_text = entry.get("lower", "")
        for item, label in _CATALYST_ITEM_LABELS.items():
            if f"item {item.lower()}" in lower_text:
                return label
        for pattern, label in _CATALYST_KEYWORD_LABELS:
            if pattern.search(lower_text):
                return label

    form = _normalize_form_name(fallback_form)
    if form in {"10-Q", "10-K", "20-F", "40-F"}:
        return "Earnings/Financial update"

    lower_fallback = fallback_text.lower()
    for pattern, label in _CATALYST_KEYWORD_LABELS:
        if pattern.search(lower_fallback):
            return label
    return ""


def _is_status_pass(status: str) -> bool:
    text = _clean_text(status).lower()
    if not text:
        return False
    return text.startswith("pass") or text.startswith("overhang")


def _normalize_checklist_status(status_value: object) -> str:
    text = _clean_text(status_value)
    if not text:
        return "Missing"

    if _is_status_pass(text):
        return "Pass"

    lowered = text.lower()
    if lowered in {"missing", "placeholder", "tbd", "n/a", "na", "not available"}:
        return "Missing"

    return "Fail"


def _parse_evidence_entries(text: str) -> list[dict]:
    entries: list[dict] = []
    if not text:
        return entries

    for segment in _SUBSCORE_SPLIT_RE.split(text):
        raw = segment.strip()
        if not raw:
            continue

        url_match = _URL_RE.search(raw)
        url = url_match.group(0) if url_match else ""

        date_match = _DATE_RE.search(raw)
        date_val = _parse_date_value(date_match.group(1)) if date_match else None

        remainder = raw
        if date_match:
            remainder = raw[date_match.end() :].lstrip(" |") or remainder

        candidate_segment = remainder.split("|")[0].strip()
        if candidate_segment:
            http_pos = candidate_segment.lower().find("http")
            if http_pos != -1:
                candidate_segment = candidate_segment[:http_pos].rstrip()

        form = ""
        if candidate_segment:
            match = re.match(r"([A-Za-z0-9./-]+(?:\s+[A-Za-z0-9.-]+)?)", candidate_segment)
            if match:
                form = match.group(1)

        normalized_form = _normalize_form_name(form)

        if date_val is None:
            # attempt to parse the leading token if a date wasn't captured by regex
            leading_token = candidate_segment.split()[0] if candidate_segment else ""
            alt_date = _parse_date_value(leading_token)
            if alt_date is not None:
                date_val = alt_date

        entries.append(
            {
                "raw": raw,
                "date": _format_date(date_val),
                "date_value": date_val,
                "form": normalized_form,
                "url": url,
                "lower": raw.lower(),
            }
        )

    return entries


def _first_url(text: str) -> str:
    if not text:
        return ""
    match = _URL_RE.search(text)
    if match:
        return match.group(0)
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _URL_RE.search(chunk)
        if match:
            return match.group(0)
    return ""


def _summarize_category(entries: list[dict], now: pd.Timestamp) -> dict:
    summary = {
        "status": "Missing",
        "primary_text": "",
        "primary_raw": "",
        "primary_form": "",
        "primary_date": "",
        "primary_days_ago": "",
        "primary_url": "",
        "primary_entry": None,
    }

    primary = _select_primary_entry(entries)
    if not primary:
        return summary

    summary["status"] = "Pass"
    summary["primary_entry"] = primary
    summary["primary_form"] = primary.get("form", "")
    summary["primary_date"] = primary.get("date", "")
    summary["primary_days_ago"] = _compute_days_ago(primary.get("date_value"), now)
    summary["primary_url"] = primary.get("url", "")
    summary["primary_text"] = _entry_to_text(primary)
    summary["primary_raw"] = primary.get("raw", "")
    return summary


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and pd.isna(value):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(result):
            return None
        return result
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        result = float(text)
    except ValueError:
        return None
    if pd.isna(result):
        return None
    return result


def _select_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", col.lower()): col for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _load_market_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    ticker_col = _select_column(df, ["Ticker"])
    cik_col = _select_column(df, ["CIK"])
    price_col = _select_column(df, ["Price", "Close", "Last", "LastPrice"])
    mc_col = _select_column(df, ["MarketCap", "Market_Cap", "MarketCapitalization"])
    adv_col = _select_column(df, ["ADV20", "AverageVolume20", "AverageVolume20Day"])

    columns = {}
    if ticker_col:
        columns["Ticker"] = df[ticker_col].apply(_normalize_ticker)
    else:
        columns["Ticker"] = pd.Series(dtype="string")

    if cik_col:
        columns["CIK"] = df[cik_col].apply(_normalize_cik)
    else:
        columns["CIK"] = pd.Series(dtype="string")

    if price_col:
        columns["Price"] = df[price_col]
    else:
        columns["Price"] = pd.Series(dtype="float64")

    if mc_col:
        columns["MarketCap"] = df[mc_col]
    else:
        columns["MarketCap"] = pd.Series(dtype="float64")

    if adv_col:
        columns["ADV20"] = df[adv_col]
    else:
        columns["ADV20"] = pd.Series(dtype="float64")

    market = pd.DataFrame(columns)
    market["Ticker"] = market["Ticker"].fillna("").astype(str).str.strip().str.upper()
    market["CIK"] = market["CIK"].fillna("").astype(str).apply(_normalize_cik)
    return market


def _merge_market_info(df: pd.DataFrame, market: pd.DataFrame) -> None:
    if market.empty:
        return

    market = market.copy()
    market["Ticker_norm"] = market["Ticker"].apply(_normalize_ticker)
    market["CIK_norm"] = market["CIK"].apply(_normalize_cik)

    by_ticker = (
        market[market["Ticker_norm"] != ""]
        .drop_duplicates(subset=["Ticker_norm"], keep="last")
        .set_index("Ticker_norm")
    )

    by_cik = (
        market[market["CIK_norm"] != ""]
        .drop_duplicates(subset=["CIK_norm"], keep="last")
        .set_index("CIK_norm")
    )

    for column in ["Price", "MarketCap", "ADV20"]:
        if column not in df.columns:
            df[column] = pd.Series(dtype="float64")

        if not by_ticker.empty and column in by_ticker.columns:
            ticker_series = df["Ticker_norm"].map(by_ticker[column])
            df[column] = df[column].where(df[column].notna(), ticker_series)

        if not by_cik.empty and column in by_cik.columns:
            cik_series = df["CIK_norm"].map(by_cik[column])
            df[column] = df[column].where(df[column].notna(), cik_series)


def run(
    data_dir: str | None = None,
    progress_fn: ProgressFn = None,
    stop_flag: dict | None = None,
) -> tuple[int, str]:
    """Build the validated watchlist.

    Returns a tuple ``(rows_written, status)`` where ``status`` is one of
    ``"ok"``, ``"missing_source"``, or ``"stopped"``.
    """

    if stop_flag is None:
        stop_flag = {}

    if data_dir is None:
        cfg = load_config()
        data_dir = cfg.get("Paths", {}).get("data", "data")

    os.makedirs(data_dir, exist_ok=True)

    research_path = _resolve_path("research_results_full.csv", data_dir)
    if not os.path.exists(research_path):
        _emit("ERROR", "build_watchlist: research_results_full.csv not found", progress_fn)
        return 0, "missing_source"

    _emit("INFO", "build_watchlist: reading research_results_full.csv", progress_fn)
    research_df = pd.read_csv(research_path, encoding="utf-8")

    if research_df.empty:
        _emit("INFO", "build_watchlist: research results empty", progress_fn)

    research_df = research_df.copy()

    if "Ticker" not in research_df.columns:
        research_df["Ticker"] = ""
    if "CIK" not in research_df.columns:
        research_df["CIK"] = ""
    if "SubscoresEvidenced" not in research_df.columns:
        research_df["SubscoresEvidenced"] = ""

    research_df["Ticker_norm"] = research_df["Ticker"].apply(_normalize_ticker)
    research_df["CIK_norm"] = research_df["CIK"].apply(_normalize_cik)
    research_df["SubscoresEvidencedCount"] = research_df["SubscoresEvidenced"].apply(_count_subscores)

    candidates_path = _resolve_path("candidates.csv", data_dir)
    prices_path = _resolve_path("prices.csv", data_dir)

    market_frames = []
    for path in (candidates_path, prices_path):
        market_df = _load_market_data(path)
        if not market_df.empty:
            _emit("INFO", f"build_watchlist: loaded market data from {os.path.basename(path)}", progress_fn)
            market_frames.append(market_df)
        else:
            if os.path.exists(path):
                _emit("INFO", f"build_watchlist: {os.path.basename(path)} has no usable rows", progress_fn)
            else:
                _emit("INFO", f"build_watchlist: {os.path.basename(path)} not found", progress_fn)

    combined_market = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
    _merge_market_info(research_df, combined_market)

    total_rows = len(research_df)
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    survivors: list[dict] = []
    status = "ok"

    for idx, row in enumerate(research_df.itertuples(index=False), start=1):
        if stop_flag.get("stop"):
            status = "stopped"
            _emit("INFO", "build_watchlist: stop requested", progress_fn)
            break

        if idx == 1 or idx % 25 == 0 or idx == total_rows:
            _emit("INFO", f"build_watchlist: processing row {idx} of {total_rows}", progress_fn)

        ticker_value = _clean_text(getattr(row, "Ticker", ""))
        cik_value = _normalize_cik(getattr(row, "CIK", ""))
        identifier = ticker_value or cik_value
        company_value = getattr(row, "Company", "")
        sector_text = _clean_text(getattr(row, "Sector", ""))

        price_val = _to_float(getattr(row, "Price", None))
        market_cap_val = _to_float(getattr(row, "MarketCap", None))
        adv_val = _to_float(getattr(row, "ADV20", None))

        if price_val is None or price_val < 1:
            continue
        if market_cap_val is None or market_cap_val >= 350_000_000:
            continue
        if adv_val is None or adv_val < 40_000:
            continue

        runway_quarters_raw = _to_float(getattr(row, "RunwayQuartersRaw", None))
        if runway_quarters_raw is None:
            runway_quarters_raw = _to_float(getattr(row, "RunwayQuarters", None))
        runway_quarters = (
            _round_half_up_number(runway_quarters_raw)
            if runway_quarters_raw is not None
            else None
        )

        runway_cash_raw = _to_float(getattr(row, "RunwayCashRaw", None))
        if runway_cash_raw is None:
            runway_cash_raw = _to_float(getattr(row, "RunwayCash", None))
        runway_cash = _round_half_up_number(
            _to_float(getattr(row, "RunwayCash", runway_cash_raw))
        )

        runway_burn_raw = _to_float(getattr(row, "RunwayQuarterlyBurnRaw", None))
        if runway_burn_raw is None:
            runway_burn_raw = _to_float(getattr(row, "RunwayQuarterlyBurn", None))
        runway_burn = _round_half_up_number(
            _to_float(getattr(row, "RunwayQuarterlyBurn", runway_burn_raw))
        )

        runway_estimate = _clean_text(getattr(row, "RunwayEstimate", ""))
        runway_notes = _clean_text(getattr(row, "RunwayNotes", ""))

        runway_burn_missing = (
            (runway_burn_raw is None or runway_burn_raw <= 0)
            and (runway_burn is None or runway_burn <= 0)
        )
        if runway_burn_missing or not runway_estimate:
            _emit("WARN", f"Dropped: Missing runway burn metrics ({identifier})", progress_fn)
            continue
        runway_source_form = _normalize_form_name(_clean_text(getattr(row, "RunwaySourceForm", "")))
        runway_source_date_ts = _parse_date_value(getattr(row, "RunwaySourceDate", ""))
        runway_source_url = _clean_text(getattr(row, "RunwaySourceURL", ""))

        has_runway_values = (
            runway_quarters is not None
            and runway_cash is not None
            and runway_burn is not None
        )

        runway_days_value: Optional[int] = None
        runway_bucket_label = ""

        if has_runway_values and runway_quarters is not None:
            runway_days_value = int(round(runway_quarters * 90))
            runway_bucket_label = _runway_bucket(runway_quarters)
            if not runway_source_form or runway_source_date_ts is None or not runway_source_url:
                _emit("WARN", f"Dropped: Missing runway source details ({identifier})", progress_fn)
                continue
            if (now_utc - runway_source_date_ts).days > 465:
                _emit("WARN", f"Dropped: Runway filing stale ({identifier})", progress_fn)
                continue
            runway_status = "Pass" if runway_quarters > 0 else "Fail"
        else:
            runway_status = "Missing"
            runway_estimate = ""
            runway_source_form = ""
            runway_source_date_ts = None
            runway_source_url = ""

        runway_source_date = _format_date(runway_source_date_ts)
        runway_source_days = _compute_days_ago(runway_source_date_ts, now_utc)

        catalyst_text = _clean_text(getattr(row, "Catalyst", ""))
        if not catalyst_text or catalyst_text.upper() == "TBD":
            _emit("WARN", f"Dropped: Catalyst evidence missing ({identifier})", progress_fn)
            continue

        dilution_text = _clean_text(getattr(row, "Dilution", ""))
        if any(term in dilution_text.lower() for term in _NEGATIVE_DILUTION_TERMS):
            continue

        governance_text = _clean_text(getattr(row, "Governance", ""))
        if not governance_text:
            continue

        filings_summary_text = _clean_text(getattr(row, "FilingsSummary", ""))
        filing_entries = [
            entry
            for entry in _parse_evidence_entries(filings_summary_text)
            if not _is_manager_form(entry.get("form", ""))
        ]

        latest_entry = _select_primary_entry(filing_entries)
        latest_form = _normalize_form_name(latest_entry.get("form", "")) if latest_entry else ""
        if not latest_form:
            fallback_form = _normalize_form_name(_clean_text(getattr(row, "LatestForm", "")))
            if fallback_form and not _is_manager_form(fallback_form):
                latest_form = fallback_form
        latest_filed_ts = latest_entry.get("date_value") if latest_entry else None
        filed_at_fallback = _parse_date_value(getattr(row, "FiledAt", None))
        if latest_filed_ts is None:
            latest_filed_ts = filed_at_fallback
        latest_filed_at = _format_date(latest_filed_ts)
        latest_filed_age_days = _compute_days_ago(latest_filed_ts, now_utc)

        catalyst_entries = _parse_evidence_entries(catalyst_text)
        catalyst_summary = _summarize_category(catalyst_entries, now_utc)
        catalyst_summary = _enforce_form_url_guard(identifier, "Catalyst", catalyst_summary, catalyst_entries, now_utc, progress_fn)
        if catalyst_summary.get("status") != "Pass":
            _emit("WARN", f"Dropped: Catalyst evidence missing ({identifier})", progress_fn)
            continue
        catalyst_primary_url = catalyst_summary.get("primary_url") or _clean_text(getattr(row, "CatalystPrimaryLink", ""))
        if catalyst_primary_url:
            catalyst_summary["primary_url"] = catalyst_primary_url
        catalyst_field = catalyst_summary.get("primary_text") or catalyst_text
        catalyst_evidence = _format_evidence_entries(catalyst_entries) or catalyst_text
        catalyst_type = _infer_catalyst_type(
            catalyst_entries,
            catalyst_summary.get("primary_form", ""),
            catalyst_text,
        ) or _clean_text(getattr(row, "CatalystType", ""))

        dilution_entries = _parse_evidence_entries(dilution_text)
        dilution_summary = _summarize_category(dilution_entries, now_utc)
        dilution_summary = _enforce_form_url_guard(identifier, "Dilution", dilution_summary, dilution_entries, now_utc, progress_fn)
        dilution_primary_url = dilution_summary.get("primary_url") or _first_url(_clean_text(getattr(row, "DilutionLinks", "")))
        if dilution_primary_url:
            dilution_summary["primary_url"] = dilution_primary_url
        dilution_forms_hit = list(
            dict.fromkeys(
                [
                    entry.get("form", "")
                    for entry in dilution_entries
                    if entry.get("form")
                    and any(entry.get("form", "").startswith(prefix) for prefix in _DILUTION_FORM_PREFIXES)
                ]
            )
        )
        dilution_keywords_raw: list[str] = []
        has_keyword_filing = False
        for entry in dilution_entries:
            lower_text = entry.get("lower", "")
            form_name = entry.get("form", "") or ""
            if form_name.startswith(("8-K", "6-K")) and any(keyword in lower_text for keyword in _DILUTION_KEYWORDS):
                has_keyword_filing = True
            for keyword in _DILUTION_KEYWORDS:
                if keyword in lower_text:
                    dilution_keywords_raw.append(keyword)
        dilution_keywords_hit = list(dict.fromkeys(dilution_keywords_raw))
        has_offering_form = any(
            entry.get("form", "").startswith(prefix)
            for entry in dilution_entries
            for prefix in _DILUTION_OFFERING_PREFIXES
        )
        has_s8_only = bool(dilution_forms_hit) and all(form.startswith("S-8") for form in dilution_forms_hit)
        dilution_status = "Missing"
        dilution_flag = False
        if has_offering_form or has_keyword_filing:
            dilution_status = "Pass (Offering)"
            dilution_flag = True
        elif has_s8_only:
            dilution_status = "Overhang (S-8)"
            dilution_flag = True
        elif dilution_entries:
            dilution_status = "None"
            dilution_flag = False

        if not dilution_flag:
            _emit("WARN", f"Dropped: Dilution filing missing ({identifier})", progress_fn)
            continue
        dilution_evidence = _format_evidence_entries(dilution_entries) or dilution_text
        dilution_field = dilution_summary.get("primary_text") or dilution_text

        governance_entries = _parse_evidence_entries(governance_text)
        governance_summary = _summarize_category(governance_entries, now_utc)
        governance_summary = _enforce_form_url_guard(
            identifier, "Governance", governance_summary, governance_entries, now_utc, progress_fn
        )
        governance_primary_url = governance_summary.get("primary_url") or _first_url(governance_text)
        if governance_primary_url:
            governance_summary["primary_url"] = governance_primary_url
        invalid_governance_form = False
        if (
            governance_summary.get("primary_form")
            and governance_summary["primary_form"] not in _GOVERNANCE_VALID_FORMS
        ):
            invalid_governance_form = True
            governance_summary["status"] = "Missing"
            governance_summary["primary_form"] = ""
            governance_summary["primary_date"] = ""
            governance_summary["primary_days_ago"] = ""
            governance_summary["primary_url"] = ""
            governance_summary["primary_text"] = ""
        governance_evidence = _format_evidence_entries(governance_entries) or governance_text
        governance_field = governance_summary.get("primary_text") or governance_text
        governance_notes = governance_summary.get("primary_raw") or governance_text
        gov_lower = governance_text.lower()
        has_going_concern = "going concern" in gov_lower
        has_material_weakness = "material weakness" in gov_lower
        auditor_text = _clean_text(getattr(row, "Auditor", ""))

        risk_flag = ""
        dilution_lower = dilution_text.lower()
        if "watch" in dilution_lower:
            risk_flag = "Dilution Watch"
        elif "watch" in gov_lower:
            risk_flag = "Governance Watch"

        insider_text = _clean_text(getattr(row, "Insider", ""))
        ownership_text = _clean_text(getattr(row, "Ownership", ""))
        insider_links = _extract_urls_list(insider_text)
        ownership_links = _extract_urls_list(ownership_text)
        insider_links = [url for url in insider_links if _INSIDER_LINK_RE.search(url)]
        ownership_links = [url for url in ownership_links if _OWNERSHIP_LINK_RE.search(url)]
        insider_buy_count_val = _to_float(getattr(row, "InsiderBuyCount", None))
        insider_buy_count = int(insider_buy_count_val) if insider_buy_count_val is not None else 0
        has_insider_cluster_raw = getattr(row, "HasInsiderCluster", False)
        if isinstance(has_insider_cluster_raw, str):
            has_insider_cluster = has_insider_cluster_raw.strip().lower() in {"true", "1", "yes"}
        else:
            has_insider_cluster = bool(has_insider_cluster_raw)

        materiality_text = _clean_text(getattr(row, "Materiality", ""))
        subscores_text = _clean_text(getattr(row, "SubscoresEvidenced", ""))
        subscores_count = int(getattr(row, "SubscoresEvidencedCount", 0) or 0)
        status_text = _clean_text(getattr(row, "Status", ""))
        status_text = status_text.replace("—", "-").replace("–", "-")

        insider_status_label = _normalize_checklist_status(getattr(row, "InsiderStatus", ""))
        ownership_status_label = _normalize_checklist_status(getattr(row, "OwnershipStatus", ""))

        checklist_statuses = {
            "Catalyst": _normalize_checklist_status(catalyst_summary.get("status", "")),
            "Dilution": _normalize_checklist_status(dilution_status),
            "Runway": _normalize_checklist_status(runway_status),
            "Governance": _normalize_checklist_status(governance_summary.get("status", "")),
            "Insider": insider_status_label,
            "Ownership": ownership_status_label,
        }

        passed_count = sum(1 for status in checklist_statuses.values() if status == "Pass")
        total_checks = len(checklist_statuses)
        checklist_passed_value = f"{passed_count}/{total_checks}"

        evidence_text = f"{catalyst_evidence} {dilution_evidence}".strip()
        tier1_type, tier1_trigger = detect_tier1(evidence_text)

        mandatory_pass = (
            checklist_statuses["Catalyst"] == "Pass"
            and checklist_statuses["Dilution"] == "Pass"
            and checklist_statuses["Runway"] == "Pass"
        )
        if not mandatory_pass:
            status_text = "TBD - exclude"

        catalyst_primary_form = catalyst_summary.get("primary_form", "")
        catalyst_primary_date = catalyst_summary.get("primary_date", "")
        catalyst_primary_days = catalyst_summary.get("primary_days_ago", "")
        catalyst_primary_url = catalyst_summary.get("primary_url", "")

        dilution_primary_form = dilution_summary.get("primary_form", "")
        dilution_primary_date = dilution_summary.get("primary_date", "")
        dilution_primary_days = dilution_summary.get("primary_days_ago", "")
        dilution_primary_url = dilution_summary.get("primary_url", "")

        governance_primary_form = governance_summary.get("primary_form", "")
        governance_primary_date = governance_summary.get("primary_date", "")
        governance_primary_days = governance_summary.get("primary_days_ago", "")
        governance_primary_url = governance_summary.get("primary_url", "")

        validation_errors: list[str] = []
        if not _is_iso_date_string(catalyst_primary_date) or not _url_matches_form(
            catalyst_primary_form, catalyst_primary_url
        ):
            validation_errors.append("CatalystPrimary")

        if dilution_primary_form:
            if not _is_iso_date_string(dilution_primary_date):
                validation_errors.append("DilutionPrimaryDate")
            if not _url_matches_form(dilution_primary_form, dilution_primary_url):
                validation_errors.append("DilutionPrimaryURL")

        if invalid_governance_form or (
            governance_primary_form and governance_primary_form not in _GOVERNANCE_VALID_FORMS
        ):
            validation_errors.append("GovernancePrimaryForm")

        if dilution_flag and dilution_status not in {"Pass (Offering)", "Overhang (S-8)"}:
            validation_errors.append("DilutionStatus")

        if runway_quarters is None and (runway_status != "Missing" or runway_source_url):
            validation_errors.append("RunwayMissing")

        if validation_errors:
            _emit(
                "WARN",
                f"Dropped: Validation failed ({identifier}) {'; '.join(validation_errors)}",
                progress_fn,
            )
            continue

        record = {
            "Ticker": ticker_value,
            "Company": company_value,
            "Sector": sector_text,
            "CIK": cik_value,
            "Price": price_val,
            "MarketCap": market_cap_val,
            "ADV20": adv_val,
            "LatestForm": latest_form,
            "LatestFiledAt": latest_filed_at,
            "LatestFiledAgeDays": latest_filed_age_days,
            "CatalystType": catalyst_type,
            "Catalyst": catalyst_field,
            "CatalystStatus": catalyst_summary.get("status", ""),
            "CatalystPrimaryForm": catalyst_primary_form,
            "CatalystPrimaryDate": catalyst_primary_date,
            "CatalystPrimaryDaysAgo": catalyst_primary_days,
            "CatalystPrimaryURL": catalyst_primary_url,
            "RunwayQuartersRaw": runway_quarters_raw,
            "RunwayQuarters": runway_quarters,
            "RunwayCashRaw": runway_cash_raw,
            "RunwayCash": runway_cash,
            "RunwayQuarterlyBurnRaw": runway_burn_raw,
            "RunwayQuarterlyBurn": runway_burn,
            "RunwayEstimate": runway_estimate,
            "RunwayNotes": runway_notes,
            "RunwayDays": runway_days_value,
            "RunwayBucket": runway_bucket_label,
            "RunwaySourceForm": runway_source_form,
            "RunwaySourceDate": runway_source_date,
            "RunwaySourceDaysAgo": runway_source_days,
            "RunwaySourceURL": runway_source_url,
            "RunwayStatus": runway_status,
            "DilutionFlag": dilution_flag,
            "Dilution": dilution_field,
            "DilutionStatus": dilution_status,
            "DilutionFormsHit": "; ".join(dilution_forms_hit),
            "DilutionKeywordsHit": "; ".join(dilution_keywords_hit),
            "DilutionPrimaryForm": dilution_primary_form,
            "DilutionPrimaryDate": dilution_primary_date,
            "DilutionPrimaryDaysAgo": dilution_primary_days,
            "DilutionPrimaryURL": dilution_primary_url,
            "DilutionEvidenceAll": dilution_evidence,
            "Governance": governance_field,
            "GovernanceStatus": governance_summary.get("status", ""),
            "GovernancePrimaryForm": governance_primary_form,
            "GovernancePrimaryDate": governance_primary_date,
            "GovernancePrimaryDaysAgo": governance_primary_days,
            "GovernancePrimaryURL": governance_primary_url,
            "GovernanceEvidenceAll": governance_evidence,
            "GovernanceNotes": governance_notes,
            "HasGoingConcern": has_going_concern,
            "HasMaterialWeakness": has_material_weakness,
            "Auditor": auditor_text,
            "Insider": insider_text,
            "InsiderStatus": insider_status_label,
            "InsiderForms345Links": "; ".join(insider_links),
            "InsiderBuyCount": insider_buy_count,
            "HasInsiderCluster": has_insider_cluster,
            "Ownership": ownership_text,
            "OwnershipStatus": ownership_status_label,
            "OwnershipLinks": "; ".join(ownership_links),
            "Materiality": materiality_text,
            "SubscoresEvidenced": subscores_text,
            "SubscoresEvidencedCount": subscores_count,
            "Status": status_text,
            "ChecklistPassed": checklist_passed_value,
            "ChecklistCatalyst": checklist_statuses["Catalyst"],
            "ChecklistDilution": checklist_statuses["Dilution"],
            "ChecklistRunway": checklist_statuses["Runway"],
            "ChecklistGovernance": checklist_statuses["Governance"],
            "ChecklistInsider": checklist_statuses["Insider"],
            "ChecklistOwnership": checklist_statuses["Ownership"],
            "RiskFlag": risk_flag,
            "Tier1Type": tier1_type,
            "Tier1Trigger": tier1_trigger,
        }

        survivors.append(record)

    output_path = os.path.join(data_dir, "validated_watchlist.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_OUTPUT_COLUMNS)
        for record in survivors:
            writer.writerow([_csv_cell_value(record.get(column)) for column in _OUTPUT_COLUMNS])

    rows_written = len(survivors)

    if status == "stopped":
        _emit(
            "INFO",
            f"build_watchlist: stop requested; wrote validated_watchlist.csv with {rows_written} rows",
            progress_fn,
        )
    else:
        _emit(
            "OK",
            f"build_watchlist: wrote validated_watchlist.csv with {rows_written} rows",
            progress_fn,
        )

    return rows_written, status
