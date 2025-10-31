"""Build a validated watchlist from deep research results and market data."""
from __future__ import annotations

import os
import re
from typing import Callable, Iterable, Optional

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
    "Catalyst",
    "CatalystStatus",
    "CatalystPrimaryForm",
    "CatalystPrimaryDate",
    "CatalystPrimaryDaysAgo",
    "CatalystPrimaryURL",
    "CatalystEvidenceAll",
    "RunwayQuarters",
    "RunwayCash",
    "RunwayQuarterlyBurn",
    "RunwayNotes",
    "RunwayStatus",
    "RunwaySourceDate",
    "RunwaySourceDaysAgo",
    "Dilution",
    "DilutionStatus",
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
    "Insider",
    "InsiderStatus",
    "Ownership",
    "OwnershipStatus",
    "Materiality",
    "SubscoresEvidenced",
    "SubscoresEvidencedCount",
    "Status",
    "ChecklistPassed",
    "ChecklistTotal",
    "ChecklistSummary",
    "RiskFlag",
]

_NEGATIVE_DILUTION_TERMS = (
    "high risk",
    "toxic",
    "at-the-market",
)

_SUBSCORE_SPLIT_RE = re.compile(r"[;\n]+")
_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_URL_RE = re.compile(r"https?://\S+")


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


def _parse_timestamp(value: object) -> Optional[pd.Timestamp]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _format_timestamp(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_days_ago(ts: Optional[pd.Timestamp], now: pd.Timestamp) -> str:
    if ts is None:
        return ""
    delta = now - ts
    if pd.isna(delta):
        return ""
    try:
        days = int(delta.days)
    except Exception:
        return ""
    return str(days)


def _parse_evidence_entries(text: str) -> list[dict]:
    entries: list[dict] = []
    if not text:
        return entries

    for segment in _SUBSCORE_SPLIT_RE.split(text):
        raw = segment.strip()
        if not raw:
            continue

        date_match = _DATE_RE.search(raw)
        date_str = date_match.group(1) if date_match else ""
        date_val: Optional[pd.Timestamp] = None
        if date_str:
            try:
                date_val = pd.to_datetime(date_str, utc=True)
            except Exception:
                date_val = None

        url_match = _URL_RE.search(raw)
        url = url_match.group(0) if url_match else ""

        if date_match:
            after = raw[date_match.end() :].lstrip(" |")
        else:
            after = raw
        primary_segment = ""
        if after:
            primary_segment = after.split("|")[0].strip()
        if not primary_segment and after:
            primary_segment = after.split()[0] if after.split() else ""
        form = ""
        if primary_segment:
            match = re.match(r"([A-Za-z0-9./-]+)", primary_segment)
            if match:
                form = match.group(1)

        entries.append(
            {
                "raw": raw,
                "date": date_str,
                "date_value": date_val,
                "form": form,
                "url": url,
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


def _summarize_category(entries: list[dict], fallback_text: str, now: pd.Timestamp) -> dict:
    summary = {
        "status": "Missing",
        "primary": "",
        "primary_form": "",
        "primary_date": "",
        "primary_days_ago": "",
        "primary_url": "",
        "evidence": "",
    }

    cleaned_fallback = _clean_text(fallback_text)

    def sort_key(entry: dict) -> tuple[int, Optional[pd.Timestamp]]:
        date_val = entry.get("date_value")
        return (0 if date_val is None else 1, date_val)

    if entries:
        ordered = sorted(entries, key=sort_key, reverse=True)
        primary = ordered[0]
        summary["status"] = "Pass"
        summary["primary"] = primary.get("raw", "")
        summary["primary_form"] = primary.get("form", "")
        summary["primary_date"] = primary.get("date", "")
        date_val = primary.get("date_value")
        if isinstance(date_val, pd.Timestamp):
            summary["primary_days_ago"] = _format_days_ago(date_val, now)
        summary["primary_url"] = primary.get("url", "")
        summary["evidence"] = "; ".join(entry.get("raw", "") for entry in ordered if entry.get("raw"))
        return summary

    if cleaned_fallback:
        summary["status"] = "Pass"
        summary["primary"] = cleaned_fallback
        summary["evidence"] = cleaned_fallback
        return summary

    return summary


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


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

        price_val = _to_float(getattr(row, "Price", None))
        market_cap_val = _to_float(getattr(row, "MarketCap", None))
        adv_val = _to_float(getattr(row, "ADV20", None))

        if price_val is None or price_val < 1:
            continue
        if market_cap_val is None or market_cap_val >= 350_000_000:
            continue
        if adv_val is None or adv_val < 40_000:
            continue

        runway_raw = getattr(row, "RunwayQuarters", "")
        if pd.isna(runway_raw):
            continue
        runway_text = str(runway_raw).strip()
        if not runway_text or runway_text.lower() == "nan":
            continue
        runway_val = _to_float(runway_text)
        if runway_val is not None and runway_val <= 0:
            continue

        catalyst_text = _clean_text(getattr(row, "Catalyst", ""))
        if not catalyst_text or catalyst_text.upper() == "TBD":
            continue

        dilution_text_raw = getattr(row, "Dilution", "")
        dilution_text = _clean_text(dilution_text_raw)
        dilution_lower = dilution_text.lower()
        if any(term in dilution_lower for term in _NEGATIVE_DILUTION_TERMS):
            continue

        governance_raw = getattr(row, "Governance", "")
        if pd.isna(governance_raw):
            continue
        governance_text = _clean_text(governance_raw)
        if governance_text.lower() == "nan":
            continue

        risk_flag = ""
        if "watch" in dilution_lower:
            risk_flag = "Dilution Watch"
        elif "watch" in governance_text.lower():
            risk_flag = "Governance Watch"

        sector_text = _clean_text(getattr(row, "Sector", ""))
        latest_form = _clean_text(getattr(row, "LatestForm", ""))
        filed_at_ts = _parse_timestamp(getattr(row, "FiledAt", None))
        latest_filed_at = _format_timestamp(filed_at_ts)
        latest_filed_age_days = _format_days_ago(filed_at_ts, now_utc)

        catalyst_entries = _parse_evidence_entries(catalyst_text)
        catalyst_summary = _summarize_category(catalyst_entries, catalyst_text, now_utc)
        catalyst_primary_url = (
            _clean_text(getattr(row, "CatalystPrimaryLink", ""))
            or catalyst_summary.get("primary_url", "")
        )

        dilution_entries = _parse_evidence_entries(dilution_text)
        dilution_summary = _summarize_category(dilution_entries, dilution_text, now_utc)
        dilution_primary_url = (
            dilution_summary.get("primary_url", "")
            or _first_url(_clean_text(getattr(row, "DilutionLinks", "")))
        )

        governance_entries = _parse_evidence_entries(governance_text)
        governance_summary = _summarize_category(governance_entries, governance_text, now_utc)

        runway_cash_text = _clean_text(getattr(row, "RunwayCash", ""))
        runway_burn_text = _clean_text(getattr(row, "RunwayQuarterlyBurn", ""))
        runway_notes = _clean_text(getattr(row, "RunwayNotes", ""))
        runway_status = "Pass" if runway_val is None or runway_val > 0 else "Missing"
        runway_source_date = latest_filed_at
        runway_source_days = latest_filed_age_days

        insider_text = _clean_text(getattr(row, "Insider", ""))
        ownership_text = _clean_text(getattr(row, "Ownership", ""))

        materiality_text = _clean_text(getattr(row, "Materiality", ""))
        subscores_text = _clean_text(getattr(row, "SubscoresEvidenced", ""))
        status_text = _clean_text(getattr(row, "Status", ""))

        checklist_items = [
            ("Catalyst", catalyst_summary.get("status", "")),
            ("Dilution", dilution_summary.get("status", "")),
            ("Runway", runway_status),
            ("Governance", governance_summary.get("status", "")),
            ("Insider", "Placeholder"),
            ("Ownership", "Placeholder"),
        ]
        checklist_summary_parts: list[str] = []
        passed_count = 0
        for name, status_value in checklist_items:
            status_norm = _clean_text(status_value).lower()
            if status_norm == "pass":
                symbol = "✅"
                passed_count += 1
            elif status_norm == "placeholder":
                symbol = "⬜"
            else:
                symbol = "❌"
            checklist_summary_parts.append(f"{name}{symbol}")

        checklist_summary = "; ".join(checklist_summary_parts)
        checklist_total = len(checklist_items)

        record = {
            "Ticker": getattr(row, "Ticker", ""),
            "Company": getattr(row, "Company", ""),
            "Sector": sector_text,
            "CIK": getattr(row, "CIK", ""),
            "Price": price_val,
            "MarketCap": market_cap_val,
            "ADV20": adv_val,
            "LatestForm": latest_form,
            "LatestFiledAt": latest_filed_at,
            "LatestFiledAgeDays": latest_filed_age_days,
            "Catalyst": catalyst_summary.get("primary", catalyst_text),
            "CatalystStatus": catalyst_summary.get("status", ""),
            "CatalystPrimaryForm": catalyst_summary.get("primary_form", ""),
            "CatalystPrimaryDate": catalyst_summary.get("primary_date", ""),
            "CatalystPrimaryDaysAgo": catalyst_summary.get("primary_days_ago", ""),
            "CatalystPrimaryURL": catalyst_primary_url,
            "CatalystEvidenceAll": catalyst_summary.get("evidence", catalyst_text),
            "RunwayQuarters": runway_text,
            "RunwayCash": runway_cash_text,
            "RunwayQuarterlyBurn": runway_burn_text,
            "RunwayNotes": runway_notes,
            "RunwayStatus": runway_status,
            "RunwaySourceDate": runway_source_date,
            "RunwaySourceDaysAgo": runway_source_days,
            "Dilution": dilution_summary.get("primary", dilution_text),
            "DilutionStatus": dilution_summary.get("status", ""),
            "DilutionPrimaryForm": dilution_summary.get("primary_form", ""),
            "DilutionPrimaryDate": dilution_summary.get("primary_date", ""),
            "DilutionPrimaryDaysAgo": dilution_summary.get("primary_days_ago", ""),
            "DilutionPrimaryURL": dilution_primary_url,
            "DilutionEvidenceAll": dilution_summary.get("evidence", dilution_text),
            "Governance": governance_summary.get("primary", governance_text),
            "GovernanceStatus": governance_summary.get("status", ""),
            "GovernancePrimaryForm": governance_summary.get("primary_form", ""),
            "GovernancePrimaryDate": governance_summary.get("primary_date", ""),
            "GovernancePrimaryDaysAgo": governance_summary.get("primary_days_ago", ""),
            "GovernancePrimaryURL": governance_summary.get("primary_url", ""),
            "GovernanceEvidenceAll": governance_summary.get("evidence", governance_text),
            "Insider": insider_text,
            "InsiderStatus": "Placeholder",
            "Ownership": ownership_text,
            "OwnershipStatus": "Placeholder",
            "Materiality": materiality_text,
            "SubscoresEvidenced": subscores_text,
            "SubscoresEvidencedCount": getattr(row, "SubscoresEvidencedCount", 0),
            "Status": status_text,
            "ChecklistPassed": passed_count,
            "ChecklistTotal": checklist_total,
            "ChecklistSummary": checklist_summary,
            "RiskFlag": risk_flag,
        }

        survivors.append(record)

    output_path = os.path.join(data_dir, "validated_watchlist.csv")
    df_out = pd.DataFrame(survivors, columns=_OUTPUT_COLUMNS)
    df_out.to_csv(output_path, index=False, encoding="utf-8")

    rows_written = len(df_out)

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
