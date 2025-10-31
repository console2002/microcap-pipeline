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
    "CIK",
    "Price",
    "MarketCap",
    "ADV20",
    "Catalyst",
    "RunwayQuarters",
    "Dilution",
    "Governance",
    "Materiality",
    "SubscoresEvidencedCount",
    "Status",
    "RiskFlag",
]

_NEGATIVE_DILUTION_TERMS = (
    "high risk",
    "toxic",
    "at-the-market",
)

_SUBSCORE_SPLIT_RE = re.compile(r"[;\n]+")


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

        catalyst_text = str(getattr(row, "Catalyst", "")).strip()
        if not catalyst_text or catalyst_text.upper() == "TBD":
            continue

        dilution_text_raw = getattr(row, "Dilution", "")
        dilution_text = str(dilution_text_raw).strip()
        dilution_lower = dilution_text.lower()
        if any(term in dilution_lower for term in _NEGATIVE_DILUTION_TERMS):
            continue

        governance_raw = getattr(row, "Governance", "")
        if pd.isna(governance_raw):
            continue
        governance_text = str(governance_raw).strip()
        if governance_text.lower() == "nan":
            continue

        risk_flag = ""
        if "watch" in dilution_lower:
            risk_flag = "Dilution Watch"
        elif "watch" in governance_text.lower():
            risk_flag = "Governance Watch"

        record = {
            "Ticker": getattr(row, "Ticker", ""),
            "Company": getattr(row, "Company", ""),
            "CIK": getattr(row, "CIK", ""),
            "Price": price_val,
            "MarketCap": market_cap_val,
            "ADV20": adv_val,
            "Catalyst": catalyst_text,
            "RunwayQuarters": runway_text,
            "Dilution": dilution_text,
            "Governance": governance_text,
            "Materiality": getattr(row, "Materiality", ""),
            "SubscoresEvidencedCount": getattr(row, "SubscoresEvidencedCount", 0),
            "Status": getattr(row, "Status", ""),
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
