import os, time, pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any

import runway_extract
from app import dr_populate
from app.build_watchlist import run as build_watchlist_run, generate_eight_k_events
from app.config import load_config, filings_max_lookback
from app.http import HttpClient
from app.utils import utc_now_iso, ensure_csv, log_line, duration_ms
from app.sec import load_sec_universe
from app.fmp import (
    fetch_profiles,
    fetch_prices,
    fetch_filings,
    fetch_aftermarket_quotes,
)
from app.fda import fetch_fda_events
from app.cache import append_antijoin_purge
from app.csv_names import csv_filename, csv_path
from app.hydrate import hydrate_candidates
from app.shortlist import build_shortlist
from app.lockfile import is_locked, create_lock, clear_lock
from app.cancel import CancelledRun
from deep_research import run as deep_research_run


@dataclass
class RunwayDropDetail:
    ticker: str
    reason: str
    forms: list[str]
    urls: list[str]
    country: str
    notes: list[str]


def _normalize_detail_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    compact = " ".join(text.split())
    if len(compact) > 160:
        compact = f"{compact[:157]}..."
    return compact


def _format_values_for_log(values, limit: int = 3) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.append(text)
    if not seen:
        return "none"
    if len(seen) <= limit:
        return ", ".join(seen)
    head = ", ".join(seen[:limit])
    return f"{head} (+{len(seen) - limit} more)"


def _make_runway_drop_detail(
    ticker: str,
    reason: str,
    info: dict[str, Any] | None,
    country: str,
    normalized_forms: set[str] | None = None,
) -> RunwayDropDetail:
    forms_raw = sorted(info.get("forms_raw", [])) if info else []
    forms_normalized = sorted(normalized_forms or [])
    forms = forms_raw or forms_normalized

    urls = list(info.get("urls", [])) if info else []

    notes_map = info.get("notes", {}) if info else {}
    notes: list[str] = []
    for col, values in notes_map.items():
        valid_values = [val for val in sorted(values) if val]
        if not valid_values:
            continue
        snippet = "; ".join(valid_values[:2])
        if len(valid_values) > 2:
            snippet = f"{snippet} (+{len(valid_values) - 2} more)"
        notes.append(f"{col}={snippet}")
    notes.sort()

    return RunwayDropDetail(
        ticker=ticker,
        reason=reason or "",
        forms=forms[:5],
        urls=urls[:5],
        country=country or "",
        notes=notes[:3],
    )


def _runway_drop_message(detail: RunwayDropDetail) -> str:
    parts: list[str] = []
    reason = detail.reason or "filtered out by runway gate"
    parts.append(reason)
    if detail.country:
        parts.append(f"country={detail.country}")

    forms_display = _format_values_for_log(detail.forms, limit=4)
    parts.append(f"forms={forms_display}")

    if detail.urls:
        url_display = _format_values_for_log(detail.urls, limit=1)
        parts.append(f"url={url_display}")

    if detail.notes:
        note_display = _format_values_for_log(detail.notes, limit=2)
        parts.append(f"notes={note_display}")

    joined = "; ".join(parts)
    return f"filings: runway drop {detail.ticker} – {joined}"


def init_logs(cfg: dict):
    runlog = os.path.join(cfg["Paths"]["logs"], "runlog.csv")
    errlog = os.path.join(cfg["Paths"]["logs"], "errorlog.csv")
    ensure_csv(runlog, ["timestamp","module","rows_added","duration_ms","note"])
    ensure_csv(errlog, ["timestamp","module","message"])
    return runlog, errlog


def make_client(cfg: dict) -> HttpClient:
    return HttpClient(
        user_agent=cfg["UserAgent"],
        timeout=cfg["TimeoutSeconds"],
        retries=cfg["Retries"],
        backoff_secs=tuple(cfg["BackoffSeconds"])
    )


def _log_step(runlog, module, rows_added, t0, note="OK"):
    log_line(runlog, [utc_now_iso(), module, rows_added, duration_ms(t0), note])


def _log_err(errlog, module, message):
    log_line(errlog, [utc_now_iso(), module, message])


def _emit(progress_fn, msg: str):
    if progress_fn:
        progress_fn(f"{utc_now_iso()} | {msg}")


US_COUNTRY_CODES = {
    "US",
    "USA",
    "UNITED STATES",
    "UNITED STATES OF AMERICA",
}

US_RUNWAY_FORM_PREFIXES = (
    "10-Q",
    "10-Q/A",
    "10-QT",
    "10-QT/A",
    "10-K",
    "10-K/A",
    "10-KT",
    "10-KT/A",
)

FPI_ANNUAL_FORM_PREFIXES = (
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
)

FPI_INTERIM_FORM_PREFIXES = (
    "6-K",
    "6-K/A",
)


def _load_cached_dataframe(cfg: dict, name: str, required_cols: list[str] | None = None) -> pd.DataFrame:
    path = csv_path(cfg["Paths"]["data"], name)
    if not os.path.exists(path):
        raise RuntimeError(f"{csv_filename(name)} missing; cannot resume at this stage")

    df = pd.read_csv(path, encoding="utf-8")
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise RuntimeError(f"{csv_filename(name)} missing required columns: {', '.join(missing)}")
    return df


def _normalize_country(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def _is_us_country(value: str) -> bool:
    if not value:
        return True
    return value in US_COUNTRY_CODES


def _profile_lookup(df_prof: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], set[str]]:
    ticker_to_country: dict[str, str] = {}
    cik_to_ticker: dict[str, str] = {}
    tickers: set[str] = set()

    if df_prof is None or df_prof.empty:
        return ticker_to_country, cik_to_ticker, tickers

    df = df_prof.copy()

    ticker_series = df.get("Ticker")
    cik_series = df.get("CIK")
    country_series = df.get("Country")

    if ticker_series is not None:
        df["Ticker_norm"] = ticker_series.fillna("").astype(str).str.upper().str.strip()
    else:
        df["Ticker_norm"] = ""

    if cik_series is not None:
        df["CIK_norm"] = cik_series.fillna("").astype(str).str.strip().str.zfill(10)
    else:
        df["CIK_norm"] = ""

    if country_series is not None:
        df["Country_norm"] = country_series.apply(_normalize_country)
    else:
        df["Country_norm"] = ""

    for row in df.itertuples(index=False):
        ticker = getattr(row, "Ticker_norm", "")
        cik = getattr(row, "CIK_norm", "")
        country = getattr(row, "Country_norm", "")

        if ticker:
            tickers.add(ticker)
            if country and ticker not in ticker_to_country:
                ticker_to_country[ticker] = country
        if ticker and cik and cik not in cik_to_ticker:
            cik_to_ticker[cik] = ticker

    return ticker_to_country, cik_to_ticker, tickers


def _normalize_filings_tickers(df_filings: pd.DataFrame, cik_to_ticker: dict[str, str]) -> pd.DataFrame:
    if df_filings is None or df_filings.empty:
        return df_filings

    df = df_filings.copy()

    if "Ticker" not in df.columns:
        df["Ticker"] = pd.Series(dtype="object")

    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.upper().str.strip()

    if "CIK" in df.columns:
        cik_series = df["CIK"].fillna("").astype(str).str.strip().str.zfill(10)
        df["CIK"] = cik_series
        if cik_to_ticker:
            missing_mask = df["Ticker"].eq("") & cik_series.ne("")
            if missing_mask.any():
                df.loc[missing_mask, "Ticker"] = (
                    cik_series.loc[missing_mask].map(cik_to_ticker).fillna("")
                )
                df["Ticker"] = df["Ticker"].fillna("").astype(str).str.upper().str.strip()

    return df


def _forms_support_runway(forms: set[str], country: str) -> tuple[bool, str]:
    if not forms:
        return False, "no filings in lookback window"

    normalized_forms = {str(f).strip().upper() for f in forms if str(f).strip()}
    if not normalized_forms:
        return False, "filings missing form codes"

    has_us_form = any(
        form.startswith(prefix) for prefix in US_RUNWAY_FORM_PREFIXES for form in normalized_forms
    )
    if has_us_form:
        return True, ""

    has_fpi_annual = any(
        form.startswith(prefix) for prefix in FPI_ANNUAL_FORM_PREFIXES for form in normalized_forms
    )
    has_fpi_interim = any(
        form.startswith(prefix) for prefix in FPI_INTERIM_FORM_PREFIXES for form in normalized_forms
    )

    if has_fpi_annual and has_fpi_interim:
        return True, ""

    if has_fpi_annual and not has_fpi_interim:
        return False, "missing 6-K alongside annual FPI filing"

    if has_fpi_interim and not has_fpi_annual:
        return False, "missing 20-F/40-F alongside 6-K"

    return False, "no qualifying core filing forms"


def _apply_runway_gate_to_filings(
    df_filings: pd.DataFrame,
    df_prof: pd.DataFrame,
    progress_fn,
    *,
    log: bool = True,
) -> tuple[pd.DataFrame, set[str], dict[str, RunwayDropDetail]]:
    if df_filings is None:
        return pd.DataFrame(), set(), {}

    ticker_to_country, cik_to_ticker, prof_tickers = _profile_lookup(df_prof)
    df_normalized = _normalize_filings_tickers(df_filings, cik_to_ticker)

    if df_normalized.empty or "Form" not in df_normalized.columns:
        if log and prof_tickers:
            _emit(progress_fn, "filings: runway gate removed all tickers (no filings available)")
        return df_normalized.iloc[0:0], set(), {}

    detail_columns = [
        col
        for col in df_normalized.columns
        if col not in {"CIK", "Ticker", "Company", "Form", "FiledAt", "URL"}
        and any(keyword in col.lower() for keyword in ("status", "reason", "error"))
    ]

    ticker_details: dict[str, dict[str, Any]] = {}
    form_map: dict[str, set[str]] = {}
    for row in df_normalized.itertuples(index=False):
        ticker = getattr(row, "Ticker", "")
        if not ticker:
            continue
        ticker_upper = str(ticker).strip().upper()
        if not ticker_upper:
            continue

        info = ticker_details.setdefault(
            ticker_upper,
            {"forms_raw": set(), "forms_normalized": set(), "urls": [], "notes": {}},
        )

        form = getattr(row, "Form", "")
        form_text = str(form).strip()
        form_upper = form_text.upper()
        if form_upper:
            info["forms_raw"].add(form_text)
            info["forms_normalized"].add(form_upper)
            form_map.setdefault(ticker_upper, set()).add(form_upper)
        else:
            # retain awareness that a filing existed but lacked a recognizable form
            info.setdefault("notes", {}).setdefault("FormStatus", set()).add("missing form code")

        url_value = getattr(row, "URL", "")
        url_text = str(url_value).strip()
        if url_text and url_text not in info["urls"]:
            info["urls"].append(url_text)

        for col in detail_columns:
            value = getattr(row, col, None)
            text = _normalize_detail_text(value)
            if text:
                info.setdefault("notes", {}).setdefault(col, set()).add(text)

    for ticker in ticker_details.keys():
        form_map.setdefault(ticker, set())

    candidate_tickers = set(prof_tickers) if prof_tickers else set(form_map.keys())

    eligible: set[str] = set()
    drop_details: dict[str, RunwayDropDetail] = {}

    def _evaluate_ticker(ticker: str) -> None:
        forms = form_map.get(ticker) or set()
        country = ticker_to_country.get(ticker, "")
        passed, reason = _forms_support_runway(forms, country)
        if passed:
            eligible.add(ticker)
            return

        info = ticker_details.setdefault(
            ticker,
            {"forms_raw": set(), "forms_normalized": set(), "urls": [], "notes": {}},
        )
        drop_details[ticker] = _make_runway_drop_detail(
            ticker,
            reason,
            info,
            country,
            normalized_forms=forms,
        )

    for ticker in candidate_tickers:
        _evaluate_ticker(ticker)

    extra_tickers = set(form_map.keys()) - candidate_tickers
    for ticker in extra_tickers:
        _evaluate_ticker(ticker)

    filtered = (
        df_normalized[df_normalized["Ticker"].isin(eligible)].copy()
        if eligible
        else df_normalized.iloc[0:0].copy()
    )

    if log:
        total_candidates = len(candidate_tickers) or len(form_map)
        if total_candidates:
            _emit(
                progress_fn,
                "filings: runway gate retained {} tickers, dropped {} lacking 10-Q/10-K or 20-F/40-F + 6-K coverage".format(
                    len(eligible),
                    len(drop_details),
                ),
            )
        elif drop_details:
            _emit(
                progress_fn,
                "filings: runway gate dropped {} tickers lacking 10-Q/10-K or 20-F/40-F + 6-K coverage".format(len(drop_details)),
            )

    return filtered, eligible, drop_details


def _restrict_profiles_to_core_filings(
    df_prof: pd.DataFrame,
    df_filings: pd.DataFrame,
    progress_fn,
    eligible_tickers: set[str] | None = None,
    drop_details: dict[str, RunwayDropDetail] | None = None,
):
    """Filter profiles to tickers that passed the core filings requirement."""

    if df_prof is None or df_prof.empty:
        return df_prof

    if eligible_tickers is None or drop_details is None:
        _, eligible_from_gate, detail_map = _apply_runway_gate_to_filings(
            df_filings, df_prof, progress_fn, log=False
        )
        if eligible_tickers is None:
            eligible_tickers = eligible_from_gate
        if drop_details is None:
            drop_details = detail_map

    eligible_tickers = eligible_tickers or set()
    drop_details = drop_details or {}

    if not eligible_tickers:
        _emit(progress_fn, "filings: no tickers have recent core filings; downstream stages will have 0 tickers")
        return df_prof.iloc[0:0]

    original_count = len(df_prof)

    df_filtered = df_prof.copy()
    ticker_series = df_filtered.get("Ticker", pd.Series(dtype="object"))
    normalized_tickers = ticker_series.fillna("").astype(str).str.upper().str.strip()
    df_filtered["Ticker"] = (
        normalized_tickers
    )
    df_filtered = df_filtered[df_filtered["Ticker"].isin(eligible_tickers)]

    dropped = original_count - len(df_filtered)
    if dropped > 0:
        _emit(
            progress_fn,
            f"filings: restricted profiles to {len(df_filtered)} tickers with recent core filings (dropped {dropped})",
        )

        dropped_tickers = sorted(
            {ticker for ticker in normalized_tickers if ticker and ticker not in eligible_tickers}
        )
        for ticker in dropped_tickers:
            detail = drop_details.get(ticker)
            if detail is None:
                detail = RunwayDropDetail(
                    ticker=ticker,
                    reason="removed by runway gate",
                    forms=[],
                    urls=[],
                    country="",
                    notes=[],
                )
            _emit(progress_fn, _runway_drop_message(detail))

    return df_filtered


def _load_cached_universe(cfg: dict) -> pd.DataFrame:
    df_prof = _load_cached_dataframe(cfg, "profiles", ["Ticker"])
    ticks = df_prof["Ticker"].dropna().unique()
    if len(ticks) == 0:
        raise RuntimeError(f"{csv_filename('profiles')} contains no tickers; cannot resume at profiles stage")
    return pd.DataFrame({"Ticker": ticks})


def _tickers_passing_adv(cfg: dict, tickers: list[str]) -> set[str]:
    """Return the subset of *tickers* whose latest ADV20 meets the configured minimum."""

    adv_min = cfg.get("HardGates", {}).get("ADV20_Min", 0) or 0
    tickers = [str(t).strip() for t in tickers if pd.notna(t) and str(t).strip()]

    if not tickers:
        return set()

    if adv_min <= 0:
        return set(tickers)

    prices_path = csv_path(cfg["Paths"]["data"], "prices")
    if not os.path.exists(prices_path):
        return set()

    prices = pd.read_csv(prices_path, encoding="utf-8")
    if prices.empty or "Ticker" not in prices.columns or "Volume" not in prices.columns or "Date" not in prices.columns:
        return set()

    prices = prices[prices["Ticker"].astype(str).isin(tickers)]
    if prices.empty:
        return set()

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"])
    if prices.empty:
        return set()

    prices = prices.sort_values(["Ticker", "Date"])
    prices["ADV20"] = prices.groupby("Ticker")["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    latest_adv = (
        prices.groupby("Ticker")
              .tail(1)[["Ticker", "ADV20"]]
    )

    latest_adv["ADV20"] = pd.to_numeric(latest_adv["ADV20"], errors="coerce").fillna(0)

    eligible = latest_adv[latest_adv["ADV20"] >= adv_min]["Ticker"].astype(str)
    return set(eligible.tolist())


def universe_step(cfg, client, runlog, errlog, stop_flag, progress_fn):
    t0 = time.time()
    _emit(progress_fn, "universe: start SEC pull")
    uni = load_sec_universe(client, cfg)
    if stop_flag.get("stop"):
        raise CancelledRun("cancel during universe")
    df_uni = pd.DataFrame(uni)
    _log_step(runlog, "SEC_universe", len(df_uni), t0, "loaded")
    _emit(progress_fn, f"universe: done {len(df_uni)} rows {client.stats_string()}")
    return df_uni


def profiles_step(cfg, client, runlog, errlog, df_uni, stop_flag, progress_fn):
    t0 = time.time()
    _emit(progress_fn, "profiles: start")
    prof_rows = fetch_profiles(
        client, cfg, df_uni["Ticker"].tolist(),
        progress_fn=progress_fn,
        stop_flag=stop_flag
    )
    if stop_flag.get("stop"):
        raise CancelledRun("cancel during profiles")

    df_prof = pd.DataFrame(prof_rows)

    if not df_prof.empty:
        df_prof = df_prof.copy()

        obj_cols = df_prof.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df_prof[col] = df_prof[col].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )
            df_prof[col] = df_prof[col].replace("", pd.NA)

        df_prof = df_prof.dropna()

        tickers = (
            df_prof.get("Ticker", pd.Series(dtype="object"))
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )
        tickers = [ticker for ticker in tickers.tolist() if ticker]

        quote_rows = fetch_aftermarket_quotes(
            client,
            cfg,
            tickers,
            progress_fn=progress_fn,
            stop_flag=stop_flag,
        )

        df_quotes = pd.DataFrame(quote_rows)

        desired_cols = [
            "Ticker",
            "BidPrice",
            "AskPrice",
            "BidSize",
            "AskSize",
            "AfterHoursVolume",
            "QuoteTimestamp",
        ]

        if not df_quotes.empty:
            for col in ["BidPrice", "AskPrice", "BidSize", "AskSize", "AfterHoursVolume"]:
                if col in df_quotes.columns:
                    df_quotes[col] = pd.to_numeric(df_quotes[col], errors="coerce")

            has_bid_ask = {"BidPrice", "AskPrice"}.issubset(df_quotes.columns)

            if has_bid_ask:
                df_quotes["Spread"] = df_quotes["AskPrice"] - df_quotes["BidPrice"]

                mid_price = (df_quotes["AskPrice"] + df_quotes["BidPrice"]) / 2.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    df_quotes["SpreadPct"] = np.where(
                        (mid_price > 0) & df_quotes["Spread"].notna(),
                        (df_quotes["Spread"] / mid_price) * 100.0,
                        np.nan,
                    )
                df_quotes.loc[~np.isfinite(df_quotes["SpreadPct"]), "SpreadPct"] = np.nan
            else:
                df_quotes["Spread"] = pd.NA
                df_quotes["SpreadPct"] = np.nan
        else:
            df_quotes = pd.DataFrame(columns=desired_cols + ["Spread", "SpreadPct"])

        for col in desired_cols:
            if col not in df_quotes.columns:
                df_quotes[col] = pd.NA

        if "Spread" not in df_quotes.columns:
            df_quotes["Spread"] = pd.NA
        if "SpreadPct" not in df_quotes.columns:
            df_quotes["SpreadPct"] = pd.NA

        df_quotes = df_quotes[desired_cols + ["Spread", "SpreadPct"]].drop_duplicates(
            subset=["Ticker"], keep="last"
        )

        df_prof = df_prof.merge(df_quotes, how="left", on="Ticker")

        hard_gates = cfg.get("HardGates", {}) if cfg else {}

        if not df_prof.empty:
            bid_series = pd.to_numeric(df_prof["BidPrice"], errors="coerce")
            ask_series = pd.to_numeric(df_prof["AskPrice"], errors="coerce")

            bad_quote_mask = (
                bid_series.isna()
                | ask_series.isna()
                | (bid_series <= 0)
                | (ask_series <= 0)
                | (ask_series <= bid_series)
            )
            dropped_bad = int(bad_quote_mask.sum())
            if dropped_bad > 0:
                df_prof = df_prof.loc[~bad_quote_mask].copy()
                _emit(
                    progress_fn,
                    f"profiles: dropped {dropped_bad} tickers with missing/bad quote data",
                )

        spread_drop_rules = hard_gates.get("SpreadDropRules")
        if spread_drop_rules and not df_prof.empty:
            bid_series = pd.to_numeric(df_prof["BidPrice"], errors="coerce")
            ask_series = pd.to_numeric(df_prof["AskPrice"], errors="coerce")
            spread_series = pd.to_numeric(df_prof["SpreadPct"], errors="coerce")
            mid_price_series = (ask_series + bid_series) / 2.0

            drop_mask = pd.Series(False, index=df_prof.index)

            for rule in spread_drop_rules:
                if not isinstance(rule, dict):
                    continue

                max_pct_raw = rule.get("MaxSpreadPercent")
                try:
                    max_pct_val = float(max_pct_raw)
                except (TypeError, ValueError):
                    continue

                min_price_raw = rule.get("MinPrice")
                max_price_raw = rule.get("MaxPrice")

                try:
                    min_price_val = float(min_price_raw) if min_price_raw is not None else float("-inf")
                except (TypeError, ValueError):
                    min_price_val = float("-inf")

                if max_price_raw is None:
                    max_price_val = float("inf")
                else:
                    try:
                        max_price_val = float(max_price_raw)
                    except (TypeError, ValueError):
                        max_price_val = float("inf")

                price_mask = (mid_price_series >= min_price_val) & (mid_price_series < max_price_val)
                rule_drop_mask = price_mask & (
                    spread_series.isna() | (spread_series >= max_pct_val)
                )
                drop_mask = drop_mask | rule_drop_mask

            dropped = int(drop_mask.sum())
            if dropped > 0:
                df_prof = df_prof.loc[~drop_mask].copy()
                _emit(
                    progress_fn,
                    f"profiles: dropped {dropped} tickers exceeding spread caps",
                )

        # ensure columns exist for downstream CSV/debugging
        for col in ["BidPrice", "AskPrice", "Spread", "SpreadPct"]:
            if col not in df_prof.columns:
                df_prof[col] = pd.NA

    rows_added = append_antijoin_purge(
        cfg, "profiles", df_prof,
        key_cols=["Ticker"],
        keep_days=None
    )
    _log_step(runlog, "profiles", rows_added, t0, "append+purge")
    _emit(progress_fn, f"profiles: done {rows_added} new rows {client.stats_string()}")

    return pd.read_csv(csv_path(cfg["Paths"]["data"], "profiles"), encoding="utf-8")


def filings_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn):
    t0 = time.time()
    _emit(progress_fn, "filings: start")
    ticks = df_prof["Ticker"].tolist()
    f_rows = fetch_filings(
        client, cfg, ticks,
        progress_fn=progress_fn,
        stop_flag=stop_flag
    )
    if stop_flag.get("stop"):
        raise CancelledRun("cancel during filings")

    df_fil = pd.DataFrame(f_rows)

    df_fil, eligible_tickers, drop_details = _apply_runway_gate_to_filings(
        df_fil, df_prof, progress_fn
    )

    if not df_fil.empty and df_prof is not None and not df_prof.empty:
        prof = df_prof.copy()

        ticker_map = {}
        if {"Ticker", "Company"}.issubset(prof.columns):
            prof_ticker = prof[["Ticker", "Company"]].copy()
            prof_ticker["Ticker"] = (
                prof_ticker["Ticker"].fillna("").astype(str).str.upper().str.strip()
            )
            prof_ticker["Company"] = (
                prof_ticker["Company"].fillna("").astype(str).str.strip()
            )
            prof_ticker = prof_ticker[
                (prof_ticker["Ticker"] != "") & (prof_ticker["Company"] != "")
            ]
            if not prof_ticker.empty:
                ticker_map = (
                    prof_ticker.drop_duplicates(subset=["Ticker"], keep="last")
                    .set_index("Ticker")["Company"].to_dict()
                )

        cik_map = {}
        if {"CIK", "Company"}.issubset(prof.columns):
            prof_cik = prof[["CIK", "Company"]].copy()
            prof_cik["CIK"] = (
                prof_cik["CIK"].fillna("").astype(str).str.zfill(10).str.strip()
            )
            prof_cik["Company"] = (
                prof_cik["Company"].fillna("").astype(str).str.strip()
            )
            prof_cik = prof_cik[
                (prof_cik["CIK"] != "") & (prof_cik["Company"] != "")
            ]
            if not prof_cik.empty:
                cik_map = (
                    prof_cik.drop_duplicates(subset=["CIK"], keep="last")
                    .set_index("CIK")["Company"].to_dict()
                )

        if "Company" not in df_fil.columns:
            df_fil["Company"] = pd.Series(dtype="object")

        if "Ticker" in df_fil.columns:
            df_fil["Ticker"] = (
                df_fil["Ticker"].fillna("").astype(str).str.upper().str.strip()
            )

        if "CIK" in df_fil.columns:
            df_fil["CIK"] = (
                df_fil["CIK"].fillna("").astype(str).str.zfill(10).str.strip()
            )

        company_series = df_fil["Company"]
        missing = company_series.isna() | company_series.astype(str).str.strip().eq("")

        if ticker_map:
            df_fil.loc[missing, "Company"] = (
                df_fil.loc[missing, "Ticker"].map(ticker_map).fillna("")
            )
            company_series = df_fil["Company"]
            missing = company_series.isna() | company_series.astype(str).str.strip().eq("")

        if cik_map:
            df_fil.loc[missing, "Company"] = (
                df_fil.loc[missing, "CIK"].map(cik_map).fillna("")
            )
            company_series = df_fil["Company"]
            missing = company_series.isna() | company_series.astype(str).str.strip().eq("")

        if "Company" in df_fil.columns:
            df_fil["Company"] = df_fil["Company"].fillna("").astype(str)

    age_dtype = "Int64"
    if "Age" not in df_fil.columns:
        df_fil["Age"] = pd.Series(dtype=age_dtype)

    if "FiledAt" in df_fil.columns:
        filed_dt = pd.to_datetime(df_fil["FiledAt"], errors="coerce", utc=True)
        today = pd.Timestamp.utcnow().normalize()
        age_series = (today - filed_dt.dt.normalize()).dt.days
        age_series = age_series.where(~filed_dt.isna())
        try:
            df_fil["Age"] = age_series.astype(age_dtype)
        except (TypeError, ValueError):
            df_fil["Age"] = age_series

    expected_cols = ["CIK","Ticker","Company","Form","FiledAt","Age","URL"]
    for col in expected_cols:
        if col not in df_fil.columns:
            df_fil[col] = pd.Series(dtype="object")

    ordered_cols = expected_cols + [c for c in df_fil.columns if c not in expected_cols]
    df_fil = df_fil[ordered_cols]

    key_cols = ["CIK"]
    url_present = False
    if "URL" in df_fil.columns:
        url_series = df_fil["URL"].fillna("").astype(str).str.strip()
        df_fil["URL"] = url_series
        url_present = url_series.ne("").any()

    if url_present:
        key_cols.append("URL")
    else:
        key_cols.extend(["Form","FiledAt","Ticker"])

    rows_added = append_antijoin_purge(
        cfg, "filings", df_fil,
        key_cols=key_cols,
        keep_days=filings_max_lookback(cfg),
        date_col="FiledAt"
    )
    _log_step(runlog, "filings", rows_added, t0, "append+purge")
    _emit(progress_fn, f"filings: done {rows_added} new rows {client.stats_string()}")

    filings_path = csv_path(cfg["Paths"]["data"], "filings")
    df_cached = pd.read_csv(filings_path, encoding="utf-8") if os.path.exists(filings_path) else pd.DataFrame()
    df_cached, eligible_tickers, drop_details = _apply_runway_gate_to_filings(
        df_cached, df_prof, progress_fn, log=False
    )
    df_cached.to_csv(filings_path, index=False, encoding="utf-8")

    return df_cached, eligible_tickers, drop_details


def fda_step(cfg, client, runlog, errlog, df_filings, stop_flag, progress_fn):
    """
    Pull FDA events only for companies that actually had recent SEC filings.
    Append+purge into fda.csv just like other caches.
    """

    if not (cfg["FDA"]["EnableDevice"] or cfg["FDA"]["EnableDrug"]):
        _emit(progress_fn, "fda: skipped (disabled in config)")
        return pd.DataFrame()

    t0 = time.time()
    _emit(progress_fn, "fda: start")

    if df_filings is None or df_filings.empty:
        _emit(progress_fn, "fda: no filings passed into stage; skipping fetch")
        return pd.DataFrame()

    _emit(progress_fn, f"fda: received {len(df_filings)} filings rows")

    if "Ticker" in df_filings.columns:
        filings_tickers = df_filings["Ticker"].dropna().astype(str).unique().tolist()
    else:
        filings_tickers = []

    eligible_tickers = _tickers_passing_adv(cfg, filings_tickers)

    _emit(
        progress_fn,
        "fda: {}/{} tickers pass ADV20 filter (min {:.0f})".format(
            len(eligible_tickers),
            len(filings_tickers),
            cfg.get("HardGates", {}).get("ADV20_Min", 0) or 0,
        ),
    )

    if eligible_tickers and "Ticker" in df_filings.columns:
        df_filings_for_fda = df_filings[df_filings["Ticker"].astype(str).isin(eligible_tickers)]
    else:
        df_filings_for_fda = df_filings.iloc[0:0]

    if df_filings_for_fda.empty:
        _emit(progress_fn, "fda: no eligible tickers after ADV filter; skipping fetch")
        fda_rows = []
    else:
        # fetch_fda_events now expects the filtered filings dataframe
        fda_rows = fetch_fda_events(
            client,
            cfg,
            df_filings_for_fda,
            progress_fn=progress_fn,
            stop_flag=stop_flag
        )

    if stop_flag.get("stop"):
        raise CancelledRun("cancel during fda")

    df_fda = pd.DataFrame(fda_rows)

    # ensure expected columns even if empty
    expected_cols = [
        "EventID",
        "ApplicantName",
        "EventType",
        "DecisionDate",
        "Product",
        "URL",
        "CIK",
        "Ticker"
    ]
    for col in expected_cols:
        if col not in df_fda.columns:
            df_fda[col] = pd.Series(dtype="object")

    rows_added = append_antijoin_purge(
        cfg,
        "fda",
        df_fda,
        key_cols=["EventID"],
        keep_days=cfg["Windows"]["DaysBack_FDA"],
        date_col="DecisionDate"
    )

    _log_step(runlog, "fda", rows_added, t0, "append+purge")
    _emit(progress_fn, f"fda: done {rows_added} new rows {client.stats_string()}")
    _emit(progress_fn, f"fda: wrote {len(df_fda)} rows to cache (including existing)")

    # read back from disk so hydrate sees a clean CSV
    return pd.read_csv(csv_path(cfg["Paths"]["data"], "fda"), encoding="utf-8")



def prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn):
    t0 = time.time()
    _emit(progress_fn, "prices: start")
    ticks = df_prof["Ticker"].tolist()
    p_rows = fetch_prices(
        client, cfg, ticks,
        progress_fn=progress_fn,
        stop_flag=stop_flag
    )
    if stop_flag.get("stop"):
        raise CancelledRun("cancel during prices")

    df_p = pd.DataFrame(p_rows)

    expected_cols = ["Date","Ticker","Open","High","Low","Close","Volume"]
    for col in expected_cols:
        if col not in df_p.columns:
            df_p[col] = pd.Series(dtype="object")

    rows_added = append_antijoin_purge(
        cfg, "prices", df_p,
        key_cols=["Ticker","Date"],
        keep_days=cfg["Windows"]["DaysBack_Prices"],
        date_col="Date"
    )
    _log_step(runlog, "prices", rows_added, t0, "append+purge")
    _emit(progress_fn, f"prices: done {rows_added} new rows {client.stats_string()}")

    return pd.read_csv(csv_path(cfg["Paths"]["data"], "prices"), encoding="utf-8")


def hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before hydrate")

    t0 = time.time()
    _emit(progress_fn, "hydrate: start")
    cands = hydrate_candidates(cfg)
    cands_path = csv_path(cfg["Paths"]["data"], "hydrated_candidates")
    cands.to_csv(cands_path, index=False, encoding="utf-8")
    _log_step(runlog, "hydrate", len(cands), t0, f"write {csv_filename('hydrated_candidates')}")
    _emit(progress_fn, f"hydrate: wrote {len(cands)} candidates")

    if stop_flag.get("stop"):
        raise CancelledRun("cancel before shortlist")

    t1 = time.time()
    _emit(progress_fn, "shortlist: start")
    short = build_shortlist(cfg, cands)
    short_path = csv_path(cfg["Paths"]["data"], "shortlist_candidates")
    short.to_csv(short_path, index=False, encoding="utf-8")
    _log_step(runlog, "shortlist", len(short), t1, f"write {csv_filename('shortlist_candidates')}")
    _emit(progress_fn, f"shortlist: wrote {len(short)} rows")


def deep_research_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before deep_research")

    data_dir = cfg["Paths"]["data"]
    short_path = csv_path(data_dir, "shortlist_candidates")
    if not os.path.exists(short_path):
        raise RuntimeError(f"{csv_filename('shortlist_candidates')} missing; run hydrate stage first or stage requires it")

    t0 = time.time()
    _emit(progress_fn, "deep_research: start")

    deep_research_run(data_dir)

    results_path = csv_path(data_dir, "deep_research_results")
    if not os.path.exists(results_path):
        raise RuntimeError(f"deep research did not create {csv_filename('deep_research_results')}")

    df_results = pd.read_csv(results_path, encoding="utf-8")
    row_count = len(df_results)

    _log_step(runlog, "deep_research", row_count, t0, f"write {csv_filename('deep_research_results')}")
    _emit(progress_fn, f"deep_research: wrote {row_count} rows")

    return results_path


def parse_q10_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before parse_q10")

    data_dir = cfg["Paths"]["data"]
    research_path = csv_path(data_dir, "deep_research_results")
    filings_path = csv_path(data_dir, "filings")

    if not os.path.exists(research_path):
        raise RuntimeError(f"{csv_filename('deep_research_results')} missing; run deep_research stage first or stage requires it")

    if not os.path.exists(filings_path):
        raise RuntimeError(f"{csv_filename('filings')} missing; run filings stage first or stage requires it")

    t0 = time.time()
    _emit(progress_fn, "parse_q10: start")

    callback = None
    if progress_fn is not None:
        callback = lambda status, message: progress_fn(
            f"parse_q10 [{status}] {message}"
        )

    if callback:
        runway_extract.set_progress_callback(callback)

    try:
        runway_extract.run(data_dir=data_dir, stop_flag=stop_flag)
    finally:
        if callback:
            runway_extract.set_progress_callback(None)

    runway_path = csv_path(data_dir, "runway_extract_results")
    row_count = 0
    if os.path.exists(runway_path):
        try:
            df_runway = pd.read_csv(runway_path, encoding="utf-8")
            row_count = len(df_runway)
        except Exception as exc:
            _log_err(errlog, "parse_q10", f"failed to read output CSV: {exc}")

    _log_step(runlog, "parse_q10", row_count, t0, f"write {csv_filename('runway_extract_results')}")
    _emit(progress_fn, f"parse_q10: wrote {row_count} rows")


def parse_8k_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before parse_8k")

    data_dir = cfg["Paths"]["data"]
    filings_path = csv_path(data_dir, "filings")

    if not os.path.exists(filings_path):
        raise RuntimeError(f"{csv_filename('filings')} missing; run filings stage first or stage requires it")

    t0 = time.time()
    _emit(progress_fn, "eight_k: start")

    callback = None
    if progress_fn is not None:
        def adapter(message: str) -> None:
            detail = message.split(" ", 1)[1] if " " in message else message
            _emit(progress_fn, detail)

        callback = adapter

    events_df, _ = generate_eight_k_events(data_dir=data_dir, progress_fn=callback)
    row_count = len(events_df.index)

    if stop_flag.get("stop"):
        raise CancelledRun("cancel during parse_8k")

    _log_step(runlog, "parse_8k", row_count, t0, f"write {csv_filename('eight_k_events')}")
    _emit(progress_fn, f"eight_k: complete – wrote {row_count} rows")


def dr_populate_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before dr_populate")

    data_dir = cfg["Paths"]["data"]
    runway_path = csv_path(data_dir, "runway_extract_results")
    filings_path = csv_path(data_dir, "filings")

    if not os.path.exists(runway_path):
        raise RuntimeError(f"{csv_filename('runway_extract_results')} missing; run parse_q10 stage first or stage requires it")

    if not os.path.exists(filings_path):
        raise RuntimeError(f"{csv_filename('filings')} missing; run filings stage first or stage requires it")

    t0 = time.time()
    _emit(progress_fn, "dr_populate: start")

    callback = None
    if progress_fn is not None:
        callback = lambda status, message: progress_fn(
            f"dr_populate [{status}] {message}"
        )

    dr_populate.run(data_dir=data_dir, progress_callback=callback)

    full_path = csv_path(data_dir, "dr_populate_results")
    row_count = 0
    if os.path.exists(full_path):
        try:
            df_full = pd.read_csv(full_path, encoding="utf-8")
            row_count = len(df_full)
        except Exception as exc:
            _log_err(errlog, "dr_populate", f"failed to read output CSV: {exc}")

    _log_step(runlog, "dr_populate", row_count, t0, f"write {csv_filename('dr_populate_results')}")
    _emit(progress_fn, f"dr_populate: wrote {row_count} rows")


def build_watchlist_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before build_watchlist")

    data_dir = cfg["Paths"]["data"]
    t0 = time.time()
    _emit(progress_fn, "build_watchlist: start")

    rows_written, status = build_watchlist_run(
        data_dir=data_dir,
        progress_fn=progress_fn,
        stop_flag=stop_flag,
    )

    if status == "missing_source":
        _emit(progress_fn, f"build_watchlist: {csv_filename('dr_populate_results')} missing")
        _log_err(errlog, "build_watchlist", f"{csv_filename('dr_populate_results')} not found")
        return 0

    if status == "stopped":
        raise CancelledRun("cancel during build_watchlist")

    _log_step(runlog, "build_watchlist", rows_written, t0, "write validated_watchlist")
    return rows_written


def run_weekly_pipeline(
    stop_flag=None,
    progress_fn=None,
    start_stage: str = "universe",
    skip_fda: bool = False,
):
    """Run the weekly pipeline starting from the requested stage."""
    if stop_flag is None:
        stop_flag = {"stop": False}

    cfg = load_config()
    runlog, errlog = init_logs(cfg)

    if cfg["GUI"]["SingleRunLock"] and is_locked(cfg):
        _log_err(errlog, "run_weekly", "locked (already running)")
        _emit(progress_fn, "run_weekly: locked – already running")
        return

    create_lock(cfg, "weekly")
    client = make_client(cfg)

    stages = [
        "universe",
        "profiles",
        "filings",
        "prices",
        "fda",
        "hydrate",
        "deep_research",
        "parse_q10",
        "parse_8k",
        "dr_populate",
        "build_watchlist",
    ]
    if start_stage not in stages:
        raise ValueError(f"Unknown weekly start_stage '{start_stage}'")

    try:
        _emit(progress_fn, f"run_weekly: start (from {start_stage})")

        df_uni = None
        df_prof = None
        df_fil = None

        start_idx = stages.index(start_stage)

        if start_idx <= stages.index("universe"):
            df_uni = universe_step(cfg, client, runlog, errlog, stop_flag, progress_fn)
        else:
            _emit(progress_fn, f"universe: skipped (starting at {start_stage})")

        if start_idx <= stages.index("profiles"):
            if df_uni is None:
                df_uni = _load_cached_universe(cfg)
                _emit(progress_fn, f"profiles: using cached tickers from {csv_filename('profiles')}")
            df_prof = profiles_step(cfg, client, runlog, errlog, df_uni, stop_flag, progress_fn)
        else:
            df_prof = _load_cached_dataframe(cfg, "profiles")
            _emit(progress_fn, f"profiles: skipped (loaded cached {csv_filename('profiles')})")

        eligible_tickers: set[str] | None = None
        drop_details: dict[str, RunwayDropDetail] | None = None

        if start_idx <= stages.index("filings"):
            if df_prof is None:
                df_prof = _load_cached_dataframe(cfg, "profiles")
                _emit(progress_fn, f"filings: using cached {csv_filename('profiles')}")
            df_fil, eligible_tickers, drop_details = filings_step(
                cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn
            )
        else:
            df_fil = _load_cached_dataframe(cfg, "filings")
            df_fil, eligible_tickers, drop_details = _apply_runway_gate_to_filings(
                df_fil, df_prof, progress_fn
            )
            _emit(progress_fn, f"filings: skipped (loaded cached {csv_filename('filings')})")

        if df_fil is not None and df_prof is not None:
            df_prof = _restrict_profiles_to_core_filings(
                df_prof, df_fil, progress_fn, eligible_tickers, drop_details
            )

        if start_idx <= stages.index("prices"):
            if df_prof is None:
                df_prof = _load_cached_dataframe(cfg, "profiles")
                _emit(progress_fn, f"prices: using cached {csv_filename('profiles')}")
            _ = prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)
        else:
            _emit(progress_fn, "prices: skipped (starting later stage)")

        if start_idx <= stages.index("fda"):
            if skip_fda:
                _emit(progress_fn, "fda: skipped (option selected)")
            else:
                if df_fil is None:
                    df_fil = _load_cached_dataframe(cfg, "filings")
                    _emit(progress_fn, f"fda: using cached {csv_filename('filings')}")
                _ = fda_step(cfg, client, runlog, errlog, df_fil, stop_flag, progress_fn)
        else:
            _emit(progress_fn, "fda: skipped (starting later stage)")

        if start_idx <= stages.index("hydrate"):
            hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            short_path = csv_path(cfg["Paths"]["data"], "shortlist_candidates")
            if not os.path.exists(short_path):
                raise RuntimeError(f"{csv_filename('shortlist_candidates')} missing; run hydrate stage first or stage requires it")
            _emit(progress_fn, "hydrate: skipped (starting later stage)")
            _emit(progress_fn, "shortlist: skipped (starting later stage)")

        if start_idx <= stages.index("deep_research"):
            deep_research_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            results_path = csv_path(cfg["Paths"]["data"], "deep_research_results")
            if not os.path.exists(results_path):
                raise RuntimeError(f"{csv_filename('deep_research_results')} missing; run deep_research stage first or stage requires it")

        if start_idx <= stages.index("parse_q10"):
            parse_q10_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            runway_path = csv_path(cfg["Paths"]["data"], "runway_extract_results")
            if not os.path.exists(runway_path):
                raise RuntimeError(f"{csv_filename('runway_extract_results')} missing; run parse_q10 stage first or stage requires it")

        if start_idx <= stages.index("parse_8k"):
            parse_8k_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            events_path = csv_path(cfg["Paths"]["data"], "eight_k_events")
            if not os.path.exists(events_path):
                raise RuntimeError(f"{csv_filename('eight_k_events')} missing; run parse_8k stage first or stage requires it")

        if start_idx <= stages.index("dr_populate"):
            dr_populate_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            full_path = csv_path(cfg["Paths"]["data"], "dr_populate_results")
            if not os.path.exists(full_path):
                raise RuntimeError(f"{csv_filename('dr_populate_results')} missing; run dr_populate stage first or stage requires it")

        if start_idx <= stages.index("build_watchlist"):
            build_watchlist_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            validated_path = csv_path(cfg["Paths"]["data"], "validated_watchlist")
            if not os.path.exists(validated_path):
                raise RuntimeError(f"{csv_filename('validated_watchlist')} missing; run build_watchlist stage first or stage requires it")

        _emit(progress_fn, "run_weekly: complete")

    except CancelledRun as e:
        _emit(progress_fn, f"run_weekly: cancelled ({e})")

    except Exception as e:
        _log_err(errlog, "run_weekly", str(e))
        _emit(progress_fn, f"run_weekly: ERROR {e}")
        raise

    finally:
        clear_lock(cfg)



def run_daily_pipeline(stop_flag=None, progress_fn=None, start_stage: str = "prices"):
    if stop_flag is None:
        stop_flag = {"stop": False}

    cfg = load_config()
    runlog, errlog = init_logs(cfg)

    if cfg["GUI"]["SingleRunLock"] and is_locked(cfg):
        _log_err(errlog, "run_daily", "locked (already running)")
        _emit(progress_fn, "run_daily: locked – already running")
        return

    create_lock(cfg, "daily")
    client = make_client(cfg)

    stages = ["prices", "hydrate"]
    if start_stage not in stages:
        raise ValueError(f"Unknown daily start_stage '{start_stage}'")

    try:
        _emit(progress_fn, f"run_daily: start (from {start_stage})")

        start_idx = stages.index(start_stage)

        if start_idx <= stages.index("prices"):
            prof_path = csv_path(cfg["Paths"]["data"], "profiles")
            if not os.path.exists(prof_path):
                raise RuntimeError(f"{csv_filename('profiles')} missing; run weekly first or stage requires it")

            df_prof = pd.read_csv(prof_path, encoding="utf-8")
            df_fil = _load_cached_dataframe(cfg, "filings")
            df_fil, eligible_tickers, drop_details = _apply_runway_gate_to_filings(
                df_fil, df_prof, progress_fn
            )
            df_prof = _restrict_profiles_to_core_filings(
                df_prof, df_fil, progress_fn, eligible_tickers, drop_details
            )
            _ = prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)
        else:
            _emit(progress_fn, "prices: skipped (starting at hydrate stage)")

        hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn)

        _emit(progress_fn, "run_daily: complete")

    except CancelledRun as e:
        _emit(progress_fn, f"run_daily: cancelled ({e})")

    except Exception as e:
        _log_err(errlog, "run_daily", str(e))
        _emit(progress_fn, f"run_daily: ERROR {e}")
        raise

    finally:
        clear_lock(cfg)
