import os, time, pandas as pd
from app.config import load_config
from app.http import HttpClient
from app.utils import utc_now_iso, ensure_csv, log_line, duration_ms
from app.sec import load_sec_universe
from app.fmp import fetch_profiles, fetch_prices, fetch_filings
from app.fda import fetch_fda_events
from app.cache import append_antijoin_purge
from app.hydrate import hydrate_candidates
from app.shortlist import build_shortlist
from app.lockfile import is_locked, create_lock, clear_lock
from app.cancel import CancelledRun
from deep_research import run as deep_research_run


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


def _load_cached_dataframe(cfg: dict, name: str, required_cols: list[str] | None = None) -> pd.DataFrame:
    path = os.path.join(cfg["Paths"]["data"], f"{name}.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"{name}.csv missing; cannot resume at this stage")

    df = pd.read_csv(path, encoding="utf-8")
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise RuntimeError(f"{name}.csv missing required columns: {', '.join(missing)}")
    return df


def _load_cached_universe(cfg: dict) -> pd.DataFrame:
    df_prof = _load_cached_dataframe(cfg, "profiles", ["Ticker"])
    ticks = df_prof["Ticker"].dropna().unique()
    if len(ticks) == 0:
        raise RuntimeError("profiles.csv contains no tickers; cannot resume at profiles stage")
    return pd.DataFrame({"Ticker": ticks})


def _tickers_passing_adv(cfg: dict, tickers: list[str]) -> set[str]:
    """Return the subset of *tickers* whose latest ADV20 meets the configured minimum."""

    adv_min = cfg.get("HardGates", {}).get("ADV20_Min", 0) or 0
    tickers = [str(t).strip() for t in tickers if pd.notna(t) and str(t).strip()]

    if not tickers:
        return set()

    if adv_min <= 0:
        return set(tickers)

    prices_path = os.path.join(cfg["Paths"]["data"], "prices.csv")
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

    rows_added = append_antijoin_purge(
        cfg, "profiles", df_prof,
        key_cols=["Ticker"],
        keep_days=None
    )
    _log_step(runlog, "profiles", rows_added, t0, "append+purge")
    _emit(progress_fn, f"profiles: done {rows_added} new rows {client.stats_string()}")

    return pd.read_csv(os.path.join(cfg["Paths"]["data"], "profiles.csv"), encoding="utf-8")


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

    expected_cols = ["CIK","Ticker","Company","Form","FiledAt","URL"]
    for col in expected_cols:
        if col not in df_fil.columns:
            df_fil[col] = pd.Series(dtype="object")

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
        keep_days=cfg["Windows"]["DaysBack_Filings"],
        date_col="FiledAt"
    )
    _log_step(runlog, "filings", rows_added, t0, "append+purge")
    _emit(progress_fn, f"filings: done {rows_added} new rows {client.stats_string()}")

    return pd.read_csv(os.path.join(cfg["Paths"]["data"], "filings.csv"), encoding="utf-8")


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
    return pd.read_csv(os.path.join(cfg["Paths"]["data"], "fda.csv"), encoding="utf-8")



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

    return pd.read_csv(os.path.join(cfg["Paths"]["data"], "prices.csv"), encoding="utf-8")


def hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before hydrate")

    t0 = time.time()
    _emit(progress_fn, "hydrate: start")
    cands = hydrate_candidates(cfg)
    cands_path = os.path.join(cfg["Paths"]["data"], "candidates.csv")
    cands.to_csv(cands_path, index=False, encoding="utf-8")
    _log_step(runlog, "hydrate", len(cands), t0, "write candidates")
    _emit(progress_fn, f"hydrate: wrote {len(cands)} candidates")

    if stop_flag.get("stop"):
        raise CancelledRun("cancel before shortlist")

    t1 = time.time()
    _emit(progress_fn, "shortlist: start")
    short = build_shortlist(cfg, cands)
    short_path = os.path.join(cfg["Paths"]["data"], "shortlist.csv")
    short.to_csv(short_path, index=False, encoding="utf-8")
    _log_step(runlog, "shortlist", len(short), t1, "write shortlist")
    _emit(progress_fn, f"shortlist: wrote {len(short)} rows")


def deep_research_step(cfg, runlog, errlog, stop_flag, progress_fn):
    if stop_flag.get("stop"):
        raise CancelledRun("cancel before deep_research")

    data_dir = cfg["Paths"]["data"]
    short_path = os.path.join(data_dir, "shortlist.csv")
    if not os.path.exists(short_path):
        raise RuntimeError("shortlist.csv missing; run hydrate stage first or stage requires it")

    t0 = time.time()
    _emit(progress_fn, "deep_research: start")

    deep_research_run(data_dir)

    results_path = os.path.join(data_dir, "research_results.csv")
    if not os.path.exists(results_path):
        raise RuntimeError("deep research did not create research_results.csv")

    df_results = pd.read_csv(results_path, encoding="utf-8")
    row_count = len(df_results)

    _log_step(runlog, "deep_research", row_count, t0, "write research_results")
    _emit(progress_fn, f"deep_research: wrote {row_count} rows")

    return results_path


def run_weekly_pipeline(stop_flag=None, progress_fn=None, start_stage: str = "universe"):
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

    stages = ["universe", "profiles", "filings", "prices", "fda", "hydrate", "deep_research"]
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
                _emit(progress_fn, "profiles: using cached tickers from profiles.csv")
            df_prof = profiles_step(cfg, client, runlog, errlog, df_uni, stop_flag, progress_fn)
        else:
            df_prof = _load_cached_dataframe(cfg, "profiles")
            _emit(progress_fn, "profiles: skipped (loaded cached profiles.csv)")

        if start_idx <= stages.index("filings"):
            if df_prof is None:
                df_prof = _load_cached_dataframe(cfg, "profiles")
                _emit(progress_fn, "filings: using cached profiles.csv")
            df_fil = filings_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)
        else:
            df_fil = _load_cached_dataframe(cfg, "filings")
            _emit(progress_fn, "filings: skipped (loaded cached filings.csv)")

        if start_idx <= stages.index("prices"):
            if df_prof is None:
                df_prof = _load_cached_dataframe(cfg, "profiles")
                _emit(progress_fn, "prices: using cached profiles.csv")
            _ = prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)
        else:
            _emit(progress_fn, "prices: skipped (starting later stage)")

        if start_idx <= stages.index("fda"):
            if df_fil is None:
                df_fil = _load_cached_dataframe(cfg, "filings")
                _emit(progress_fn, "fda: using cached filings.csv")
            _ = fda_step(cfg, client, runlog, errlog, df_fil, stop_flag, progress_fn)
        else:
            _emit(progress_fn, "fda: skipped (starting later stage)")

        if start_idx <= stages.index("hydrate"):
            hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn)
        else:
            short_path = os.path.join(cfg["Paths"]["data"], "shortlist.csv")
            if not os.path.exists(short_path):
                raise RuntimeError("shortlist.csv missing; run hydrate stage first or stage requires it")
            _emit(progress_fn, "hydrate: skipped (starting later stage)")
            _emit(progress_fn, "shortlist: skipped (starting later stage)")

        if start_idx <= stages.index("deep_research"):
            deep_research_step(cfg, runlog, errlog, stop_flag, progress_fn)

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
            prof_path = os.path.join(cfg["Paths"]["data"], "profiles.csv")
            if not os.path.exists(prof_path):
                raise RuntimeError("profiles.csv missing; run weekly first or stage requires it")

            df_prof = pd.read_csv(prof_path, encoding="utf-8")
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
