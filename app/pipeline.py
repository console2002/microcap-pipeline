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

    expected_cols = ["CIK","Ticker","Company","Form","FiledAt","Title","URL","Accession"]
    for col in expected_cols:
        if col not in df_fil.columns:
            df_fil[col] = pd.Series(dtype="object")

    key_cols = ["CIK"]
    if "Accession" in df_fil.columns:
        key_cols.append("Accession")
    elif "URL" in df_fil.columns:
        key_cols.append("URL")

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

    # fetch_fda_events now expects the filings dataframe
    fda_rows = fetch_fda_events(
        client,
        cfg,
        df_filings,
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


def run_weekly_pipeline(stop_flag=None, progress_fn=None):
    """
    stop_flag is {"stop": bool}, progress_fn is callable(str).
    """
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

    try:
        _emit(progress_fn, "run_weekly: start")

        df_uni = universe_step(cfg, client, runlog, errlog, stop_flag, progress_fn)
        df_prof = profiles_step(cfg, client, runlog, errlog, df_uni, stop_flag, progress_fn)
        df_fil = filings_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)

        # FDA now uses df_fil (companies with actual filings)
        _ = fda_step(cfg, client, runlog, errlog, df_fil, stop_flag, progress_fn)

        _ = prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)

        hydrate_and_shortlist_step(cfg, runlog, errlog, stop_flag, progress_fn)

        _emit(progress_fn, "run_weekly: complete")

    except CancelledRun as e:
        _emit(progress_fn, f"run_weekly: cancelled ({e})")

    except Exception as e:
        _log_err(errlog, "run_weekly", str(e))
        _emit(progress_fn, f"run_weekly: ERROR {e}")
        raise

    finally:
        clear_lock(cfg)



def run_daily_pipeline(stop_flag=None, progress_fn=None):
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

    try:
        _emit(progress_fn, "run_daily: start")

        prof_path = os.path.join(cfg["Paths"]["data"], "profiles.csv")
        if not os.path.exists(prof_path):
            raise RuntimeError("profiles.csv missing; run weekly first")

        df_prof = pd.read_csv(prof_path, encoding="utf-8")

        _ = prices_step(cfg, client, runlog, errlog, df_prof, stop_flag, progress_fn)
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
