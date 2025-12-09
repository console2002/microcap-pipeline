import os
from datetime import datetime

import pandas as pd

from app import pipeline
from app.csv_names import csv_path
from app.utils import ensure_csv


class DummyClient:
    def stats_string(self):
        return "stub_stats"


def _base_cfg(tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "FMPKey": "TEST",
        "BatchSizes": {"Profiles": 10},
        "RateLimitsPerMin": {"FMP": 9999},
        "HardGates": {
            "MinPrice": 1.0,
            "CapMin": 0,
            "CapMax": 350_000_000,
            "ADV20_Min": 40_000,
        },
        "Universe": {"Exchanges": ["NASDAQ", "NYSE", "NYSEAM"], "DropPatterns": [], "DropWordPatterns": []},
        "Paths": {"data": str(data_dir), "logs": str(logs_dir)},
        "GUI": {"SingleRunLock": False},
        "UserAgent": "test",
        "TimeoutSeconds": 1,
        "Retries": 0,
        "BackoffSeconds": [0, 0, 0],
    }


def _write_profiles_cache(cfg, df):
    path = csv_path(cfg["Paths"]["data"], "profiles")
    df.to_csv(path, index=False)
    return path


def _fake_fetch_profiles(client, cfg, tickers, progress_fn=None, stop_flag=None, include_raw=False):
    rows = [
        {
            "Ticker": "AA",
            "Exchange": "NASDAQ",
            "Industry": "Health",
            "Price": 2.0,
            "MarketCap": 150_000_000,
            "CIK": "1",
            "SecurityType": "Common Stock",
            "ADV20": 50_000,
        },
        {
            "Ticker": "BB",
            "Exchange": "NYSE",
            "Industry": "Tech",
            "Price": 3.0,
            "MarketCap": 120_000_000,
            "CIK": "2",
            "SecurityType": "Common Stock",
            "ADV20": 60_000,
        },
    ]

    gate_stats = {"exchange_security": 0, "shell": 0, "price": 0, "cap": 0, "adv": 0}
    if include_raw:
        return rows, rows, gate_stats
    return rows


def _fake_aftermarket_quotes(client, cfg, tickers, progress_fn=None, stop_flag=None):
    quotes = []
    for ticker in tickers:
        quotes.append(
            {
                "Ticker": ticker,
                "BidPrice": 10.0,
                "AskPrice": 10.5,
                "BidSize": 100,
                "AskSize": 100,
                "AfterHoursVolume": 1_000,
                "QuoteTimestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    return quotes


def _noop_append_antijoin_purge(cfg, name, df_new, key_cols, keep_days=None, date_col="Date"):
    _write_profiles_cache(cfg, df_new)
    return len(df_new)


def _dummy_event_stage(*args, **kwargs):
    return pipeline.EventStageResult(status="hard_failure")


def test_resume_rewrites_gated_universe(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path)
    runlog = tmp_path / "runlog.csv"
    errlog = tmp_path / "errorlog.csv"
    ensure_csv(runlog, ["timestamp", "module", "rows_added", "duration_ms", "note"])
    ensure_csv(errlog, ["timestamp", "module", "message"])

    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "make_client", lambda _cfg: DummyClient())
    monkeypatch.setattr(pipeline, "EdgarAdapter", lambda _cfg: None)
    monkeypatch.setattr(pipeline, "set_adapter", lambda adapter: None)
    monkeypatch.setattr(pipeline, "universe_step", lambda *args, **kwargs: pd.DataFrame({"Ticker": ["AA", "BB"]}))
    monkeypatch.setattr(pipeline, "fetch_profiles", _fake_fetch_profiles)
    monkeypatch.setattr(pipeline, "fetch_aftermarket_quotes", _fake_aftermarket_quotes)
    monkeypatch.setattr(pipeline, "append_antijoin_purge", _noop_append_antijoin_purge)
    monkeypatch.setattr(pipeline, "filings_step", lambda *args, **kwargs: (pd.DataFrame({"Ticker": ["AA"]}), {"AA"}, {}))
    monkeypatch.setattr(pipeline, "_restrict_profiles_to_core_filings", lambda df_prof, df_fil, progress_fn, eligible_tickers, drop_details: df_prof[df_prof["Ticker"].isin(df_fil["Ticker"])] )
    monkeypatch.setattr(pipeline, "parse_8k_step", _dummy_event_stage)

    pipeline.run_weekly_pipeline(
        stop_flag={"stop": False},
        progress_fn=None,
        start_stage="universe",
    )

    data_dir = tmp_path / "data"
    gated_path = data_dir / "01_universe_gated.csv"
    raw_path = data_dir / "01_universe_raw.csv"

    assert gated_path.exists()
    assert raw_path.exists()

    df_gated_first = pd.read_csv(gated_path)
    assert set(df_gated_first["Ticker"]) == {"AA", "BB"}

    os.remove(gated_path)

    pipeline.run_weekly_pipeline(
        stop_flag={"stop": False},
        progress_fn=None,
        start_stage="filings",
    )

    assert gated_path.exists(), "Expected gated universe to be rewritten on resume"
    df_gated_resumed = pd.read_csv(gated_path)
    assert set(df_gated_resumed["Ticker"]) == {"AA", "BB"}

    assert raw_path.exists(), "Raw universe should only be written in full run and remain untouched"
