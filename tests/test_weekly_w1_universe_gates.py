import logging
from datetime import datetime

import pandas as pd
import pytest

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
    raw_rows = [
        {"Ticker": "OTCX", "Exchange": "OTC", "Industry": "Tech", "Price": 2.0, "MarketCap": 100_000_000, "CIK": "1", "SecurityType": "Common Stock"},
        {"Ticker": "SHELL", "Exchange": "NASDAQ", "Industry": "Shell Companies", "Price": 5.0, "MarketCap": 100_000_000, "CIK": "2", "SecurityType": "Common Stock"},
        {"Ticker": "LOWP", "Exchange": "NASDAQ", "Industry": "Health", "Price": 0.5, "MarketCap": 100_000_000, "CIK": "3", "SecurityType": "Common Stock"},
        {"Ticker": "PREF", "Exchange": "NASDAQ", "Industry": "Health", "Price": 2.0, "MarketCap": 200_000_000, "CIK": "4", "SecurityType": "Common Stock"},
        {"Ticker": "DISC", "Exchange": "NYSE", "Industry": "Health", "Price": 3.0, "MarketCap": 320_000_000, "CIK": "5", "SecurityType": "Common Stock"},
        {"Ticker": "BIG", "Exchange": "NYSE", "Industry": "Health", "Price": 3.0, "MarketCap": 360_000_000, "CIK": "6", "SecurityType": "Common Stock"},
        {"Ticker": "ADVX", "Exchange": "NASDAQ", "Industry": "Health", "Price": 3.0, "MarketCap": 150_000_000, "CIK": "7", "SecurityType": "Common Stock"},
        {"Ticker": "NOSEC", "Exchange": "NASDAQ", "Industry": "Health", "Price": 2.5, "MarketCap": 120_000_000, "CIK": "8", "SecurityType": ""},
        {"Ticker": "NOADV", "Exchange": "NYSE", "Industry": "Health", "Price": 2.5, "MarketCap": 125_000_000, "CIK": "9", "SecurityType": "Common Stock", "ADV20": None},
    ]

    gate_stats = {"exchange_security": 1, "shell": 1, "price": 1, "cap": 1, "adv": 1}
    filtered = [row for row in raw_rows if row["Ticker"] not in {"OTCX", "SHELL", "LOWP", "BIG", "ADVX"}]

    if include_raw:
        return filtered, raw_rows, gate_stats
    return filtered


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


def test_raw_and_gated_universe_outputs(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path)
    client = DummyClient()
    runlog = tmp_path / "runlog.csv"
    errlog = tmp_path / "errorlog.csv"
    ensure_csv(runlog, ["timestamp", "module", "rows_added", "duration_ms", "note"])
    ensure_csv(errlog, ["timestamp", "module", "message"])

    monkeypatch.setattr(pipeline, "fetch_profiles", _fake_fetch_profiles)
    monkeypatch.setattr(pipeline, "fetch_aftermarket_quotes", _fake_aftermarket_quotes)
    monkeypatch.setattr(pipeline, "append_antijoin_purge", _noop_append_antijoin_purge)

    df_uni = pd.DataFrame({"Ticker": [row["Ticker"] for row in _fake_fetch_profiles(None, cfg, [], include_raw=True)[1]]})

    df_prof = pipeline.profiles_step(
        cfg=cfg,
        client=client,
        runlog=str(runlog),
        errlog=str(errlog),
        df_uni=df_uni,
        stop_flag={"stop": False},
        progress_fn=None,
    )

    raw_path = tmp_path / "data" / "01_universe_raw.csv"
    gated_path = tmp_path / "data" / "01_universe_gated.csv"

    pipeline._write_weekly_universe(cfg["Paths"]["data"], df_prof)

    df_raw = pd.read_csv(raw_path)
    df_gated = pd.read_csv(gated_path)

    assert set(df_raw["Ticker"]) == {"OTCX", "SHELL", "LOWP", "PREF", "DISC", "BIG", "ADVX", "NOSEC", "NOADV"}

    assert set(df_gated["Ticker"]) == {"PREF", "DISC", "NOSEC", "NOADV"}
    assert "CapBand" in df_gated.columns
    assert (df_gated.loc[df_gated["Ticker"] == "PREF", "CapBand"] == "preferred").all()
    assert (df_gated.loc[df_gated["Ticker"] == "DISC", "CapBand"] == "discovery").all()
    assert (df_gated.loc[df_gated["Ticker"] == "NOSEC", "CapBand"] == "preferred").all()
    assert (df_gated.loc[df_gated["Ticker"] == "NOADV", "CapBand"] == "preferred").all()
    assert (df_gated["MarketCap"] < 350_000_000).all()


def test_gate_summary_logging(tmp_path, monkeypatch, caplog):
    cfg = _base_cfg(tmp_path)
    client = DummyClient()
    runlog = tmp_path / "runlog.csv"
    errlog = tmp_path / "errorlog.csv"
    ensure_csv(runlog, ["timestamp", "module", "rows_added", "duration_ms", "note"])
    ensure_csv(errlog, ["timestamp", "module", "message"])

    monkeypatch.setattr(pipeline, "fetch_profiles", _fake_fetch_profiles)
    monkeypatch.setattr(pipeline, "fetch_aftermarket_quotes", _fake_aftermarket_quotes)
    monkeypatch.setattr(pipeline, "append_antijoin_purge", _noop_append_antijoin_purge)

    df_uni = pd.DataFrame({"Ticker": [row["Ticker"] for row in _fake_fetch_profiles(None, cfg, [], include_raw=True)[1]]})

    caplog.set_level(logging.INFO, logger="app.pipeline")

    pipeline.profiles_step(
        cfg=cfg,
        client=client,
        runlog=str(runlog),
        errlog=str(errlog),
        df_uni=df_uni,
        stop_flag={"stop": False},
        progress_fn=None,
    )

    records = [rec for rec in caplog.records if "WEEKLY_W1_UNIVERSE_GATES" in rec.getMessage()]
    assert records, "Expected universe gate summary log"
    message = records[-1].getMessage().lower()
    assert "raw=" in message and "final_gated_count" in message
    for label in ["price", "adv", "cap", "exchange", "shell"]:
        assert label in message


def test_gated_universe_written_before_filings_restriction(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path)
    runlog = tmp_path / "runlog.csv"
    errlog = tmp_path / "errorlog.csv"
    ensure_csv(runlog, ["timestamp", "module", "rows_added", "duration_ms", "note"])
    ensure_csv(errlog, ["timestamp", "module", "message"])

    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "make_client", lambda _cfg: DummyClient())
    monkeypatch.setattr(pipeline, "EdgarAdapter", lambda _cfg: None)
    monkeypatch.setattr(pipeline, "set_adapter", lambda adapter: None)

    def _filings_universe_step(cfg, adapter, runlog, errlog, stop_flag, progress_fn):
        return pd.DataFrame({"Ticker": ["HASFIL", "NOFIL"]})

    def _filings_fetch_profiles(
        client, cfg, tickers, progress_fn=None, stop_flag=None, include_raw=False
    ):
        rows = [
            {
                "Ticker": "HASFIL",
                "Exchange": "NASDAQ",
                "Industry": "Health",
                "Price": 2.0,
                "MarketCap": 150_000_000,
                "CIK": "10",
                "SecurityType": "Common Stock",
                "ADV20": 50_000,
            },
            {
                "Ticker": "NOFIL",
                "Exchange": "NASDAQ",
                "Industry": "Health",
                "Price": 2.0,
                "MarketCap": 140_000_000,
                "CIK": "11",
                "SecurityType": "Common Stock",
                "ADV20": 50_000,
            },
        ]
        gate_stats = {"exchange_security": 0, "shell": 0, "price": 0, "cap": 0, "adv": 0}
        if include_raw:
            return rows, rows, gate_stats
        return rows

    restricted_calls: dict[str, set[str]] = {}

    def _filings_restrict(df_prof, df_fil, progress_fn, eligible_tickers, drop_details):
        restricted_calls["tickers_before"] = set(df_prof["Ticker"])
        return df_prof[df_prof["Ticker"].isin(df_fil["Ticker"])]

    def _filings_step(cfg, adapter, runlog, errlog, df_prof, stop_flag, progress_fn):
        df_fil = pd.DataFrame({"Ticker": ["HASFIL"]})
        return df_fil, {"HASFIL"}, {}

    def _filings_aftermarket_quotes(client, cfg, tickers, progress_fn=None, stop_flag=None):
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

    monkeypatch.setattr(pipeline, "universe_step", _filings_universe_step)
    monkeypatch.setattr(pipeline, "fetch_profiles", _filings_fetch_profiles)
    monkeypatch.setattr(pipeline, "fetch_aftermarket_quotes", _filings_aftermarket_quotes)
    monkeypatch.setattr(pipeline, "append_antijoin_purge", _noop_append_antijoin_purge)
    monkeypatch.setattr(pipeline, "filings_step", _filings_step)
    monkeypatch.setattr(pipeline, "_restrict_profiles_to_core_filings", _filings_restrict)
    monkeypatch.setattr(
        pipeline, "parse_8k_step", lambda *args, **kwargs: pipeline.EventStageResult(status="hard_failure")
    )

    pipeline.run_weekly_pipeline(
        stop_flag={"stop": False},
        progress_fn=None,
        start_stage="universe",
    )

    gated_path = tmp_path / "data" / "01_universe_gated.csv"
    assert gated_path.exists(), "Expected gated universe CSV to be written"

    df_gated = pd.read_csv(gated_path)
    assert set(df_gated["Ticker"]) == {"HASFIL", "NOFIL"}

    assert restricted_calls.get("tickers_before") == {"HASFIL", "NOFIL"}
