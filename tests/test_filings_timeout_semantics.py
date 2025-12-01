import logging
import time

import pytest

from app.edgar_adapter import EdgarAdapter


def _make_adapter(timeout_value):
    cfg = {"Workers": {"Edgar": {"Workers": 2}}, "Edgar": {}}
    if timeout_value is not None:
        cfg["Workers"]["Edgar"]["TimeoutSeconds"] = timeout_value
    return EdgarAdapter(cfg)


def _fake_fetch(delay_seconds=0.0, delay_map=None):
    def _fn(ticker_norm, **_kwargs):
        delay = delay_seconds
        if delay_map and ticker_norm in delay_map:
            delay = delay_map[ticker_norm]
        if delay:
            time.sleep(delay)
        batch = [
            {
                "Ticker": ticker_norm,
                "Form": "10-K",
                "FiledAt": "2024-01-01",
                "URL": f"url-{ticker_norm}",
                "CIK": "0000000000",
            }
        ]
        stats = {
            "ticker": ticker_norm,
            "raw_count": 1,
            "kept_count": 1,
            "duration_ms": 0,
            "fetch_ms": 0,
            "cik": "0000000000",
        }
        return batch, stats

    return _fn


def test_no_timeout_runs_to_completion(monkeypatch, caplog):
    adapter = _make_adapter(timeout_value=0)

    monkeypatch.setattr(adapter, "_filing_fetch_workers", lambda: 2)
    monkeypatch.setattr(adapter, "_filing_fetch_timeout", lambda: None)
    monkeypatch.setattr(adapter, "_fetch_filings_for_ticker", _fake_fetch(delay_seconds=0.01))

    tickers = ["SLOW1", "SLOW2", "SLOW3"]

    with caplog.at_level(logging.INFO):
        rows = adapter.fetch_recent_filings(tickers)

    returned_tickers = {row.get("Ticker") for row in rows}
    assert returned_tickers == set(tickers)

    summary_logs = [rec for rec in caplog.records if "[edgar filings] summary" in rec.message]
    assert summary_logs and summary_logs[-1].message.endswith("timeout_pending=0")


def test_timeout_preserves_partial_results(monkeypatch, caplog):
    adapter = _make_adapter(timeout_value=1)

    monkeypatch.setattr(adapter, "_filing_fetch_workers", lambda: 2)
    monkeypatch.setattr(adapter, "_filing_fetch_timeout", lambda: 0.05)
    delays = {"FAST1": 0.0, "FAST2": 0.0, "SLOW1": 0.1, "SLOW2": 0.1, "SLOW3": 0.1}
    monkeypatch.setattr(adapter, "_fetch_filings_for_ticker", _fake_fetch(delay_map=delays))

    tickers = ["FAST1", "SLOW1", "SLOW2", "FAST2", "SLOW3"]

    with caplog.at_level(logging.INFO):
        rows = adapter.fetch_recent_filings(tickers)

    returned_tickers = {row.get("Ticker") for row in rows}

    # Only the futures that completed before the timeout should be present
    assert returned_tickers.intersection({"FAST1", "FAST2"}), "expected at least one fast ticker to complete"
    assert not {"SLOW1", "SLOW2", "SLOW3"}.issubset(returned_tickers)

    timeout_logs = [rec for rec in caplog.records if "timeout" in rec.message]
    assert timeout_logs, "expected timeout log line"

    summary_logs = [rec for rec in caplog.records if "[edgar filings] summary" in rec.message]
    assert summary_logs and "timeout_pending=" in summary_logs[-1].message

