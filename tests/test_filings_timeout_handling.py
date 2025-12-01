import logging
import time

import pytest

from app.edgar_adapter import EdgarAdapter, partition_completed_and_pending


def test_partition_completed_and_pending():
    all_tickers = ["AAA", "BBB", "CCC", "DDD"]
    completed = ["AAA", "CCC"]

    done, pending = partition_completed_and_pending(all_tickers, completed)

    assert done == {"AAA", "CCC"}
    assert pending == {"BBB", "DDD"}


def test_timeout_yields_partial_results(monkeypatch, caplog):
    cfg = {"Workers": {"Edgar": {"Workers": 2, "TimeoutSeconds": 0.05}}, "Edgar": {}}
    adapter = EdgarAdapter(cfg)

    monkeypatch.setattr(adapter, "_filing_fetch_workers", lambda: 2)
    monkeypatch.setattr(adapter, "_filing_fetch_timeout", lambda: 0.05)

    def fake_fetch(ticker_norm, **_kwargs):
        if ticker_norm == "SLOW":
            time.sleep(0.1)
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

    monkeypatch.setattr(adapter, "_fetch_filings_for_ticker", fake_fetch)

    tickers = ["FAST1", "SLOW", "FAST2"]

    with caplog.at_level(logging.INFO):
        rows = adapter.fetch_recent_filings(tickers)

    # Only the fast tickers should complete; the slow one should time out and be absent
    returned_tickers = {row.get("Ticker") for row in rows}
    assert returned_tickers == {"FAST1", "FAST2"}

    # Ensure summary and timeout logging were emitted
    timeout_logs = [rec for rec in caplog.records if "timeout" in rec.message]
    assert timeout_logs, "expected timeout log line"

    summary_logs = [rec for rec in caplog.records if "[edgar filings] summary" in rec.message]
    assert summary_logs and summary_logs[-1].message.endswith("timeout_pending=1")

    # Adapter stats should record the timeout
    assert adapter.last_filings_stats.get("timeout_pending") == 1
    assert adapter.last_filings_stats.get("scheduled_tickers") == 3
