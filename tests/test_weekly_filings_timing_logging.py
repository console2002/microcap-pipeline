import logging
import os

import pandas as pd

from app.config import load_config
from app.edgar_adapter import EdgarAdapter
from app.pipeline import filings_step, init_logs


def _make_cfg(tmp_path) -> dict:
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg.setdefault("Workers", {}).setdefault("EDGAR", {})["Workers"] = 1
    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    return cfg


def _fake_fetch(self, ticker, *, whitelist, start_expr, progress_fn=None, stop_flag=None, idx=0, total=0):
    filed_at = pd.Timestamp.utcnow().date().isoformat()
    rows = [
        {
            "CIK": "0000000001",
            "Ticker": ticker,
            "Company": f"Company {ticker}",
            "Form": "10-Q",
            "FiledAt": filed_at,
            "URL": f"https://example.com/{ticker}/10q",
            "Desc": "",
        },
        {
            "CIK": "0000000001",
            "Ticker": ticker,
            "Company": f"Company {ticker}",
            "Form": "8-K",
            "FiledAt": filed_at,
            "URL": f"https://example.com/{ticker}/8k",
            "Desc": "",
        },
        {
            "CIK": "0000000001",
            "Ticker": ticker,
            "Company": f"Company {ticker}",
            "Form": "4",
            "FiledAt": filed_at,
            "URL": f"https://example.com/{ticker}/4",
            "Desc": "",
        },
    ]
    message = (
        "WEEKLY_FILINGS_TICKER: "
        f"ticker={ticker} cik=0000000001 raw_filings={len(rows)} kept_after_form_filter={len(rows)} "
        "fetch_ms=1 duration_ms=1"
    )
    if progress_fn:
        progress_fn(message)
    return rows, {
        "ticker": ticker,
        "raw_count": len(rows),
        "kept_count": len(rows),
        "duration_ms": 1,
        "fetch_ms": 1,
        "cik": "0000000001",
    }


def test_weekly_filings_summary_includes_roles_and_rl_wait(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(EdgarAdapter, "_fetch_filings_for_ticker", _fake_fetch, raising=False)
    monkeypatch.setattr(
        "app.pipeline.compute_runway_quarters",
        lambda url, adapter=None, **_: (None, None, "", "", {}),
    )

    cfg = _make_cfg(tmp_path)
    runlog, errlog = init_logs(cfg)
    adapter = EdgarAdapter(cfg)
    caplog.set_level(logging.INFO)

    df_prof = pd.DataFrame([
        {"Ticker": "AAA", "CIK": "1", "Company": "Alpha"},
    ])

    progress_messages: list[str] = []
    filings_step(
        cfg,
        adapter,
        runlog,
        errlog,
        df_prof,
        {},
        progress_messages.append,
    )

    combined = progress_messages + [record.message for record in caplog.records]

    assert any("WEEKLY_FILINGS_AFTER_PRUNE" in msg for msg in combined)
    assert any("WEEKLY_FILINGS_SUMMARY" in msg and "roles=" in msg for msg in combined)
    assert any("rl_wait_sec" in msg for msg in combined)
