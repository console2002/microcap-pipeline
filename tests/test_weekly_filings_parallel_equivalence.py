import pandas as pd
import pytest

import os

from app.config import load_config
from app.edgar_adapter import EdgarAdapter
from app.pipeline import filings_step, init_logs


def _make_cfg(tmp_path, workers: int) -> dict:
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg.setdefault("Workers", {}).setdefault("EDGAR", {})["Workers"] = workers
    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    return cfg


@pytest.fixture(autouse=True)
def patch_runway(monkeypatch):
    monkeypatch.setattr(
        "app.pipeline.compute_runway_quarters", lambda url, adapter=None: (None, None)
    )
    yield


def _fake_fetch(self, ticker, *, whitelist, start_expr, progress_fn=None, stop_flag=None, idx=0, total=0):
    filed_at = pd.Timestamp.utcnow().date().isoformat()
    rows = [
        {
            "CIK": f"{idx:010d}",
            "Ticker": ticker,
            "Company": f"Company {ticker}",
            "Form": "8-K",
            "FiledAt": filed_at,
            "URL": f"https://example.com/{ticker}/8k",
            "Desc": "",
        },
        {
            "CIK": f"{idx:010d}",
            "Ticker": ticker,
            "Company": f"Company {ticker}",
            "Form": "10-Q",
            "FiledAt": filed_at,
            "URL": f"https://example.com/{ticker}/10q",
            "Desc": "",
        },
    ]
    return rows, {"ticker": ticker, "raw_count": len(rows), "kept_count": len(rows), "duration_ms": 1}


def _run_filings(tmp_path, workers: int):
    cfg = _make_cfg(tmp_path, workers)
    runlog, errlog = init_logs(cfg)
    adapter = EdgarAdapter(cfg)

    df_prof = pd.DataFrame(
        [
            {"Ticker": "AAA", "CIK": "1", "Company": "Alpha"},
            {"Ticker": "BBB", "CIK": "2", "Company": "Beta"},
        ]
    )

    stop_flag = {}
    progress_messages: list[str] = []

    result, _, _ = filings_step(
        cfg,
        adapter,
        runlog,
        errlog,
        df_prof,
        stop_flag,
        progress_messages.append,
    )
    return result


def test_weekly_filings_parallel_equivalence(monkeypatch, tmp_path):
    monkeypatch.setattr(EdgarAdapter, "_fetch_filings_for_ticker", _fake_fetch, raising=False)

    df_seq = _run_filings(tmp_path / "seq", workers=1)
    df_par = _run_filings(tmp_path / "par", workers=4)

    sorted_seq = df_seq.sort_values(["Ticker", "Form", "FiledAt", "URL"]).reset_index(drop=True)
    sorted_par = df_par.sort_values(["Ticker", "Form", "FiledAt", "URL"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(sorted_seq.reset_index(drop=True), sorted_par.reset_index(drop=True))
