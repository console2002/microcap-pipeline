import os

import pandas as pd

from app.config import load_config
from app.edgar_adapter import EdgarAdapter
from app.pipeline import filings_step, init_logs


def test_filings_step_uses_fmp_when_configured(monkeypatch, tmp_path):
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg["Filings"] = {"Source": "FMP"}

    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["Paths"]["data"], exist_ok=True)

    runlog, errlog = init_logs(cfg)
    adapter = EdgarAdapter(cfg)

    monkeypatch.setattr(
        EdgarAdapter,
        "fetch_recent_filings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("EDGAR filings fetch should be skipped for FMP source")
        ),
        raising=False,
    )

    monkeypatch.setattr(
        "app.pipeline.compute_runway_quarters", lambda url, adapter=None: (None, None)
    )

    fmp_calls: list[list[str]] = []

    def _fake_fmp_fetch(client, cfg, tickers, progress_fn=None, stop_flag=None):
        fmp_calls.append(list(tickers))
        filed_at = pd.Timestamp.utcnow().date().isoformat()
        return [
            {
                "CIK": "0000000001",
                "Ticker": "AAA",
                "Company": "Alpha",
                "Form": "10-Q",
                "FiledAt": filed_at,
                "URL": "https://example.com/aaa/10q",
                "Desc": "",
            }
        ]

    monkeypatch.setattr("app.pipeline.fetch_filings_fmp", _fake_fmp_fetch)

    df_prof = pd.DataFrame([{"Ticker": "AAA", "CIK": "1", "Company": "Alpha"}])

    progress_messages: list[str] = []
    df_fil, _, _ = filings_step(
        cfg, adapter, runlog, errlog, df_prof, {}, progress_messages.append
    )

    assert fmp_calls == [["AAA"]]
    assert not df_fil.empty
    assert set(df_fil.get("Form", [])) == {"10-Q"}
    assert any("source=FMP" in msg for msg in progress_messages)
