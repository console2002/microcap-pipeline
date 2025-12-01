import os

import pandas as pd

from app.config import load_config
from app.csv_names import csv_path
from app.pipeline import filings_step, init_logs


def test_filings_csv_includes_accession_and_master(monkeypatch, tmp_path):
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg["Filings"] = {"Source": "EDGAR"}

    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["Paths"]["data"], exist_ok=True)

    runlog, errlog = init_logs(cfg)

    monkeypatch.setattr(
        "app.pipeline.compute_runway_quarters",
        lambda url, adapter=None, **_: (None, None, "", ""),
    )

    class DummyAdapter:
        def __init__(self):
            self.last_filings_stats = {}

        def fetch_recent_filings(
            self,
            tickers,
            progress_fn=None,
            stop_flag=None,
            *,
            skip_tickers=None,
            on_batch=None,
        ):
            filed_at = pd.Timestamp.utcnow().isoformat()
            batch = [
                {
                    "CIK": "0000000001",
                    "Ticker": "AAA",
                    "Company": "Alpha",
                    "Form": "10-Q",
                    "FiledAt": filed_at,
                    "URL": "https://example.com/aaa/10q",
                    "Desc": "",
                    "Accession": "0001-24-000001",
                    "MasterTxtURL": "https://example.com/aaa/10q.txt",
                }
            ]

            if on_batch:
                on_batch(batch, tickers[0] if tickers else "")

            self.last_filings_stats = {
                "source": "EDGAR",
                "raw_filings": len(batch),
                "kept_filings": len(batch),
                "rl_wait_sec": 0,
            }

        def stats_string(self) -> str:
            return "dummy"

    adapter = DummyAdapter()

    df_prof = pd.DataFrame([{"Ticker": "AAA", "CIK": "1", "Company": "Alpha"}])

    progress_messages: list[str] = []
    filings_step(cfg, adapter, runlog, errlog, df_prof, {}, progress_messages.append)

    filings_csv = csv_path(cfg["Paths"]["data"], "filings")
    df_out = pd.read_csv(filings_csv)

    expected_header = [
        "CIK",
        "URL",
        "Ticker",
        "Company",
        "Form",
        "FiledAt",
        "Age",
        "RunwayQuarters",
        "HasRunway",
        "RunwaySourceURL",
        "RunwayReasonCode",
        "RunwayReasonDetail",
        "Desc",
        "Accession",
        "MasterTxtURL",
    ]

    assert list(df_out.columns) == expected_header
    assert df_out.loc[0, "Accession"] == "0001-24-000001"
    assert df_out.loc[0, "MasterTxtURL"] == "https://example.com/aaa/10q.txt"
    assert df_out.loc[0, "URL"] == "https://example.com/aaa/10q"
