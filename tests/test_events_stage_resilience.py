import logging

import pandas as pd
import pytest

from app.build_watchlist import _EightKProcessResult, EightKEvent
from app.pipeline import parse_8k_step, run_weekly_pipeline


class _DummyAdapter:
    def __init__(self, *_args, **_kwargs):
        pass


def _make_cfg(base_dir):
    return {
        "Paths": {"data": str(base_dir), "logs": str(base_dir)},
        "UserAgent": "test-agent",
        "TimeoutSeconds": 1,
        "Retries": 1,
        "BackoffSeconds": [1, 1, 1],
        "GUI": {"SingleRunLock": False},
    }


def test_parse_8k_per_filing_error_continues(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    cfg = _make_cfg(tmp_path)
    runlog = tmp_path / "runlog.csv"
    errlog = tmp_path / "errlog.csv"
    runlog.write_text("")
    errlog.write_text("")

    filings = pd.DataFrame(
        [
            {"Ticker": "BAD", "CIK": "0001", "Form": "8-K", "URL": "https://bad", "AccessionNo": "0001"},
            {
                "Ticker": "GOOD",
                "CIK": "0002",
                "Form": "8-K",
                "URL": "https://good",
                "AccessionNo": "0002",
                "FilingDate": "2024-01-01",
            },
        ]
    )
    filings.to_csv(tmp_path / "02_filings.csv", index=False)

    event = EightKEvent(
        cik="0002",
        ticker="GOOD",
        filing_date="2024-01-01",
        filing_url="https://good",
        items_present="1.01",
        is_catalyst=True,
        catalyst_type="Other",
        catalyst_label="",
        tier1_type="",
        tier1_trigger="",
        is_dilution=False,
        dilution_tags=[],
        ignore_reason="",
        company="Good Co",
        form="8-K",
    )

    good_row = {
        "CIK": event.cik,
        "Company": event.company,
        "Ticker": event.ticker,
        "Form": event.form,
        "FilingDate": event.filing_date,
        "EventDate": event.filing_date,
        "DateOfReport": "",
        "AccessionNo": "0002",
        "FilingURL": event.filing_url,
        "FilingUrlTxt": event.filing_url,
        "PeriodOfReport": "",
        "AcceptanceDateTime": "",
        "HomepageURL": "",
        "ItemsPresent": event.items_present,
        "ItemsNormalized": "1.01",
        "HasPressRelease": False,
        "HasExhibits": False,
        "PrimaryEx99Docs": "",
        "PrimaryEx10Docs": "",
        "HasXBRL": False,
        "EventType": "Other",
        "EventTier": "Tier-2",
        "PrimarySourceURL": event.filing_url,
        "SecondarySourceURL": "",
        "IsCatalyst": True,
        "CatalystType": "Other",
        "Tier1Type": "",
        "Tier1Trigger": "",
        "IsDilution": False,
        "DilutionTags": "",
        "IgnoreReason": "",
    }

    def fake_process(row):
        if getattr(row, "Ticker", "") == "BAD":
            raise RuntimeError("boom")
        return _EightKProcessResult(
            url=event.filing_url,
            event=event,
            csv_row=good_row,
            debug_entry=None,
            log_messages=[],
        )

    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", fake_process)

    result = parse_8k_step(
        cfg,
        str(runlog),
        str(errlog),
        {"stop": False},
        None,
        adapter=_DummyAdapter(),
    )

    assert result.status == "partial_success_with_errors"
    events_path = tmp_path / "09_events.csv"
    assert events_path.exists()
    events_df = pd.read_csv(events_path)
    assert len(events_df) == 1
    assert "ticker=BAD" in " ".join(caplog.messages)


def test_weekly_pipeline_hard_failure_short_circuits(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("")

    pd.DataFrame({"Ticker": ["AAA"], "Company": ["Alpha"], "CIK": ["0001"]}).to_csv(
        tmp_path / "01_profiles.csv", index=False
    )
    pd.DataFrame(
        [{"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://good", "AccessionNo": "0001"}]
    ).to_csv(tmp_path / "02_filings.csv", index=False)

    monkeypatch.setattr("app.pipeline.load_config", lambda: cfg)
    monkeypatch.setattr("app.pipeline.make_client", lambda _cfg: None)
    monkeypatch.setattr("app.pipeline.create_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.pipeline.clear_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.pipeline.is_locked", lambda _cfg: False)
    monkeypatch.setattr("app.pipeline.EdgarAdapter", lambda *_args, **_kwargs: _DummyAdapter())
    monkeypatch.setattr("app.pipeline.set_adapter", lambda *_args, **_kwargs: None)

    downstream_called = {"shortlist": False}

    def fake_generate(*_args, **_kwargs):
        raise RuntimeError("fatal")

    def fake_shortlist(*_args, **_kwargs):
        downstream_called["shortlist"] = True

    monkeypatch.setattr("app.pipeline.generate_eight_k_events", fake_generate)
    monkeypatch.setattr("app.pipeline.hydrate_and_shortlist_step", fake_shortlist)

    run_weekly_pipeline(start_stage="events")

    events_path = tmp_path / "09_events.csv"
    assert events_path.exists()
    events_df = pd.read_csv(events_path)
    assert events_df.empty
    assert downstream_called["shortlist"] is False
