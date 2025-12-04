import pandas as pd
import pytest

from app.build_watchlist import (
    EightKEvent,
    _EIGHT_K_EVENTS_COLUMNS,
    _EightKProcessResult,
    generate_eight_k_events,
)
from app.csv_names import csv_filename


@pytest.fixture
def filings_df(tmp_path):
    rows = [
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a1", "AccessionNo": "0001"},
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a2", "AccessionNo": "0002"},
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a3", "AccessionNo": "0003"},
        {"Ticker": "BBB", "CIK": "0002", "Form": "8-K", "URL": "https://b1", "AccessionNo": "0004"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / csv_filename("filings"), index=False)
    return df


def _make_cfg(base_dir, early_exit_on_tier1=None):
    cfg = {
        "Paths": {"data": str(base_dir), "logs": str(base_dir)},
        "UserAgent": "test-agent",
        "TimeoutSeconds": 1,
        "Retries": 1,
        "BackoffSeconds": [1, 1, 1],
        "GUI": {"SingleRunLock": False},
    }
    if early_exit_on_tier1 is not None:
        cfg["Events"] = {"EarlyExitOnTier1": early_exit_on_tier1}
    return cfg


def _csv_row_from_event(event: EightKEvent, accession: str) -> dict:
    row = {
        "CIK": event.cik,
        "Company": event.company,
        "Ticker": event.ticker,
        "Form": event.form,
        "FilingDate": event.filing_date,
        "EventDate": event.filing_date,
        "DateOfReport": event.date_of_report,
        "AccessionNo": accession,
        "FilingURL": event.filing_url,
        "FilingUrlTxt": event.filing_url,
        "PeriodOfReport": event.period_of_report,
        "AcceptanceDateTime": event.acceptance_datetime,
        "HomepageURL": event.homepage_url,
        "ItemsPresent": event.items_present,
        "ItemsNormalized": event.items_normalized or event.items_present,
        "HasPressRelease": event.has_press_release,
        "HasExhibits": event.has_exhibits,
        "PrimaryEx99Docs": event.primary_ex99_docs,
        "PrimaryEx10Docs": event.primary_ex10_docs,
        "HasXBRL": event.has_xbrl,
        "EventType": event.event_type,
        "EventTier": event.event_tier,
        "PrimarySourceURL": event.primary_source_url or event.filing_url,
        "SecondarySourceURL": event.secondary_source_url,
        "IsCatalyst": event.is_catalyst,
        "CatalystType": event.catalyst_type,
        "Tier1Type": event.tier1_type,
        "Tier1Trigger": event.tier1_trigger,
        "IsDilution": event.is_dilution,
        "DilutionTags": event.dilution_tags_joined(),
        "IgnoreReason": event.ignore_reason,
    }
    return {key: row.get(key, "") for key in _EIGHT_K_EVENTS_COLUMNS}


def _make_event(url: str, ticker: str, tier: str, accession: str) -> tuple[EightKEvent, dict]:
    event = EightKEvent(
        cik="0001" if ticker == "AAA" else "0002",
        ticker=ticker,
        filing_date="2024-01-01",
        filing_url=url,
        items_present="1.01",
        is_catalyst=True,
        catalyst_type=tier,
        catalyst_label="",
        tier1_type="",
        tier1_trigger="",
        is_dilution=False,
        dilution_tags=[],
        ignore_reason="",
        company=f"{ticker} Corp",
        form="8-K",
        event_type="ContractAward",
        event_tier=tier,
        primary_source_url=url,
    )
    return event, _csv_row_from_event(event, accession)


def _make_process_stub(events_by_url):
    def _stub(row):
        url = getattr(row, "URL", "")
        event, csv_row = events_by_url[url]
        return _EightKProcessResult(url=url, event=event, csv_row=csv_row, debug_entry=None, log_messages=[])

    return _stub


def test_events_default_processes_all(tmp_path, filings_df, monkeypatch):
    events_by_url = {
        "https://a1": _make_event("https://a1", "AAA", "Tier-1", "0001"),
        "https://a2": _make_event("https://a2", "AAA", "Tier-2", "0002"),
        "https://a3": _make_event("https://a3", "AAA", "Tier-1", "0003"),
        "https://b1": _make_event("https://b1", "BBB", "Tier-1", "0004"),
    }
    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    cfg = _make_cfg(tmp_path)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()
    output = pd.read_csv(events_path)
    assert len(output) == len(events_by_url)
    assert set(output["FilingURL"]) == set(events_by_url.keys())


def test_early_exit_limits_per_ticker_only(tmp_path, filings_df, monkeypatch):
    events_by_url = {
        "https://a1": _make_event("https://a1", "AAA", "Tier-1", "0001"),
        "https://a2": _make_event("https://a2", "AAA", "Tier-2", "0002"),
        "https://a3": _make_event("https://a3", "AAA", "Tier-1", "0003"),
        "https://b1": _make_event("https://b1", "BBB", "Tier-1", "0004"),
    }
    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    cfg = _make_cfg(tmp_path, early_exit_on_tier1=True)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()
    output = pd.read_csv(events_path)

    baseline_count = len(events_by_url)
    assert len(output) <= baseline_count

    aaa_rows = output[output["Ticker"] == "AAA"]
    assert not aaa_rows.empty
    assert (aaa_rows["EventTier"] == "Tier-1").any()

    bbb_rows = output[output["Ticker"] == "BBB"]
    assert len(bbb_rows) == 1
    assert (bbb_rows["EventTier"] == "Tier-1").all()
