import numpy as np
import pandas as pd
import pytest

from app.build_watchlist import (
    EightKEvent,
    _EIGHT_K_EVENTS_COLUMNS,
    _EightKProcessResult,
    _event_bucket_key,
    _generate_eight_k_events,
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


def _make_event(
    url: str, ticker: str, tier: str, accession: str, filing_date: str = "2024-01-01", cik: str | None = None
) -> tuple[EightKEvent, dict]:
    event = EightKEvent(
        cik=cik if cik is not None else ("0001" if ticker == "AAA" else "0002"),
        ticker=ticker,
        filing_date=filing_date,
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


class _DummyEvent:
    def __init__(self, ticker, cik):
        self.ticker = ticker
        self.cik = cik


def test_event_bucket_key_prefers_ticker():
    event = _DummyEvent("KLTR", "0001432133")
    assert _event_bucket_key(event) == "KLTR"


def test_event_bucket_key_falls_back_to_cik_for_nan():
    event = _DummyEvent(np.nan, "0001432133")
    assert _event_bucket_key(event) == "0001432133"


def test_event_bucket_key_falls_back_to_cik_for_none():
    event = _DummyEvent(None, "0001432133")
    assert _event_bucket_key(event) == "0001432133"


def test_event_bucket_key_returns_none_without_identifiers():
    event = _DummyEvent(None, None)
    assert _event_bucket_key(event) is None


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


def test_early_exit_collapses_only_when_tier1_present(tmp_path, monkeypatch):
    rows = [
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a1", "AccessionNo": "0001"},
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a2", "AccessionNo": "0002"},
        {"Ticker": "BBB", "CIK": "0002", "Form": "8-K", "URL": "https://b1", "AccessionNo": "0003"},
        {"Ticker": "BBB", "CIK": "0002", "Form": "8-K", "URL": "https://b2", "AccessionNo": "0004"},
    ]
    filings_df = pd.DataFrame(rows)
    filings_df.to_csv(tmp_path / csv_filename("filings"), index=False)

    events_by_url = {
        "https://a1": _make_event("https://a1", "AAA", "Tier-2", "0001", filing_date="2025-01-01"),
        "https://a2": _make_event("https://a2", "AAA", "Tier-1", "0002", filing_date="2025-02-01"),
        "https://b1": _make_event("https://b1", "BBB", "Tier-2", "0003", filing_date="2024-03-01"),
        "https://b2": _make_event("https://b2", "BBB", "Tier-2", "0004", filing_date="2024-04-01"),
    }
    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    cfg = _make_cfg(tmp_path, early_exit_on_tier1=False)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()
    baseline_output = pd.read_csv(events_path)

    early_exit_cfg = _make_cfg(tmp_path, early_exit_on_tier1=True)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=early_exit_cfg)

    output = pd.read_csv(events_path)

    aaa_rows = output[output["Ticker"] == "AAA"]
    assert len(aaa_rows) == 1
    assert (aaa_rows["EventTier"] == "Tier-1").all()
    assert pd.Timestamp(aaa_rows.iloc[0]["EventDate"]) == pd.Timestamp("2025-02-01")

    aaa_baseline = baseline_output[baseline_output["Ticker"] == "AAA"]
    assert len(aaa_baseline) == 2

    bbb_rows = output[output["Ticker"] == "BBB"]
    bbb_baseline = baseline_output[baseline_output["Ticker"] == "BBB"]
    assert len(bbb_rows) == len(bbb_baseline)
    assert set(bbb_rows["FilingURL"]) == set(bbb_baseline["FilingURL"])


def test_early_exit_selects_primary_deterministically(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a1", "AccessionNo": "0001"},
            {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a2", "AccessionNo": "0002"},
            {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a3", "AccessionNo": "0003"},
            {"Ticker": "BBB", "CIK": "0002", "Form": "8-K", "URL": "https://b1", "AccessionNo": "0004"},
        ]
    )
    df.to_csv(tmp_path / csv_filename("filings"), index=False)

    events_by_url = {
        "https://a1": _make_event("https://a1", "AAA", "Tier-1", "0001", filing_date="2025-06-25"),
        "https://a2": _make_event("https://a2", "AAA", "Tier-2", "0002", filing_date="2025-12-02"),
        "https://a3": _make_event("https://a3", "AAA", "Tier-1", "0003", filing_date="2025-12-03"),
        "https://b1": _make_event("https://b1", "BBB", "Tier-1", "0004", filing_date="2024-02-01"),
    }
    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    cfg = _make_cfg(tmp_path, early_exit_on_tier1=False)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()
    output = pd.read_csv(events_path)
    assert len(output) == len(events_by_url)

    early_exit_cfg = _make_cfg(tmp_path, early_exit_on_tier1=True)
    events_df, _, _ = generate_eight_k_events(data_dir=str(tmp_path), cfg=early_exit_cfg)

    early_exit_output = pd.read_csv(events_path)
    primary_dates = early_exit_output.loc[early_exit_output["Ticker"] == "AAA", "EventDate"].unique()
    assert len(primary_dates) == 1
    assert pd.Timestamp(primary_dates[0]) == pd.Timestamp("2025-12-03")

    bbb_rows = early_exit_output.loc[early_exit_output["Ticker"] == "BBB"]
    assert len(bbb_rows) == 1
    assert (bbb_rows["EventTier"] == "Tier-1").all()


def test_eight_k_heartbeat_uses_parsed_results(tmp_path, monkeypatch):
    df = pd.DataFrame([
        {"Ticker": "AAA", "CIK": "0001", "Form": "8-K", "URL": "https://a1", "AccessionNo": "0001"}
    ])
    df.to_csv(tmp_path / csv_filename("filings"), index=False)

    event, csv_row = _make_event("https://a1", "AAA", "Tier-1", "0001")

    def _stub(row):
        return _EightKProcessResult(url=row.URL, event=event, csv_row=csv_row, debug_entry=None, log_messages=[])

    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _stub)

    base_time = 1_000_000.0
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        return base_time + calls["n"] * 31

    monkeypatch.setattr("app.build_watchlist.time.time", fake_time)

    messages: list[str] = []

    events_df, _, counts = _generate_eight_k_events(
        data_dir=str(tmp_path),
        progress_fn=lambda msg: messages.append(msg),
        early_exit_on_tier1=False,
    )

    assert counts["parsed"] == 1
    assert not events_df.empty
    assert any("eight_k: heartbeat processed" in message for message in messages)
    assert any("parsed" in message and "failed" in message for message in messages)


def test_early_exit_does_not_collapse_blank_tickers(tmp_path, monkeypatch):
    rows = [
        {"Ticker": "", "CIK": "", "Form": "8-K", "URL": "https://blank1", "AccessionNo": "0010"},
        {"Ticker": "", "CIK": "", "Form": "8-K", "URL": "https://blank2", "AccessionNo": "0011"},
        {"Ticker": "", "CIK": "", "Form": "8-K", "URL": "https://blank3", "AccessionNo": "0012"},
        {"Ticker": "", "CIK": "0099", "Form": "8-K", "URL": "https://cik1", "AccessionNo": "0013"},
        {"Ticker": "", "CIK": "0099", "Form": "8-K", "URL": "https://cik2", "AccessionNo": "0014"},
    ]
    filings_df = pd.DataFrame(rows)
    filings_df.to_csv(tmp_path / csv_filename("filings"), index=False)

    events_by_url = {
        "https://blank1": _make_event("https://blank1", "", "Tier-1", "0010", filing_date="2024-03-01", cik=""),
        "https://blank2": _make_event("https://blank2", "", "Tier-2", "0011", filing_date="2024-04-01", cik=""),
        "https://blank3": _make_event("https://blank3", "", "Tier-1", "0012", filing_date="2024-05-01", cik=""),
        "https://cik1": _make_event("https://cik1", "", "Tier-2", "0013", filing_date="2024-01-01", cik="0099"),
        "https://cik2": _make_event("https://cik2", "", "Tier-1", "0014", filing_date="2024-02-01", cik="0099"),
    }

    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    early_exit_cfg = _make_cfg(tmp_path, early_exit_on_tier1=True)
    generate_eight_k_events(data_dir=str(tmp_path), cfg=early_exit_cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()

    output = pd.read_csv(events_path, dtype=str).fillna("")

    blank_ticker_rows = output[output["Ticker"] == ""]

    no_bucket_rows = blank_ticker_rows[blank_ticker_rows["CIK"] == ""]
    assert len(no_bucket_rows) == 3
    assert set(no_bucket_rows["FilingURL"]) == {"https://blank1", "https://blank2", "https://blank3"}

    cik_bucket_rows = blank_ticker_rows[blank_ticker_rows["CIK"] == "0099"]
    assert len(cik_bucket_rows) == 1
    assert set(cik_bucket_rows["FilingURL"]) == {"https://cik2"}


def test_early_exit_buckets_nan_tickers_by_cik(tmp_path, monkeypatch):
    rows = [
        {"Ticker": "", "CIK": "0111", "Form": "8-K", "URL": "https://nan1", "AccessionNo": "0101"},
        {"Ticker": "", "CIK": "0111", "Form": "8-K", "URL": "https://nan2", "AccessionNo": "0102"},
        {"Ticker": "", "CIK": "0222", "Form": "8-K", "URL": "https://nan3", "AccessionNo": "0201"},
    ]
    filings_df = pd.DataFrame(rows)
    filings_df.to_csv(tmp_path / csv_filename("filings"), index=False)

    events_by_url = {
        "https://nan1": _make_event(
            "https://nan1", np.nan, "Tier-1", "0101", filing_date="2024-06-01", cik="0111"
        ),
        "https://nan2": _make_event(
            "https://nan2", np.nan, "Tier-2", "0102", filing_date="2024-05-01", cik="0111"
        ),
        "https://nan3": _make_event(
            "https://nan3", np.nan, "Tier-1", "0201", filing_date="2024-07-01", cik="0222"
        ),
    }

    monkeypatch.setattr("app.build_watchlist._process_eight_k_row", _make_process_stub(events_by_url))

    early_exit_cfg = _make_cfg(tmp_path, early_exit_on_tier1=True)
    generate_eight_k_events(data_dir=str(tmp_path), cfg=early_exit_cfg)

    events_path = tmp_path / csv_filename("eight_k_events")
    assert events_path.exists()

    output = pd.read_csv(events_path, dtype=str).fillna("")

    first_cik_rows = output[output["CIK"] == "0111"]
    second_cik_rows = output[output["CIK"] == "0222"]

    assert len(first_cik_rows) == 1
    assert set(first_cik_rows["FilingURL"]) == {"https://nan1"}

    assert len(second_cik_rows) == 1
    assert set(second_cik_rows["FilingURL"]) == {"https://nan3"}

    assert len(output) == 2
