import pandas as pd

from app.build_watchlist import _EIGHT_K_EVENTS_COLUMNS, EightKEvent, load_or_generate_eight_k_events
from app.csv_names import csv_filename


def _csv_row_from_event(event: EightKEvent) -> dict:
    row = {key: "" for key in _EIGHT_K_EVENTS_COLUMNS}
    row.update(
        {
            "CIK": event.cik,
            "Company": event.company,
            "Ticker": event.ticker,
            "Form": event.form,
            "FilingDate": event.filing_date,
            "EventDate": event.date_of_report or event.filing_date,
            "DateOfReport": event.date_of_report,
            "AccessionNo": event.accession_no,
            "FilingURL": event.filing_url,
            "FilingUrlTxt": event.filing_url_txt,
            "PeriodOfReport": event.period_of_report,
            "AcceptanceDateTime": event.acceptance_datetime,
            "HomepageURL": event.homepage_url,
            "ItemsPresent": event.items_present,
            "ItemsNormalized": event.items_normalized,
            "HasPressRelease": event.has_press_release,
            "HasExhibits": event.has_exhibits,
            "PrimaryEx99Docs": event.primary_ex99_docs,
            "PrimaryEx10Docs": event.primary_ex10_docs,
            "HasXBRL": event.has_xbrl,
            "EventType": event.event_type,
            "EventTier": event.event_tier,
            "PrimarySourceURL": event.primary_source_url,
            "SecondarySourceURL": event.secondary_source_url,
            "IsCatalyst": event.is_catalyst,
            "CatalystType": event.catalyst_type,
            "Tier1Type": event.tier1_type,
            "Tier1Trigger": event.tier1_trigger,
            "IsDilution": event.is_dilution,
            "DilutionTags": event.dilution_tags_joined(),
            "IgnoreReason": event.ignore_reason,
        }
    )
    return row


def test_cached_events_load_without_config(tmp_path, monkeypatch):
    event = EightKEvent(
        cik="0001",
        ticker="AAA",
        filing_date="2024-01-01",
        filing_url="https://example.com/a1",
        items_present="1.01",
        is_catalyst=True,
        catalyst_type="Tier-1",
        catalyst_label="",
        tier1_type="",
        tier1_trigger="",
        is_dilution=False,
        dilution_tags=[],
        ignore_reason="",
        company="AAA Corp",
        form="8-K",
        event_type="ContractAward",
        event_tier="Tier-1",
        primary_source_url="https://example.com/a1",
    )

    cached_events_path = tmp_path / csv_filename("eight_k_events")
    pd.DataFrame([_csv_row_from_event(event)], columns=_EIGHT_K_EVENTS_COLUMNS).to_csv(
        cached_events_path, index=False
    )

    def _fail_load_config():
        raise AssertionError("load_config should not be called when cached events exist")

    monkeypatch.setattr("app.build_watchlist.load_config", _fail_load_config)

    df, lookup, counts = load_or_generate_eight_k_events(
        str(tmp_path), progress_fn=lambda *_args, **_kwargs: None, cfg=None
    )

    assert not df.empty
    assert counts == {"parsed": 1, "failed": 0, "total_filings": 1}
    assert lookup.by_ticker.get("AAA")
    assert pd.read_csv(cached_events_path).equals(df)
