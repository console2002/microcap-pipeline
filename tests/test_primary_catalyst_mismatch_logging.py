import logging

import pandas as pd

from app.weekly_deep_research import (
    _assemble_primary_catalyst_fields,
    _catalyst_details,
    _log_primary_catalyst_mismatches,
)


def test_primary_catalyst_tier_propagates_from_primary_event():
    shortlist = pd.DataFrame(
        [
            {
                "Ticker": "KLTR",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystDate": pd.Timestamp("2025-12-03"),
                "PrimaryCatalystURL": "https://sec.gov/kltr-20251203.htm",
            }
        ]
    )

    events = pd.DataFrame(
        [
            {
                "Ticker": "KLTR",
                "event_type": "ListingChange",
                "event_tier": "Tier-1",
                "event_date": pd.Timestamp("2025-12-03"),
                "primary_source_url": "https://sec.gov/kltr-20251203.htm",
            }
        ]
    )

    (
        _,
        catalyst_date,
        catalyst_type,
        catalyst_url,
        _,
        _,
        primary_event,
    ) = _catalyst_details(events)

    fields = _assemble_primary_catalyst_fields(
        primary_event, shortlist.iloc[0].to_dict(), catalyst_type, catalyst_date, catalyst_url
    )

    assert fields["PrimaryCatalystType"] == "ListingChange"
    assert fields["PrimaryCatalystDate"] == pd.Timestamp("2025-12-03")
    assert fields["PrimaryCatalystURL"] == "https://sec.gov/kltr-20251203.htm"
    assert fields["PrimaryCatalystTier"] == "Tier-1"


def test_primary_catalyst_mismatch_logged(caplog):
    shortlist = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "CIK": "0001",
                "PrimaryCatalystDate": "2025-12-03",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": "https://example.com/shortlist",
            }
        ]
    )

    deep_research = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "CIK": "0001",
                "PrimaryCatalystDate": "2025-06-25",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": "https://example.com/deep",
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.weekly_deep_research"):
        _log_primary_catalyst_mismatches(shortlist, deep_research)

    mismatch_logs = [rec for rec in caplog.records if rec.message == "PRIMARY_CATALYST_MISMATCH"]
    assert len(mismatch_logs) == 1
    record = mismatch_logs[0]
    assert record.ticker == "AAA"
    assert record.diffs["PrimaryCatalystDate"] == ("2025-12-03", "2025-06-25")


def test_primary_catalyst_match_silent(caplog):
    shortlist = pd.DataFrame(
        [
            {
                "Ticker": "BBB",
                "PrimaryCatalystDate": "2025-05-05",
                "PrimaryCatalystType": "ContractAward",
                "PrimaryCatalystTier": "Tier-2",
                "PrimaryCatalystURL": "https://example.com/shared",
            }
        ]
    )

    deep_research = pd.DataFrame(
        [
            {
                "Ticker": "BBB",
                "PrimaryCatalystDate": "2025-05-05",
                "PrimaryCatalystType": "ContractAward",
                "PrimaryCatalystTier": "Tier-2",
                "PrimaryCatalystURL": "https://example.com/shared",
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.weekly_deep_research"):
        _log_primary_catalyst_mismatches(shortlist, deep_research)

    mismatch_logs = [rec for rec in caplog.records if "PRIMARY_CATALYST_MISMATCH" in rec.message]
    assert not mismatch_logs


def test_primary_catalyst_mismatch_ignores_missing_tier(caplog):
    shortlist = pd.DataFrame(
        [
            {
                "Ticker": "KLTR",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystDate": pd.Timestamp("2025-12-03"),
                "PrimaryCatalystURL": "https://sec.gov/kltr-20251203.htm",
            }
        ]
    )

    deep = pd.DataFrame(
        [
            {
                "Ticker": "KLTR",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystDate": pd.Timestamp("2025-12-03"),
                "PrimaryCatalystURL": "https://sec.gov/kltr-20251203.htm",
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.weekly_deep_research"):
        _log_primary_catalyst_mismatches(shortlist, deep)

    assert not any("PRIMARY_CATALYST_MISMATCH" in r.message for r in caplog.records)


def test_primary_catalyst_mismatch_logs_when_tier_differs(caplog):
    shortlist = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystDate": pd.Timestamp("2025-12-03"),
                "PrimaryCatalystURL": "https://sec.gov/a-20251203.htm",
            }
        ]
    )

    deep = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "PrimaryCatalystType": "ListingChange",
                "PrimaryCatalystTier": "Tier-2",
                "PrimaryCatalystDate": pd.Timestamp("2025-12-03"),
                "PrimaryCatalystURL": "https://sec.gov/a-20251203.htm",
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.weekly_deep_research"):
        _log_primary_catalyst_mismatches(shortlist, deep)

    msgs = [r.message for r in caplog.records]
    assert any("PRIMARY_CATALYST_MISMATCH" in m for m in msgs)
