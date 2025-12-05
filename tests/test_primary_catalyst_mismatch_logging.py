import logging

import pandas as pd

from app.weekly_deep_research import _log_primary_catalyst_mismatches


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

    mismatch_logs = [rec for rec in caplog.records if "PRIMARY_CATALYST_MISMATCH" in rec.message]
    assert len(mismatch_logs) == 1
    message = mismatch_logs[0].message
    assert "AAA" in message
    assert "2025-12-03" in message
    assert "2025-06-25" in message


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
