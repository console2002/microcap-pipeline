import numpy as np
import pandas as pd

from app.candidate_shortlist import _select_primary_event
from app.events_utils import first_non_na, select_primary_catalyst
from app.weekly_deep_research import _catalyst_details


def test_shared_primary_catalyst_policy_matches_w2_w3():
    events = pd.DataFrame(
        [
            {
                "Ticker": "XYZ",
                "event_type": "ListingChange",
                "event_date": "2025-06-25",
                "event_tier": "Tier-1",
                "primary_source_url": "https://example.com/0625",
            },
            {
                "Ticker": "XYZ",
                "event_type": "ListingChange",
                "event_date": "2025-12-03",
                "event_tier": "Tier-1",
                "primary_source_url": "https://example.com/1203",
            },
            {
                "Ticker": "XYZ",
                "event_type": "ATMTermination",
                "event_date": "2025-11-10",
                "event_tier": "Tier-1",
                "primary_source_url": "https://example.com/1110",
            },
            {
                "Ticker": "XYZ",
                "event_type": "ContractAward",
                "event_date": "2025-07-01",
                "event_tier": "Tier-2",
                "primary_source_url": "https://example.com/0701",
            },
        ]
    )

    primary = select_primary_catalyst(events)
    shortlist_info = _select_primary_event(events)
    catalyst_score, catalyst_date, catalyst_type, catalyst_url = _catalyst_details(events)

    assert primary["event_date"] == "2025-12-03"
    assert shortlist_info["EventDate"] == "2025-12-03"
    assert catalyst_date == "2025-12-03"
    assert shortlist_info["CatalystType"] == "ListingChange"
    assert catalyst_type == "ListingChange"
    assert catalyst_score == "Tier-1"
    assert shortlist_info["PrimaryCatalystURL"] == "https://example.com/1203"
    assert catalyst_url == "https://example.com/1203"


def test_single_tier_one_selected():
    events = pd.DataFrame(
        [
            {"event_date": "2024-04-01", "event_type": "Guidance", "event_tier": "Tier-2"},
            {"event_date": "2024-05-01", "event_type": "Earnings", "event_tier": "Tier-1"},
        ]
    )

    primary = select_primary_catalyst(events)
    assert primary["event_type"] == "Earnings"
    assert primary["event_date"] == "2024-05-01"


def test_latest_tier_two_when_no_tier_one():
    events = pd.DataFrame(
        [
            {"event_date": "2024-05-01", "event_type": "Contract", "event_tier": "Tier-2"},
            {"event_date": "2024-06-01", "event_type": "Contract", "event_tier": "Tier-2"},
        ]
    )

    primary = select_primary_catalyst(events)
    assert primary["event_date"] == "2024-06-01"


def test_latest_none_when_no_tiers():
    events = pd.DataFrame(
        [
            {"event_date": "2024-01-01", "event_type": "Other"},
            {"event_date": "2024-02-01", "event_type": "Other"},
        ]
    )

    primary = select_primary_catalyst(events)
    assert primary["event_date"] == "2024-02-01"


def test_first_non_na_coalesces_dates():
    primary = {
        "event_type": "ListingChange",
        "event_date": np.nan,
        "EventDate": pd.Timestamp("2025-12-03"),
        "FilingDate": pd.Timestamp("2025-12-01"),
    }

    shortlist_info = _select_primary_event(pd.DataFrame([primary]))
    assert shortlist_info["EventDate"] == pd.Timestamp("2025-12-03")
    assert not pd.isna(shortlist_info["EventDate"])


def test_first_non_na_prefers_event_date():
    primary = {
        "event_type": "ListingChange",
        "event_date": pd.Timestamp("2025-12-03"),
        "EventDate": pd.Timestamp("2025-10-03"),
        "FilingDate": pd.Timestamp("2025-09-01"),
    }

    shortlist_info = _select_primary_event(pd.DataFrame([primary]))
    assert shortlist_info["EventDate"] == pd.Timestamp("2025-12-03")


def test_w2_w3_alignment_with_nan_event_date():
    events = pd.DataFrame(
        [
            {
                "Ticker": "XYZ",
                "event_type": "ListingChange",
                "event_date": np.nan,
                "EventDate": pd.Timestamp("2025-12-03"),
                "FilingDate": pd.Timestamp("2025-12-01"),
                "event_tier": "Tier-1",
                "primary_source_url": "https://example.com/1203",
            }
        ]
    )

    shortlist_info = _select_primary_event(events)
    _, catalyst_date, _, _ = _catalyst_details(events)

    assert shortlist_info["EventDate"] == pd.Timestamp("2025-12-03")
    assert catalyst_date == pd.Timestamp("2025-12-03")
    assert not pd.isna(shortlist_info["EventDate"])
    assert not pd.isna(catalyst_date)
