import pandas as pd

from app.build_watchlist import _classify_eight_k, _write_canonical_events
from app.candidate_shortlist import build_candidate_shortlist
from app.eight_k_parser import EdgarEightKParseResult
from edgar_core.eight_k import classify_event


class _FakeEightK:
    def __init__(self, items, text):
        self.items = items
        self._text = text

    def __getitem__(self, key):  # pragma: no cover - trivial delegation
        return self._text


class _FakeFiling:
    def __init__(self):
        self.exhibits = []


def test_classify_event_and_canonical_table(tmp_path):
    event_type, tier = classify_event(
        _FakeEightK(["1.01"], "entered an at-the-market equity offering"),
        _FakeFiling(),
    )
    assert event_type == "ATM_CREATION"
    assert tier == "Tier-2"

    events_df = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "EventDate": "2024-05-01",
                "FilingDate": "2024-05-01",
                "EventType": "ATM_TERMINATION",
                "EventTier": "Tier-1",
                "FilingURL": "https://www.sec.gov/aaa",
                "Form": "8-K",
            }
        ]
    )
    canonical = _write_canonical_events(events_df, str(tmp_path))
    assert (tmp_path / "09_events.csv").exists()
    required_cols = {
        "Ticker",
        "Company",
        "CIK",
        "event_date",
        "event_type",
        "event_tier",
        "primary_source_url",
        "secondary_source_url",
    }
    assert required_cols.issubset(canonical.columns)
    row = canonical.iloc[0]
    assert row["event_type"] == "ATM_TERMINATION"
    assert row["event_tier"] == "Tier-1"
    assert row["primary_source_url"].startswith("https://www.sec.gov")


def test_atm_creation_vs_termination(tmp_path):
    creation_type, creation_tier = classify_event(
        _FakeEightK(["1.01", "8.01"], "entered into a sales agreement for an at-the-market offering"),
        _FakeFiling(),
    )
    assert creation_type == "ATM_CREATION"
    assert creation_tier == "Tier-2"

    termination_type, termination_tier = classify_event(
        _FakeEightK(["1.01"], "the sales agreement was terminated and no further sales will occur"),
        _FakeFiling(),
    )
    assert termination_type == "ATM_TERMINATION"
    assert termination_tier == "Tier-1"


def test_non_actionable_item_not_catalyst():
    parsed = EdgarEightKParseResult(
        filing=_FakeFiling(),
        eight_k=_FakeEightK(["Item 3.01"], "notice of delisting"),
        items_raw=["Item 3.01"],
        items_normalized=["3.01"],
        exhibits=[],
        press_releases=[],
        press_release_text="",
        event_text="",
        filing_url_txt="",
        primary_ex99_docs="",
        primary_ex10_docs="",
        has_press_release=False,
        has_exhibits=False,
        has_xbrl=False,
        raw_submission_text="",
    )

    result = _classify_eight_k(parsed)

    assert result["tier"] == "Other"
    assert not result["is_catalyst"]


def test_build_candidate_shortlist_w2_columns(tmp_path):
    data_dir = tmp_path
    # universe input
    pd.DataFrame(
        [
            {"Ticker": "AAA", "Company": "Alpha", "CIK": "0000000001", "Exchange": "NASDAQ", "MarketCap": 50_000_000, "ADV20": 80_000, "Close": 3.5},
            {"Ticker": "BBB", "Company": "Beta", "CIK": "0000000002", "Exchange": "NYSE", "MarketCap": None, "ADV20": None, "Close": None},
        ]
    ).to_csv(data_dir / "01_universe_gated.csv", index=False)

    # events
    events = pd.DataFrame(
        [
            {"Ticker": "AAA", "Company": "Alpha", "CIK": "0000000001", "event_date": "2024-06-02", "event_type": "ContractAward", "event_tier": "Tier-1", "primary_source_url": "https://www.sec.gov/a"},
            {"Ticker": "AAA", "Company": "Alpha", "CIK": "0000000001", "event_date": "2024-05-01", "event_type": "Other", "event_tier": "Tier-2", "primary_source_url": "https://www.sec.gov/a-old"},
            {"Ticker": "BBB", "Company": "Beta", "CIK": "0000000002", "event_date": "2024-06-01", "event_type": "GuidanceUp", "event_tier": "Tier-2", "primary_source_url": "https://www.sec.gov/b"},
        ]
    )
    events.to_csv(data_dir / "09_events.csv", index=False)

    # hydrated cache for prices/cap/adv
    hydrated = pd.DataFrame(
        [
            {"Ticker": "AAA", "Close": 3.5, "ADV20": 80_000, "MarketCap": 50_000_000},
        ]
    )
    hydrated.to_csv(data_dir / "05_hydrated_candidates.csv", index=False)

    shortlist = build_candidate_shortlist(str(data_dir))
    assert (data_dir / "20_candidate_shortlist.csv").exists()
    expected_cols = {
        "Ticker",
        "Company",
        "CIK",
        "Venue",
        "Price($)",
        "Cap($M)",
        "ADV20(k)",
        "CatalystType",
        "EventDate",
        "PrimaryCatalystDate",
        "PrimaryCatalystType",
        "PrimaryCatalystTier",
        "PrimaryCatalystURL",
        "PrimarySource",
        "SecondarySource",
        "NotesStatus",
    }
    assert expected_cols.issubset(shortlist.columns)

    aaa_row = shortlist.set_index("Ticker").loc["AAA"]
    assert aaa_row["EventDate"] == "2024-06-02"
    assert aaa_row["EventTier"] == "Tier-1"
    assert aaa_row["PrimaryCatalystDate"] == "2024-06-02"
    assert aaa_row["PrimaryCatalystType"] == "ContractAward"
    assert aaa_row["PrimaryCatalystTier"] == "Tier-1"
    assert aaa_row["PrimaryCatalystURL"].startswith("https://www.sec.gov")
    assert aaa_row["NotesStatus"] == "Pass"

    bbb_row = shortlist.set_index("Ticker").loc["BBB"]
    assert bbb_row["NotesStatus"] == "TBD — exclude"

    assert len(shortlist) <= 40
