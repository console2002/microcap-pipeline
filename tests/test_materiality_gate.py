import logging

import pandas as pd

from app.weekly_deep_research import _materiality_passed, is_material_catalyst, run_weekly_deep_research


def test_is_material_catalyst_rules():
    assert is_material_catalyst("FDAMilestone", 1, {})
    assert not is_material_catalyst("AnnualMeetingResults", 1, {})
    assert not is_material_catalyst("FDAMilestone", 2, {})


def test_materiality_pass_labels():
    assert _materiality_passed("PASS - Tier1 catalyst")
    assert _materiality_passed("pass-high")
    assert _materiality_passed("pass_med")
    assert not _materiality_passed("maybe")
    assert not _materiality_passed("fail-low")


def test_materiality_gate_integration(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    def _fake_runway(url: str, adapter=None, return_reason: bool = False):
        if return_reason:
            return 4.0, True, "OK", ""
        return 4.0, True

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)

    def _write_csv(path, rows):
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    _write_csv(
        tmp_path / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "Sector": "Healthcare",
                "Industry": "Biotechnology",
                "Price": 5.0,
                "MarketCap": 150_000_000,
                "ADV20": 100000,
            },
            {
                "Ticker": "BBB",
                "Company": "Bravo",
                "CIK": "0000000002",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 5.0,
                "MarketCap": 150_000_000,
                "ADV20": 100000,
            },
        ],
    )

    _write_csv(
        tmp_path / "02_filings.csv",
        [
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Form": "10-Q",
                "RunwayQuarters": 4.0,
                "URL": "https://www.sec.gov/Archives/aaa-10q",
            },
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Form": "S-3",
                "FilingDate": "2024-05-01",
                "URL": "https://www.sec.gov/Archives/aaa-s3",
            },
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Form": "DEF 14A",
                "FilingDate": "2024-04-15",
                "URL": "https://www.sec.gov/Archives/aaa-def14a",
            },
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Form": "4",
                "FilingDate": "2024-05-03",
                "URL": "https://www.sec.gov/Archives/aaa-form4",
            },
            {
                "Ticker": "BBB",
                "CIK": "0000000002",
                "Form": "10-Q",
                "RunwayQuarters": 4.0,
                "URL": "https://www.sec.gov/Archives/bbb-10q",
            },
            {
                "Ticker": "BBB",
                "CIK": "0000000002",
                "Form": "S-3",
                "FilingDate": "2024-05-02",
                "URL": "https://www.sec.gov/Archives/bbb-s3",
            },
            {
                "Ticker": "BBB",
                "CIK": "0000000002",
                "Form": "DEF 14A",
                "FilingDate": "2024-04-16",
                "URL": "https://www.sec.gov/Archives/bbb-def14a",
            },
            {
                "Ticker": "BBB",
                "CIK": "0000000002",
                "Form": "4",
                "FilingDate": "2024-05-04",
                "URL": "https://www.sec.gov/Archives/bbb-form4",
            },
        ],
    )

    _write_csv(
        tmp_path / "09_events.csv",
        [
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Tier": "Tier-1",
                "EventDate": "2024-04-01",
                "EventType": "FDAMilestone",
                "URL": "https://www.sec.gov/Archives/event-aaa",
            },
            {
                "Ticker": "BBB",
                "CIK": "0000000002",
                "Tier": "Tier-1",
                "EventDate": "2024-04-02",
                "EventType": "AnnualMeetingResults",
                "URL": "https://www.sec.gov/Archives/event-bbb",
            },
        ],
    )

    dr_df = run_weekly_deep_research(str(tmp_path))

    assert len(dr_df) == 2
    gate_values = {row["Ticker"]: row["Materiality"].startswith("PASS") for _, row in dr_df.iterrows()}
    assert gate_values["AAA"] is True
    assert gate_values["BBB"] is False
