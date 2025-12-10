import logging

import pandas as pd

from app.weekly_deep_research import (
    _conviction_from_subscores,
    _materiality,
    _materiality_passed,
    run_weekly_deep_research,
)
from app.weekly_validated import evaluate_validation


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_materiality_and_conviction_rules():
    missing_mat = _materiality(3, "Tier-1", False)
    assert missing_mat.startswith("FAIL")
    assert not _materiality_passed(missing_mat)
    assert _conviction_from_subscores(5, "Tier-1", missing_mat) == "Low"

    tier1 = _materiality(4, "Tier-1", True, event_type="FDAMilestone", event_tier=1)
    assert tier1.startswith("PASS")
    assert _materiality_passed(tier1)
    assert _conviction_from_subscores(4, "Tier-1", tier1) == "High"

    tier2 = _materiality(4, "Tier-2", True, event_type="Earnings", event_tier=2)
    assert tier2.startswith("FAIL")
    assert _conviction_from_subscores(4, "Tier-2", tier2) == "Low"


def test_runway_reuse_and_materiality_output(tmp_path, caplog, monkeypatch):
    data_dir = tmp_path
    caplog.set_level(logging.INFO)

    def _fake_runway(url: str, adapter=None, return_reason: bool = False, include_reason_meta: bool = False, **kwargs):
        if return_reason:
            base = (3.5, True, "OK", "")
            if include_reason_meta:
                return (*base, {})
            return base
        return 3.5, True

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "Sector": "Healthcare",
                "Industry": "Biotechnology",
                "Price": 10.0,
                "MarketCap": 100_000_000,
                "ADV20": 100000,
            }
        ],
    )

    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Form": "10-Q",
                "RunwayQuarters": 3.5,
                "URL": "https://example.com/aaa-10q",
            }
        ],
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "AAA",
                "CIK": "0000000001",
                "Tier": "Tier-1",
                "EventDate": "2024-05-01",
                "EventType": "Trial",
                "URL": "https://example.com/event",
            }
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))

    assert "WEEKLY_RUNWAY" in " ".join(record.getMessage() for record in caplog.records)
    assert len(dr_df) == 1
    row = dr_df.iloc[0]
    assert row["RunwayQuarters"] == 3.5
    assert row["Runway (qtrs)"] == "3.50"
    assert row["RunwayEvidencePrimary"] == "https://example.com/aaa-10q"
    assert row["Materiality"].startswith("PASS") or row["Materiality"].startswith("FAIL")
    assert row["ConvictionScore"] in {"High", "Medium", "Low"}


def test_validation_reasons_and_pass_fail():
    base_row = {
        "Ticker": "AAA",
        "CIK": "0000000001",
        "Price": 2.0,
        "ADV20": 50_000,
        "MarketCap": 200_000_000,
        "RunwayQuarters": 3.0,
        "Runway (qtrs)": "3.0",
        "RunwayEvidencePrimary": "https://example.com/runway",
        "Dilution": "High",
        "DilutionEvidencePrimary": "https://example.com/dilution",
        "Catalyst": "Tier-1",
        "CatalystEvidencePrimary": "https://example.com/catalyst",
        "SubscoresEvidencedCount": 4,
        "Materiality": "PASS - Tier1 catalyst",
        "BiotechPeerRead": "Y:peer",
    }

    status, reason = evaluate_validation(pd.Series(base_row))
    assert status == "Validated"
    assert reason == ""

    missing_runway = base_row.copy()
    missing_runway["RunwayQuarters"] = None
    missing_runway["Runway (qtrs)"] = ""
    status, reason = evaluate_validation(pd.Series(missing_runway))
    assert status.startswith("TBD")
    assert "Runway missing/invalid" in reason

    weak_materiality = base_row.copy()
    weak_materiality["Materiality"] = "FAIL - weak profile"
    status, reason = evaluate_validation(pd.Series(weak_materiality))
    assert status.startswith("TBD")
    assert "Materiality fail" in reason


def test_missing_mandatory_subscore_sets_tbd(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    def _fake_runway(url: str, adapter=None, return_reason: bool = False, include_reason_meta: bool = False, **kwargs):
        if return_reason:
            base = (4.0, True, "OK", "")
            if include_reason_meta:
                return (*base, {})
            return base
        return 4.0, True

    def _fake_dilution_details(filings, form_col):
        tickers = set(filings.get("Ticker", []))
        ticker = next(iter(tickers)) if tickers else ""
        has_evidence = ticker == "BBB"
        evidence = "https://sec.gov/dilution" if has_evidence else ""
        return {
            "score": "Low",
            "evidence": evidence,
            "key_url": evidence,
            "last_event_date": "",
            "classification": "OVERHANG_CREATION",
        }

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)
    monkeypatch.setattr("app.weekly_deep_research._dilution_details", _fake_dilution_details)

    _write_csv(
        tmp_path / "20_candidate_shortlist.csv",
        [
            {"Ticker": "AAA", "Company": "Alpha", "CIK": "1", "Sector": "Healthcare", "Industry": "Biotechnology", "Price": 5.0, "MarketCap": 150_000_000, "ADV20": 80_000},
            {"Ticker": "BBB", "Company": "Beta", "CIK": "2", "Sector": "Technology", "Industry": "Software", "Price": 6.0, "MarketCap": 120_000_000, "ADV20": 90_000},
        ],
    )

    _write_csv(
        tmp_path / "02_filings.csv",
        [
            {"Ticker": "AAA", "CIK": "1", "Form": "10-Q", "RunwayQuarters": 4.0, "URL": "https://sec.gov/runway_aaa", "FilingDate": "2024-05-01"},
            {"Ticker": "AAA", "CIK": "1", "Form": "DEF 14A", "FilingDate": "2024-04-01", "URL": "https://sec.gov/def14a_aaa"},
            {"Ticker": "AAA", "CIK": "1", "Form": "4", "FilingDate": "2024-04-15", "URL": "https://sec.gov/form4_aaa"},
            {"Ticker": "BBB", "CIK": "2", "Form": "10-Q", "RunwayQuarters": 4.0, "URL": "https://sec.gov/runway_bbb", "FilingDate": "2024-05-02"},
            {"Ticker": "BBB", "CIK": "2", "Form": "DEF 14A", "FilingDate": "2024-04-02", "URL": "https://sec.gov/def14a_bbb"},
            {"Ticker": "BBB", "CIK": "2", "Form": "4", "FilingDate": "2024-04-16", "URL": "https://sec.gov/form4_bbb"},
        ],
    )

    _write_csv(
        tmp_path / "09_events.csv",
        [
            {"Ticker": "AAA", "CIK": "1", "Tier": "Tier-1", "EventDate": "2024-05-10", "EventType": "FDAMilestone", "URL": "https://sec.gov/event_aaa"},
            {"Ticker": "BBB", "CIK": "2", "Tier": "Tier-1", "EventDate": "2024-05-11", "EventType": "FDAMilestone", "URL": "https://sec.gov/event_bbb"},
        ],
    )

    dr_df = run_weekly_deep_research(str(tmp_path))

    statuses = dict(zip(dr_df["Ticker"], dr_df["Status"]))
    missing = dict(zip(dr_df["Ticker"], dr_df.get("MissingMandatorySubscores", "")))

    assert statuses["AAA"].startswith("TBD")
    assert "Dilution" in missing["AAA"]
    assert statuses["BBB"] == "Validated"
    assert missing["BBB"] in {"", None}
