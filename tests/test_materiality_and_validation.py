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

    tier1 = _materiality(4, "Tier-1", True)
    assert tier1.startswith("PASS")
    assert _materiality_passed(tier1)
    assert _conviction_from_subscores(4, "Tier-1", tier1) == "High"

    tier2 = _materiality(4, "Tier-2", True)
    assert tier2.startswith("PASS")
    assert _conviction_from_subscores(4, "Tier-2", tier2) == "Medium"


def test_runway_reuse_and_materiality_output(tmp_path, caplog):
    data_dir = tmp_path
    caplog.set_level(logging.INFO)

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
    assert row["Runway (qtrs)"] == "3.5"
    assert row["RunwayEvidencePrimary"] == "https://example.com/aaa-10q"
    assert row["Materiality"].startswith("PASS") or row["Materiality"].startswith("FAIL")
    assert row["ConvictionScore"] in {"High", "Medium", "Low"}


def test_validation_reasons_and_pass_fail():
    base_row = {
        "Ticker": "AAA",
        "CIK": "0000000001",
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
    missing_runway["Runway (qtrs)"] = "TBD"
    status, reason = evaluate_validation(pd.Series(missing_runway))
    assert status.startswith("TBD")
    assert "Mandatory subscore missing: Runway" in reason

    weak_materiality = base_row.copy()
    weak_materiality["Materiality"] = "FAIL - weak profile"
    status, reason = evaluate_validation(pd.Series(weak_materiality))
    assert status.startswith("TBD")
    assert "Materiality fail" in reason
