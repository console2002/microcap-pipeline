import os
import pandas as pd

from app.weekly_deep_research import run_weekly_deep_research
from app.weekly_validated import build_validated_selections


def _write_csv(path, rows):
    if not rows:
        raise ValueError("rows required")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_w3_deep_research_and_w4(tmp_path):
    data_dir = tmp_path
    # create candidate shortlist
    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "ABC",
                "Company": "Alpha Beta",
                "CIK": "0000000001",
                "Sector": "Healthcare",
                "Industry": "Biotechnology",
                "Price": 5.0,
                "MarketCap": 100_000_000,
                "ADV20": 50000,
                "PrimaryCatalystType": "Earnings",
                "PrimaryCatalystDate": "2024-05-01",
                "PrimaryFilingURL": "file://" + str((tmp_path / "filing.html")),
            },
            {
                "Ticker": "XYZ",
                "Company": "Xylophone",
                "CIK": "0000000002",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 2.5,
                "MarketCap": 50_000_000,
                "ADV20": 80000,
                "PrimaryCatalystType": "Contract",
                "PrimaryCatalystDate": "2024-05-02",
                "PrimaryFilingURL": "",
            },
        ],
    )

    # filings
    filing_path = tmp_path / "filing.html"
    filing_path.write_text(
        "Cash and cash equivalents: $12,000,000\nNet cash used in operating activities: $(3,000,000)",
        encoding="utf-8",
    )
    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "ABC",
                "CIK": "0000000001",
                "FormType": "10-Q",
                "FilingURL": f"file://{filing_path}",
            },
            {
                "Ticker": "ABC",
                "CIK": "0000000001",
                "FormType": "S-3",
                "FilingURL": "https://example.com/s3",
            },
            {
                "Ticker": "XYZ",
                "CIK": "0000000002",
                "FormType": "8-K",
                "FilingURL": "https://example.com/8k",
            },
        ],
    )

    # events
    _write_csv(
        data_dir / "09_events.csv",
        [
            {"Ticker": "ABC", "CIK": "0000000001", "Tier": "Tier-1", "FilingURL": "https://example.com/abc"},
            {"Ticker": "XYZ", "CIK": "0000000002", "Tier": "Tier-2", "FilingURL": "https://example.com/xyz"},
        ],
    )

    # universe
    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {"Ticker": "ABC", "CIK": "0000000001", "Sector": "Healthcare", "Industry": "Biotechnology"},
            {"Ticker": "XYZ", "CIK": "0000000002", "Sector": "Technology", "Industry": "Software"},
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))
    assert (data_dir / "30_deep_research.csv").exists()
    assert len(dr_df) == 2
    abc_row = dr_df.set_index("Ticker").loc["ABC"]
    assert abc_row["RunwayQuarters"] == 4.0
    assert abc_row["DilutionScore"] == "High"
    assert abc_row["CatalystScore"] == "Tier-1"
    assert abc_row["BiotechPeerRead"] == "Y"
    assert abc_row["RunwayEvidencePrimary"]
    assert abc_row["DilutionEvidencePrimary"]
    assert abc_row["CatalystEvidencePrimary"]

    validated, exclusions = build_validated_selections(str(data_dir))
    assert (data_dir / "40_validated_selections.csv").exists()
    assert len(validated) == 1
    assert validated.iloc[0]["Ticker"] == "ABC"
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["Ticker"] == "XYZ"
