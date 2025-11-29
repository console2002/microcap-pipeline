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
                "FilingURL": "https://www.sec.gov/Archives/edgar/data/0000000001/abc-s3.htm",
            },
            {
                "Ticker": "XYZ",
                "CIK": "0000000002",
                "FormType": "8-K",
                "FilingURL": "https://www.sec.gov/Archives/edgar/data/0000000002/xyz-8k.htm",
            },
        ],
    )

    # events
    _write_csv(
        data_dir / "09_events.csv",
        [
            {"Ticker": "ABC", "CIK": "0000000001", "Tier": "Tier-1", "FilingURL": "https://www.sec.gov/abc"},
            {"Ticker": "XYZ", "CIK": "0000000002", "Tier": "Tier-2", "FilingURL": "https://www.sec.gov/xyz"},
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
    required_cols = {
        "Dilution",
        "Runway (qtrs)",
        "Catalyst",
        "Governance",
        "Insider",
        "Evidence (Primary links)",
        "Evidence (Secondary links)",
        "Biotech Peer Read-Through (Y/N + link)",
        "Subscores Evidenced (x/5)",
        "Materiality (pass/fail + note)",
        "Status",
    }
    assert required_cols.issubset(dr_df.columns)
    assert abc_row["RunwayQuarters"] == 4.0
    assert abc_row["Dilution"] == "High"
    assert abc_row["Catalyst"].startswith("Tier-1")
    assert abc_row["Biotech Peer Read-Through (Y/N + link)"].startswith("Y")
    assert abc_row["Evidence (Primary links)"]
    assert abc_row["Subscores Evidenced (x/5)"] >= 4
    assert abc_row["Status"] == "Validated"

    validated, exclusions = build_validated_selections(str(data_dir))
    assert (data_dir / "40_validated_selections.csv").exists()
    assert len(validated) == 1
    assert validated.iloc[0]["Ticker"] == "ABC"
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["Ticker"] == "XYZ"
    assert set(validated.columns) >= {"Validation status", "Catalyst rationale"}
