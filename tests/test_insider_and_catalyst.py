import pandas as pd

from app.weekly_deep_research import run_weekly_deep_research


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_insider_and_catalyst_paths(tmp_path):
    data_dir = tmp_path
    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {"Ticker": "INS", "Company": "Insider Corp", "CIK": "111", "Sector": "Tech", "Industry": "Software"},
        ],
    )
    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "INS",
                "CIK": "111",
                "FormType": "4",
                "FilingURL": "https://example.com/form4",
                "FilingDate": "2024-04-01",
            },
            {
                "Ticker": "INS",
                "CIK": "111",
                "FormType": "8-K",
                "FilingURL": "https://example.com/8k",
                "FilingDate": "2024-03-20",
            },
        ],
    )
    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "INS",
                "CIK": "111",
                "Tier": "Tier-2",
                "EventDate": "2024-03-21",
                "EventType": "Clinical",
                "FilingURL": "https://example.com/8k",
            }
        ],
    )

    df = run_weekly_deep_research(str(data_dir))
    assert not df.empty
    row = df.iloc[0]
    assert row["InsiderScore"] == "Strong"
    assert row["LastInsiderBuyDate"] == "2024-04-01"
    assert row["InsiderEvidencePrimary"]
    assert row["CatalystScore"] == "Tier-2"
    assert row["PrimaryCatalystType"] == "Clinical"
    assert row["PrimaryCatalystURL"] == "https://example.com/8k"
