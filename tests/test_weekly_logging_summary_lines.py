import pandas as pd

from app.weekly_validated import build_validated_selections


def test_weekly_summary_and_validation_logs(tmp_path):
    data_dir = tmp_path

    pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "Sector": "Tech",
                "Industry": "Software",
            }
        ]
    ).to_csv(data_dir / "01_universe_gated.csv", index=False)

    pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "Venue": "NASDAQ",
                "Price": 5.5,
                "MarketCap": 150_000_000,
                "ADV20": 100_000,
                "PrimaryCatalystDate": "2024-07-01",
                "PrimaryCatalystType": "ContractAward",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": "https://www.sec.gov/aaa",
            }
        ]
    ).to_csv(data_dir / "20_candidate_shortlist.csv", index=False)

    pd.DataFrame().to_csv(data_dir / "02_filings.csv", index=False)
    pd.DataFrame().to_csv(data_dir / "09_events.csv", index=False)

    pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Company": "Alpha",
                "CIK": "0000000001",
                "Sector": "Tech",
                "Industry": "Software",
                "RunwayQuarters": 5,
                "Runway (qtrs)": 5,
                "RunwayEvidencePrimary": "https://www.sec.gov/runway",
                "Dilution": "Low",
                "DilutionScore": "Low",
                "DilutionEvidencePrimary": "https://www.sec.gov/dilution",
                "Catalyst": "Tier-1",
                "CatalystScore": "Tier-1",
                "CatalystEvidencePrimary": "https://www.sec.gov/catalyst",
                "GovernanceScore": "OK",
                "InsiderScore": "Strong",
                "BiotechPeerRead": "N",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - catalyst",
                "ConvictionScore": "High",
                "PrimaryCatalystDate": "2024-07-01",
                "PrimaryCatalystType": "ContractAward",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": "https://www.sec.gov/aaa",
            }
        ]
    ).to_csv(data_dir / "30_deep_research.csv", index=False)

    progress_messages: list[str] = []

    validated, exclusions = build_validated_selections(
        str(data_dir), progress_fn=progress_messages.append
    )

    assert not validated.empty
    assert any("WEEKLY_SUMMARY" in msg for msg in progress_messages)
    assert any("WEEKLY_VALIDATION" in msg for msg in progress_messages)
