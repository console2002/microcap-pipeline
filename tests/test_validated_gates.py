import pandas as pd

import app.weekly_validated as weekly_validated
from app.weekly_validated import build_validated_selections


def test_validation_gates_cover_universe_mandatory_and_biotech(tmp_path, monkeypatch):
    data_dir = tmp_path

    monkeypatch.setattr(weekly_validated, "BIOTECH_PEER_REQUIRED_FOR_VALIDATION", True)

    deep_rows = pd.DataFrame(
        [
            {
                "Ticker": "PASS",
                "Company": "Valid Corp",
                "CIK": "0001",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 3.5,
                "MarketCap": 150_000_000,
                "ADV20": 75_000,
                "RunwayQuarters": 6,
                "Runway (qtrs)": 6,
                "RunwayEvidencePrimary": "http://sec.example.com/runway",
                "DilutionScore": "Low",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution",
                "CatalystScore": "Tier-1",
                "PrimaryCatalystType": "Earnings",
                "CatalystEvidencePrimary": "http://sec.example.com/catalyst",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - ok",
                "BiotechPeerRead": "N",
                "Evidence (Primary links)": "http://sec.example.com/primary",
                "Evidence (Secondary links)": "",
                "EvidenceSecondary": "",
                "Status": "Validated",
            },
            {
                "Ticker": "UNIVFAIL",
                "Company": "Cheap Co",
                "CIK": "0002",
                "Sector": "Industrials",
                "Industry": "Tools",
                "Price": 0.5,
                "MarketCap": 50_000_000,
                "ADV20": 100_000,
                "RunwayQuarters": 5,
                "RunwayEvidencePrimary": "http://sec.example.com/runway2",
                "DilutionScore": "Low",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution2",
                "CatalystScore": "Tier-1",
                "PrimaryCatalystType": "Contract",
                "CatalystEvidencePrimary": "http://sec.example.com/catalyst2",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - ok",
                "BiotechPeerRead": "N",
                "Status": "Validated",
            },
            {
                "Ticker": "MANDFAIL",
                "Company": "Missing Evidence",
                "CIK": "0003",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 2.0,
                "MarketCap": 120_000_000,
                "ADV20": 90_000,
                "RunwayQuarters": 4,
                "RunwayEvidencePrimary": "http://sec.example.com/runway3",
                "DilutionScore": "Low",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution3",
                "CatalystScore": "Tier-1",
                "PrimaryCatalystType": "Contract",
                "CatalystEvidencePrimary": "",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - ok",
                "BiotechPeerRead": "N",
                "Status": "Validated",
            },
            {
                "Ticker": "BIOFAIL",
                "Company": "BioTech Inc",
                "CIK": "0004",
                "Sector": "Healthcare",
                "Industry": "Biotechnology",
                "Price": 4.0,
                "MarketCap": 200_000_000,
                "ADV20": 80_000,
                "RunwayQuarters": 3,
                "RunwayEvidencePrimary": "http://sec.example.com/runway4",
                "DilutionScore": "Medium",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution4",
                "CatalystScore": "Tier-1",
                "PrimaryCatalystType": "Clinical",
                "CatalystEvidencePrimary": "http://sec.example.com/catalyst4",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - ok",
                "BiotechPeerRead": "TBD",
                "Status": "Validated",
            },
            {
                "Ticker": "CAPFAIL",
                "Company": "Large Cap",
                "CIK": "0005",
                "Sector": "Technology",
                "Industry": "Hardware",
                "Price": 3.0,
                "ADV20": 100_000,
                "RunwayQuarters": 5,
                "RunwayEvidencePrimary": "http://sec.example.com/runway5",
                "DilutionScore": "Low",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution5",
                "CatalystScore": "Tier-1",
                "PrimaryCatalystType": "Product",
                "CatalystEvidencePrimary": "http://sec.example.com/catalyst5",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - ok",
                "BiotechPeerRead": "N",
                "Status": "Validated",
            },
        ]
    )
    deep_rows.to_csv(data_dir / "30_deep_research.csv", index=False)

    pd.DataFrame(
        [
            {"Ticker": "PASS", "CIK": "0001", "Price": 3.5, "MarketCap": 150_000_000, "ADV20": 75_000},
            {"Ticker": "UNIVFAIL", "CIK": "0002", "Price": 0.5, "MarketCap": 50_000_000, "ADV20": 100_000},
            {"Ticker": "MANDFAIL", "CIK": "0003", "Price": 2.0, "MarketCap": 120_000_000, "ADV20": 90_000},
            {"Ticker": "BIOFAIL", "CIK": "0004", "Price": 4.0, "MarketCap": 200_000_000, "ADV20": 80_000},
            {"Ticker": "CAPFAIL", "CIK": "0005", "Price": 3.0, "Cap_Musd": 600, "ADV20": 100_000},
        ]
    ).to_csv(data_dir / "01_universe_gated.csv", index=False)

    validated, exclusions = build_validated_selections(data_dir=data_dir)

    assert list(validated["Ticker"]) == ["PASS"]
    assert all(validated[label].iloc[0] for label in validated.columns if label.startswith("GATE_"))

    assert set(exclusions["Ticker"]) == {"UNIVFAIL", "MANDFAIL", "BIOFAIL", "CAPFAIL"}

    reasons = dict(zip(exclusions["Ticker"], exclusions["Reason"]))
    assert "Price<1" in reasons["UNIVFAIL"]
    assert "Catalyst evidence missing" in reasons["MANDFAIL"]
    assert "Biotech peer read missing/failed" in reasons["BIOFAIL"]
    assert "Cap≥400M" in reasons["CAPFAIL"]

    # Secondary evidence remains optional for the passing row.
    assert validated.iloc[0]["SecondarySource"] == ""
