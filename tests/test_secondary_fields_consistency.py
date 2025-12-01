import os
import csv

import pandas as pd

from app.build_watchlist import _write_canonical_events
from app.utils import ensure_csv
from app.weekly_deep_research import _aggregate_evidence, _classify_evidence
from app.weekly_validated import build_validated_selections


def test_events_secondary_fields_normalized(tmp_path):
    events_df = pd.DataFrame(
        [
            {
                "Ticker": "ABC",
                "Company": "Alpha",
                "CIK": "0001",
                "EventDate": "2024-01-01",
                "EventType": "Merger",
                "EventTier": "Tier-1",
                "PrimarySourceURL": "http://primary.example.com",
                "SecondarySourceURL": None,
                "HomepageURL": None,
            }
        ]
    )

    output_df = _write_canonical_events(events_df, tmp_path)

    assert "SecondarySourceURL" in output_df.columns
    assert "SecondarySource" in output_df.columns
    assert output_df.loc[0, "SecondarySourceURL"] == ""
    assert output_df.loc[0, "SecondarySource"] == ""


def test_deep_research_secondary_fields(tmp_path):
    # Mimic evidence processing branch to ensure empty secondary evidence normalizes to "".
    primary_links, secondary_links = _classify_evidence([None, "", "http://news.example.com"])
    # Only the non-SEC URL should land in secondary links.
    assert secondary_links == ["http://news.example.com"]

    evidence_secondary = _aggregate_evidence([])
    assert evidence_secondary == ""

    output_path = os.path.join(tmp_path, "30_deep_research.csv")
    fieldnames = [
        "Ticker",
        "EvidenceSecondary",
        "Evidence (Secondary links)",
    ]
    ensure_csv(output_path, fieldnames)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Ticker": "ABC", "EvidenceSecondary": evidence_secondary, "Evidence (Secondary links)": evidence_secondary})

    df = pd.read_csv(output_path, keep_default_na=False)
    assert df.loc[0, "EvidenceSecondary"] == ""
    assert df.loc[0, "Evidence (Secondary links)"] == ""


def test_validated_selections_status_and_secondary(tmp_path):
    data_dir = tmp_path

    # Create deep research input with all gates satisfied.
    deep_rows = pd.DataFrame(
        [
            {
                "Ticker": "XYZ",
                "Company": "Xylo Corp",
                "CIK": "1000",
                "Sector": "Tech",
                "Industry": "Software",
                "Venue": "NASDAQ",
                "Price": 10.5,
                "MarketCap": 250.0,
                "ADV20": 100.0,
                "RunwayQuarters": 4,
                "Runway (qtrs)": 4,
                "DilutionScore": "Low",
                "CatalystScore": "High",
                "GovernanceScore": "OK",
                "InsiderScore": "TBD",
                "BiotechPeerRead": "N",
                "Subscores Evidenced (x/5)": 5,
                "Materiality": "PASS - demo",
                "ConvictionScore": "High",
                "PrimaryCatalystDate": "2024-06-01",
                "PrimaryCatalystType": "Approval",
                "PrimaryCatalystURL": "http://primary.example.com",
                "RunwayEvidencePrimary": "http://sec.example.com/runway",
                "DilutionEvidencePrimary": "http://sec.example.com/dilution",
                "CatalystEvidencePrimary": "http://sec.example.com/catalyst",
                "Evidence (Primary links)": "http://sec.example.com/primary",
                "Evidence (Secondary links)": "",
                "EvidenceSecondary": "",
                "Materiality (pass/fail + note)": "PASS - demo",
                "Status": "Validated",
            }
        ]
    )
    deep_rows.to_csv(os.path.join(data_dir, "30_deep_research.csv"), index=False)

    # Optional source files may be empty but must exist for merge logic.
    for filename in ["01_universe_gated.csv", "20_candidate_shortlist.csv"]:
        ensure_csv(os.path.join(data_dir, filename), ["Ticker", "CIK"])

    validated_df, _ = build_validated_selections(data_dir=data_dir)

    assert not validated_df.empty
    row = validated_df.iloc[0]
    assert row["Status"]
    assert row["ValidationStatus"] == row["Status"]
    assert "SecondarySource" in validated_df.columns
    assert row["SecondarySource"] == ""
