from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from parse import k8 as parse_k8
from app import build_watchlist


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "8k"


def load_fixture(name: str) -> tuple[Path, str]:
    path = FIXTURES_DIR / name
    return path, path.read_text(encoding="utf-8")


def test_k8_parser_tier1_contract():
    path, html = load_fixture("tier1_contract_101.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    items = result["items"]
    codes = {item["item"] for item in items}
    assert "1.01" in codes
    classification = result["classification"]
    assert classification["is_catalyst"] is True
    assert classification["tier"] == "Tier-1"
    assert classification["tier1_type"] == "FUNDED_AWARD"
    trigger = classification["tier1_trigger"]
    assert trigger and "25" in trigger


def test_k8_parser_guidance_up():
    path, html = load_fixture("guidance_up_202.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    items = {item["item"] for item in result["items"]}
    assert "2.02" in items
    classification = result["classification"]
    assert classification["is_catalyst"] is True
    assert classification["tier"] == "Tier-1"
    assert classification["tier1_type"] == "GUIDANCE_UP"


def test_k8_parser_reg_fd_ignore():
    path, html = load_fixture("reg_fd_701.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    classification = result["classification"]
    assert classification["is_catalyst"] is False
    assert classification["ignore_reason"] == "Reg FD only"


def test_k8_parser_atm_dilution():
    path, html = load_fixture("atm_activation_801.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    classification = result["classification"]
    assert classification["is_dilution"] is True
    tags = classification["dilution_tags"]
    assert "ATM" in tags
    assert "SHELF" in tags
    assert classification["is_catalyst"] is False


def test_k8_parser_nonbinding_loi():
    path, html = load_fixture("nonbinding_loi_801.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    classification = result["classification"]
    assert classification["is_catalyst"] is False
    assert classification["ignore_reason"] == "Non-binding LOI"


def test_k8_parser_plain_text_items():
    path, text = load_fixture("plain_item_801.txt")
    result = parse_k8.parse(path.as_uri(), html=text)
    items = {item["item"] for item in result["items"]}
    assert "8.01" in items
    assert result["exhibits"] == []


def test_k8_parser_ix_viewer_exhibits():
    path, html = load_fixture("ix_viewer_item_502.htm")
    base_url = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1234567/0000000000-24-000001/ix_viewer_item_502.htm"
    with patch.object(parse_k8, "_fetch_exhibit_text", return_value="press release text") as mocked_fetch:
        result = parse_k8.parse(base_url, html=html)
    items = {item["item"] for item in result["items"]}
    assert "5.02" in items
    assert result["exhibits"]
    assert mocked_fetch.called


def test_k8_parser_additional_items_detected():
    path, html = load_fixture("items_302_901.htm")
    result = parse_k8.parse(path.as_uri(), html=html)
    items = {item["item"] for item in result["items"]}
    assert {"3.02", "9.01"}.issubset(items)


def test_k8_watchlist_integration(tmp_path):
    data_dir = Path(tmp_path)
    tier1_path, _ = load_fixture("tier1_contract_101.htm")
    filings_df = pd.DataFrame(
        [
            {
                "CIK": "0000123456",
                "Ticker": "ACME",
                "Form": "8-K",
                "FiledAt": "2025-01-15",
                "URL": tier1_path.as_uri(),
            }
        ]
    )
    filings_df.to_csv(data_dir / "filings.csv", index=False)

    research_rows = [
        {
            "Ticker": "ACME",
            "CIK": "123456",
            "Company": "Acme Corp",
            "Sector": "Industrials",
            "Price": 5.25,
            "MarketCap": 150_000_000,
            "ADV20": 60_000,
            "RunwayQuartersRaw": 3.0,
            "RunwayCashRaw": 6_000_000,
            "RunwayCash": 6_000_000,
            "RunwayQuarterlyBurnRaw": 2_000_000,
            "RunwayQuarterlyBurn": 2_000_000,
            "RunwayEstimate": "interim",
            "RunwayNotes": "values parsed",
            "RunwaySourceForm": "10-Q",
            "RunwaySourceDate": "2025-01-01",
            "RunwaySourceURL": "https://example.com/10q",
            "Catalyst": f"2025-01-15 | 8-K | {tier1_path.as_uri()}",
            "Dilution": "2025-01-01 | S-3 | https://example.com/s3",
            "Governance": "2024-12-31 | 10-K | https://example.com/10k",
            "FilingsSummary": f"2025-01-15 | 8-K | {tier1_path.as_uri()}",
            "Insider": "2025-01-05 | Form 4 | https://example.com/form4",
            "Ownership": "2025-01-10 | SC 13G | https://example.com/13g",
            "Status": "Active",
            "InsiderStatus": "Pass",
            "OwnershipStatus": "Pass",
        }
    ]
    pd.DataFrame(research_rows).to_csv(data_dir / "05_dr_populate_results.csv", index=False)

    rows_written, status = build_watchlist.run(data_dir=str(data_dir))
    assert status == "ok"
    assert rows_written == 1

    events_df = pd.read_csv(data_dir / "8k_events.csv")
    assert len(events_df) == 1
    event_row = events_df.iloc[0]
    assert bool(event_row["IsCatalyst"]) is True
    assert event_row["CatalystType"] == "Tier-1"
    assert event_row["Tier1Type"] == "FUNDED_AWARD"

    validated = pd.read_csv(data_dir / "validated_watchlist.csv")
    assert len(validated) == 1
    validated_row = validated.iloc[0]
    assert validated_row["CatalystType"] == "Tier-1"
    assert validated_row["Catalyst"] == "Tier-1 FUNDED_AWARD"
    assert validated_row["CatalystPrimaryForm"] == "8-K"
    assert validated_row["CatalystPrimaryDate"] == "2025-01-15"
    assert validated_row["Tier1Type"] == "FUNDED_AWARD"
    assert "25" in str(validated_row["Tier1Trigger"])
    assert bool(validated_row["DilutionFlag"]) is True
    assert validated_row["DilutionStatus"] == "Pass (Offering)"
