from pathlib import Path
import sys
from unittest.mock import patch

from types import SimpleNamespace

import pandas as pd

from app.csv_names import csv_filename

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
    filings_df.to_csv(data_dir / csv_filename("filings"), index=False)

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
    pd.DataFrame(research_rows).to_csv(data_dir / csv_filename("dr_populate_results"), index=False)

    rows_written, status = build_watchlist.run(data_dir=str(data_dir))
    assert status == "ok"
    assert rows_written == 1

    events_df = pd.read_csv(data_dir / csv_filename("eight_k_events"))
    assert len(events_df) == 1
    event_row = events_df.iloc[0]
    assert bool(event_row["IsCatalyst"]) is True
    assert event_row["CatalystType"] == "Tier-1"
    assert event_row["Tier1Type"] == "FUNDED_AWARD"

    validated = pd.read_csv(data_dir / csv_filename("validated_watchlist"))
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


def test_generate_eight_k_events_logs_progress(monkeypatch, tmp_path):
    progress_messages: list[str] = []

    def record_progress(message: str) -> None:
        progress_messages.append(message)

    def fake_process(row):
        url = getattr(row, "URL", "")
        event = build_watchlist.EightKEvent(
            cik=str(getattr(row, "CIK", "")),
            ticker=str(getattr(row, "Ticker", "")),
            filing_date=str(getattr(row, "FiledAt", "")),
            filing_url=url,
            items_present="1.01",
            is_catalyst=True,
            catalyst_type="Tier-1",
            catalyst_label="Tier-1 FUNDED_AWARD",
            tier1_type="FUNDED_AWARD",
            tier1_trigger="Award signed",
            is_dilution=False,
            dilution_tags=[],
            ignore_reason="",
        )
        csv_row = {
            "CIK": event.cik,
            "Ticker": event.ticker,
            "FilingDate": event.filing_date,
            "FilingURL": event.filing_url,
            "ItemsPresent": event.items_present,
            "IsCatalyst": event.is_catalyst,
            "CatalystType": event.catalyst_type,
            "Tier1Type": event.tier1_type,
            "Tier1Trigger": event.tier1_trigger,
            "IsDilution": event.is_dilution,
            "DilutionTags": event.dilution_tags_joined(),
            "IgnoreReason": event.ignore_reason,
        }
        return build_watchlist._EightKProcessResult(
            url=url,
            event=event,
            csv_row=csv_row,
            debug_entry=None,
            log_messages=[],
        )

    monkeypatch.setattr(build_watchlist, "_process_eight_k_row", fake_process)

    filings_df = pd.DataFrame(
        [
            {
                "CIK": "0000001",
                "Ticker": "ABC",
                "Form": "8-K",
                "FiledAt": "2025-01-01",
                "URL": "https://example.com/8k1",
            },
            {
                "CIK": "0000002",
                "Ticker": "XYZ",
                "Form": "8-K",
                "FiledAt": "2025-01-02",
                "URL": "https://example.com/8k2",
            },
        ]
    )
    filings_df.to_csv(tmp_path / csv_filename("filings"), index=False)

    events_df, _ = build_watchlist._generate_eight_k_events(
        data_dir=str(tmp_path), progress_fn=record_progress
    )

    assert len(events_df) == 2
    assert any("eight_k:" in msg and "processing" in msg for msg in progress_messages)
    assert any("eight_k:" in msg and "complete" in msg for msg in progress_messages)


def test_process_eight_k_row_fetch_failure(tmp_path):
    row = SimpleNamespace(
        URL="file:///nonexistent_8k.htm",
        Ticker="MISS",
        CIK="0000003",
        FiledAt="2025-02-01",
    )

    result = build_watchlist._process_eight_k_row(row)

    assert result.debug_entry is not None
    assert any("fetch_failed" in msg and "nonexistent_8k.htm" in msg for msg in result.log_messages)


def test_process_eight_k_row_no_actionable_items(monkeypatch):
    row = SimpleNamespace(
        URL="https://example.com/8k_missing.htm",
        Ticker="MISS",
        CIK="0000004",
        FiledAt="2025-02-02",
        Form="8-K",
    )

    monkeypatch.setattr(build_watchlist, "_read_eight_k_html", lambda url: ("<html></html>", None))
    monkeypatch.setattr(
        build_watchlist.parse_k8,
        "parse",
        lambda url, html, form_hint=None: {"items": [], "classification": {}, "exhibits": []},
    )

    result = build_watchlist._process_eight_k_row(row)

    assert result.event is None
    assert result.debug_entry is not None
    assert any("no actionable items" in msg and "8k_missing" in msg for msg in result.log_messages)


def test_process_eight_k_row_logs_pass_details(monkeypatch):
    row = SimpleNamespace(
        URL="https://example.com/8k_success.htm",
        Ticker="WINN",
        CIK="0000005",
        FiledAt="2025-02-03",
        Form="8-K",
    )

    monkeypatch.setattr(build_watchlist, "_read_eight_k_html", lambda url: ("<html></html>", None))
    monkeypatch.setattr(
        build_watchlist.parse_k8,
        "parse",
        lambda url, html, form_hint=None: {
            "items": [{"item": "8.01"}],
            "classification": {
                "is_catalyst": True,
                "tier": "Tier-1",
                "tier1_type": "Regulatory",
                "tier1_trigger": "PDUFA",
                "is_dilution": False,
                "dilution_tags": [],
                "ignore_reason": "",
            },
            "exhibits": [],
        },
    )

    result = build_watchlist._process_eight_k_row(row)

    assert result.event is not None
    assert any("parsed url https://example.com/8k_success.htm" in msg for msg in result.log_messages)
    assert any("catalyst Tier-1" in msg for msg in result.log_messages)
    assert any("dilution no" in msg for msg in result.log_messages)
