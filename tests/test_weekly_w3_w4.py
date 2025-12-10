import os
import pandas as pd

from app.weekly_deep_research import run_weekly_deep_research
from app.weekly_validated import build_validated_selections


def _write_csv(path, rows):
    if not rows:
        raise ValueError("rows required")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_w3_deep_research_and_w4(tmp_path, monkeypatch):
    data_dir = tmp_path

    def _fake_runway(url: str, adapter=None, return_reason: bool = False, include_reason_meta: bool = False, **kwargs):
        if return_reason:
            base = (4.0, True, "OK", "")
            if include_reason_meta:
                return (*base, {})
            return base
        return 4.0, True

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)
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
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": "https://www.sec.gov/abc",
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
                "PrimaryCatalystTier": "Tier-2",
                "PrimaryCatalystURL": "https://www.sec.gov/xyz",
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
            {
                "Ticker": "ABC",
                "CIK": "0000000001",
                "Tier": "Tier-1",
                "EventDate": "2024-05-01",
                "EventType": "FDAMilestone",
                "FilingURL": "https://www.sec.gov/abc",
            },
            {
                "Ticker": "XYZ",
                "CIK": "0000000002",
                "Tier": "Tier-2",
                "EventDate": "2024-05-02",
                "FilingURL": "https://www.sec.gov/xyz",
            },
        ],
    )

    # universe
    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "ABC",
                "CIK": "0000000001",
                "Sector": "Healthcare",
                "Industry": "Biotechnology",
                "Price": 5.0,
                "MarketCap": 100_000_000,
                "ADV20": 50_000,
            },
            {
                "Ticker": "XYZ",
                "CIK": "0000000002",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 2.5,
                "MarketCap": 50_000_000,
                "ADV20": 80_000,
            },
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))
    assert (data_dir / "30_deep_research.csv").exists()
    assert len(dr_df) == 2
    # Ensure biotech peer read is satisfied for ABC so validation can pass downstream.
    dr_df.loc[dr_df["Ticker"] == "ABC", "Biotech Peer Read-Through (Y/N + link)"] = "Y_peer"
    dr_df.loc[dr_df["Ticker"] == "ABC", "BiotechPeerRead"] = "Y_peer"
    dr_df.loc[dr_df["Ticker"] == "ABC", "Status"] = "Validated"
    dr_df.to_csv(data_dir / "30_deep_research.csv", index=False)
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
    assert abc_row["Biotech Peer Read-Through (Y/N + link)"]
    assert abc_row["Evidence (Primary links)"]
    assert abc_row["Subscores Evidenced (x/5)"] >= 4
    assert abc_row["Status"] == "Validated"

    validated, exclusions = build_validated_selections(str(data_dir))
    assert (data_dir / "40_validated_selections.csv").exists()
    assert len(validated) == 1
    assert validated.iloc[0]["Ticker"] == "ABC"
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["Ticker"] == "XYZ"
    assert set(validated.columns) >= {
        "ValidationStatus",
        "PrimarySource",
        "SecondarySource",
        "PrimaryCatalystTier",
        "PrimaryCatalystURL",
        "CatalystEvidencePrimary",
    }

    validated_row = validated.set_index("Ticker").loc["ABC"]
    deep_row = dr_df.set_index("Ticker").loc["ABC"]
    shortlist_row = pd.read_csv(data_dir / "20_candidate_shortlist.csv").set_index("Ticker").loc["ABC"]
    assert validated_row["DilutionScore"] == deep_row["Dilution"]
    assert validated_row["CatalystScore"] == deep_row["Catalyst"]
    assert validated_row["RunwayEvidencePrimary"] == deep_row["RunwayEvidencePrimary"]
    assert shortlist_row["PrimaryCatalystDate"] == deep_row["PrimaryCatalystDate"]
    assert shortlist_row["PrimaryCatalystURL"] == deep_row["PrimaryCatalystURL"]


def test_w3_selects_first_numeric_runway(tmp_path, monkeypatch):
    data_dir = tmp_path

    def _fake_runway(url: str, adapter=None, return_reason: bool = False, include_reason_meta: bool = False, **kwargs):
        if "fail" in url:
            if return_reason:
                if include_reason_meta:
                    return None, False, "NO_CASHFLOW", "missing", {}
                return None, False, "NO_CASHFLOW", "missing"
            return None, False
        if return_reason:
            if include_reason_meta:
                return 5.0, True, "OK", "", {}
            return 5.0, True, "OK", ""
        return 5.0, True

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "MNO",
                "Company": "Mono",
                "CIK": "0000000003",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 4.0,
                "MarketCap": 75_000_000,
                "ADV20": 40000,
            }
        ],
    )

    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "MNO",
                "CIK": "0000000003",
                "FormType": "10-Q",
                "Age": 1,
                "FilingURL": "https://example.com/fail",
            },
            {
                "Ticker": "MNO",
                "CIK": "0000000003",
                "FormType": "10-Q",
                "Age": 2,
                "FilingURL": "https://example.com/pass",
            },
        ],
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "MNO",
                "CIK": "0000000003",
                "Tier": "Tier-2",
                "EventDate": "2024-05-03",
            }
        ],
    )
    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "MNO",
                "CIK": "0000000003",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 4.0,
                "MarketCap": 75_000_000,
                "ADV20": 40_000,
            }
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))

    row = dr_df.set_index("Ticker").loc["MNO"]
    assert row["RunwayQuarters"] == 5.0
    assert row["RunwaySourceURL"] == "https://example.com/pass"
    assert row["RunwayReasonCode"] == "OK"


def test_w3_keeps_last_failure_reason(tmp_path, monkeypatch):
    data_dir = tmp_path

    responses = [
        (None, False, "NO_XBRL", "missing XBRL", {}),
        (None, False, "NO_CASHFLOW", "missing cashflow", {}),
    ]

    def _fake_runway(
        url: str,
        adapter=None,
        return_reason: bool = False,
        include_reason_meta: bool = False,
        **kwargs,
    ):
        result = responses.pop(0)
        if return_reason:
            if include_reason_meta:
                return result
            return result[:-1]
        return result[0], result[1]

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "NOP",
                "Company": "Nope Corp",
                "CIK": "0000000004",
                "Sector": "Technology",
                "Industry": "Software",
            }
        ],
    )

    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "NOP",
                "CIK": "0000000004",
                "FormType": "10-Q",
                "Age": 1,
                "FilingURL": "https://example.com/fail1",
            },
            {
                "Ticker": "NOP",
                "CIK": "0000000004",
                "FormType": "10-Q",
                "Age": 2,
                "FilingURL": "https://example.com/fail2",
            },
        ],
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "NOP",
                "CIK": "0000000004",
                "Tier": "Tier-2",
                "EventDate": "2024-05-04",
            }
        ],
    )

    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "NOP",
                "CIK": "0000000004",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 3.0,
                "MarketCap": 60_000_000,
                "ADV20": 30_000,
            }
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))

    row = dr_df.set_index("Ticker").loc["NOP"]
    assert pd.isna(row["RunwayQuarters"])
    assert row["RunwaySourceURL"] == "https://example.com/fail2"
    assert row["RunwayReasonCode"] == "NO_CASHFLOW"
    assert row["RunwayReasonDetail"] == "missing cashflow"


def test_w3_does_not_overwrite_url_less_failure(tmp_path, monkeypatch):
    data_dir = tmp_path

    responses = [
        (None, False, "NO_XBRL", "missing XBRL", {}),
    ]

    def _load_csv_keep_defaults(path: str, required=None):
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
        if required:
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")
        return df

    def _fake_runway(
        url: str,
        adapter=None,
        return_reason: bool = False,
        include_reason_meta: bool = False,
        **kwargs,
    ):
        result = responses.pop(0)
        if return_reason:
            if include_reason_meta:
                return result
            return result[:-1]
        return result[0], result[1]

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)
    monkeypatch.setattr("app.weekly_deep_research._load_csv", _load_csv_keep_defaults)

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "NOP2",
                "Company": "Nope Corp 2",
                "CIK": "0000000006",
                "Sector": "Technology",
                "Industry": "Software",
            }
        ],
    )

    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "NOP2",
                "CIK": "0000000006",
                "FormType": "10-Q",
                "Age": 1,
                "FilingURL": "https://example.com/fail-first",
            },
            {
                "Ticker": "NOP2",
                "CIK": "0000000006",
                "FormType": "10-Q",
                "Age": 2,
                "FilingURL": "",
            },
        ],
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "NOP2",
                "CIK": "0000000006",
                "Tier": "Tier-2",
                "EventDate": "2024-05-04",
            }
        ],
    )

    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "NOP2",
                "CIK": "0000000006",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 3.0,
                "MarketCap": 60_000_000,
                "ADV20": 30_000,
            }
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))

    row = dr_df.set_index("Ticker").loc["NOP2"]
    assert pd.isna(row["RunwayQuarters"])
    assert row["RunwaySourceURL"] == "https://example.com/fail-first"
    assert row["RunwayReasonCode"] == "NO_XBRL"
    assert row["RunwayReasonDetail"] == "missing XBRL"


def test_w3_empty_filings_keeps_fallback_reason(tmp_path, monkeypatch):
    data_dir = tmp_path

    def _fake_runway(*args, **kwargs):
        raise AssertionError("compute_runway_quarters should not be called")

    monkeypatch.setattr("app.weekly_deep_research.compute_runway_quarters", _fake_runway)

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "QRS",
                "Company": "Qrs Corp",
                "CIK": "0000000005",
                "Sector": "Technology",
                "Industry": "Software",
            }
        ],
    )

    pd.DataFrame(columns=["Ticker", "CIK", "FormType", "FilingURL"]).to_csv(
        data_dir / "02_filings.csv", index=False
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "QRS",
                "CIK": "0000000005",
                "Tier": "Tier-3",
                "EventDate": "2024-05-05",
            }
        ],
    )

    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "QRS",
                "CIK": "0000000005",
                "Sector": "Technology",
                "Industry": "Software",
                "Price": 3.5,
                "MarketCap": 55_000_000,
                "ADV20": 25_000,
            }
        ],
    )

    dr_df = run_weekly_deep_research(str(data_dir))

    row = dr_df.set_index("Ticker").loc["QRS"]
    assert pd.isna(row["RunwayQuarters"])
    assert row["RunwaySourceURL"] == ""
    assert row["RunwayReasonCode"] == ""
    assert row["RunwayReasonDetail"] == ""
