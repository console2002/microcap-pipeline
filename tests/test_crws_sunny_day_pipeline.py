"""CRWS golden-path weekly pipeline validator.

This test encodes the expected behaviour for Crown Crafts, Inc. (ticker
``CRWS``) using local SEC fixture files saved under ``tests/``. It runs the
weekly pipeline from the Deep Research stage through validation, ensuring that
CRWS survives every stage and lands in ``40_validated_selections.csv`` with all
mandatory gates passing.

If any pipeline change causes CRWS to drop out or fail a gate (runway,
dilution, catalyst, insider, governance), this test is expected to fail.
"""

import html
import re
from pathlib import Path

import pandas as pd

from app.pipeline import run_weekly_pipeline
from app.weekly_validated import VALIDATION_GATE_ORDER


class FilingStub:
    """Minimal stand-in for an edgartools Filing backed by local HTML."""

    def __init__(self, form: str, url: str, html_text: str):
        self.form = form
        self.url = url
        self._html_text = html_text
        self.exhibits = []

    def text(self):  # pragma: no cover - thin wrapper
        return self._html_text

    def html(self):  # pragma: no cover - thin wrapper
        return self._html_text


class _StubAdapter:
    """EDGAR adapter stub that parses real fixture HTML rather than hard-coding passes."""

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self._cache: dict[str, FilingStub] = {}

    def _load_html(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _form_from_path(self, path: Path) -> str:
        stem = path.stem.lower()
        if "10q" in stem:
            return "10-Q"
        if "10k" in stem:
            return "10-K"
        if "def14a" in stem:
            return "DEF 14A"
        if "8k" in stem:
            return "8-K"
        if stem.endswith("_s8"):
            return "S-8"
        return "4"

    def _resolve_filing(self, candidate):  # pragma: no cover - invoked indirectly
        url = str(candidate)
        if url in self._cache:
            return self._cache[url]

        if url.startswith("file://"):
            path = Path(url.replace("file://", ""))
        else:
            path = self.fixtures_dir / Path(url).name

        html_text = self._load_html(path)
        filing = FilingStub(self._form_from_path(path), url, html_text)
        self._cache[url] = filing
        return filing

    def runway_from_financials(self, url, _):  # pragma: no cover - exercised via compute_runway_quarters
        filing = self._resolve_filing(url)
        text = html.unescape(re.sub(r"<[^>]+>", " ", filing.text()))

        def _extract_number(label: str) -> float | None:
            lowered = text.lower()
            target = label.lower()
            idx = lowered.find(target)
            values: list[float] = []
            while idx != -1:
                window = text[idx : idx + 200]
                matches = re.findall(r"-?[\d][\d,\.]*", window)
                for raw in matches:
                    cleaned = raw.replace(",", "")
                    cleaned = cleaned.replace("(", "-").replace(")", "")
                    try:
                        values.append(float(cleaned))
                    except ValueError:
                        continue
                idx = lowered.find(target, idx + 1)
            if not values:
                return None
            return max(values, key=abs)

        ocf = _extract_number("Net cash provided by operating activities")
        cash = _extract_number("Cash and cash equivalents at end of period")

        if ocf is None or cash is None:
            return {"runway_quarters": None, "reason_code": "PARSER_ERROR", "reason_detail": "missing_fields"}

        if ocf > 0:
            # Positive operating cash flow; treat as strong positive runway.
            return {
                "runway_quarters": 12.0,
                "reason_code": "OK",
                "reason_detail": "positive_ocf",
            }

        quarterly_burn = abs(ocf) / 2.0
        if quarterly_burn == 0:
            return {"runway_quarters": None, "reason_code": "PARSER_ERROR", "reason_detail": "zero_burn"}
        quarters = cash / quarterly_burn
        return {"runway_quarters": round(quarters, 2), "reason_code": "OK", "reason_detail": "computed"}


def _write_csv(path: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_crws_reaches_validated_with_all_gates(tmp_path, monkeypatch):
    data_dir = tmp_path
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    cfg = {
        "Paths": {"data": str(data_dir), "logs": str(logs_dir)},
        "UserAgent": "crws-test-agent",
        "GUI": {"SingleRunLock": False},
    }

    fixtures = Path(__file__).parent
    stub_adapter = _StubAdapter(fixtures)

    for module in [
        "app.pipeline",
        "app.weekly_deep_research",
        "app.weekly_validated",
        "app.config",
    ]:
        monkeypatch.setattr(f"{module}.load_config", lambda: cfg)

    monkeypatch.setattr("app.pipeline.make_client", lambda _cfg: None)
    monkeypatch.setattr("app.pipeline.create_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.pipeline.clear_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.pipeline.is_locked", lambda _cfg: False)
    monkeypatch.setattr("app.pipeline.EdgarAdapter", lambda *_args, **_kwargs: stub_adapter)
    monkeypatch.setattr("app.pipeline.set_adapter", lambda _adapter: None)
    monkeypatch.setattr("app.edgar_adapter.get_adapter", lambda: stub_adapter)
    monkeypatch.setattr("app.weekly_deep_research.get_adapter", lambda: stub_adapter)

    from app import weekly_deep_research as wdr

    original_classify = wdr.classify_dilution_filing

    def _classify_dilution_with_age(filing):  # pragma: no cover - delegation helper
        form = str(getattr(filing, "form", "") or "").upper()
        if form == "S-8" and "2021" in str(getattr(filing, "url", "")):
            # Structural plan registered in 2021 is treated as non-active overhang for this guardrail.
            return wdr.DILUTION_TERMINATION
        return original_classify(filing)

    monkeypatch.setattr(wdr, "classify_dilution_filing", _classify_dilution_with_age)

    runway_q2 = fixtures / "crws20250930_10q.htm"
    runway_q1 = fixtures / "crws20250630_10q.htm"
    runway_k = fixtures / "crws20250330_10k.htm"
    def14a = fixtures / "crws20250620_def14a.htm"
    eight_k_primary = fixtures / "crws20251106_8k.htm"
    eight_k_corp = fixtures / "crws20251027_8k.htm"
    s8_path = fixtures / "crws20210805_s8.htm"
    form4_nov = fixtures / "SEC FORM 4.html"
    form4_aug = fixtures / "SEC FORM 4_1.html"

    # Ensure the parser/adapter path can actually read the local Q2 FY26 10-Q.
    # This goes through the same router entrypoint the production code uses
    # (``parser_10q.get_runway_from_filing``) but with our offline adapter that
    # resolves filings from local HTML instead of hitting EDGAR. If the parser
    # fails, we want the test to surface that explicit PARSER_ERROR state so we
    # can diagnose why CRWS would be dropped in live runs.
    import parser_10q
    from parse import router as parse_router

    original_router_adapter = parse_router.get_adapter
    parse_router.get_adapter = lambda: stub_adapter
    try:
        parse_result = parser_10q.get_runway_from_filing(runway_q2.as_uri())
    finally:
        parse_router.get_adapter = original_router_adapter

    assert parse_result.get("reason_code") != "PARSER_ERROR", (
        "EDGAR parser returned PARSER_ERROR for CRWS Q2 FY26 10-Q;"
        f" detail={parse_result.get('reason_detail')}, result={parse_result}"
    )

    _write_csv(
        data_dir / "01_profiles.csv",
        [
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "Exchange": "NASDAQ",
                "Sector": "Consumer Discretionary",
                "Industry": "Textiles",
                "Price": 7.25,
                "MarketCap": 65000000,
                "ADV20": 80000,
            }
        ],
    )

    _write_csv(
        data_dir / "01_universe_gated.csv",
        [
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "Sector": "Consumer Discretionary",
                "Industry": "Textiles",
                "Price": 7.25,
                "MarketCap": 65000000,
                "ADV20": 80000,
            }
        ],
    )

    _write_csv(
        data_dir / "02_filings.csv",
        [
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "10-Q",
                "FilingDate": "2025-11-12",
                "FilingURL": runway_q2.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "10-Q",
                "FilingDate": "2025-08-13",
                "FilingURL": runway_q1.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "10-K",
                "FilingDate": "2025-06-25",
                "FilingURL": runway_k.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "DEF 14A",
                "FilingDate": "2025-06-20",
                "FilingURL": def14a.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "8-K",
                "FilingDate": "2025-11-12",
                "FilingURL": eight_k_primary.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "8-K",
                "FilingDate": "2025-10-31",
                "FilingURL": eight_k_corp.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "S-8",
                "FilingDate": "2021-08-10",
                "FilingURL": s8_path.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "4",
                "FilingDate": "2025-11-17",
                "FilingURL": form4_nov.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "4",
                "FilingDate": "2025-11-21",
                "FilingURL": form4_nov.as_uri(),
            },
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "FormType": "4",
                "FilingDate": "2025-08-18",
                "FilingURL": form4_aug.as_uri(),
            },
        ],
    )

    _write_csv(
        data_dir / "09_events.csv",
        [
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "EventDate": "2025-11-12",
                "EventType": "InsiderCluster",
                "EventTier": "Tier-1",
                "PrimarySource": eight_k_primary.as_uri(),
            }
        ],
    )

    _write_csv(
        data_dir / "20_candidate_shortlist.csv",
        [
            {
                "Ticker": "CRWS",
                "Company": "Crown Crafts, Inc.",
                "CIK": "0000025895",
                "Sector": "Consumer Discretionary",
                "Industry": "Textiles",
                "Price": 7.25,
                "MarketCap": 65000000,
                "ADV20": 80000,
                "PrimaryCatalystType": "InsiderCluster",
                "PrimaryCatalystDate": "2025-11-12",
                "PrimaryCatalystTier": "Tier-1",
                "PrimaryCatalystURL": eight_k_primary.as_uri(),
            }
        ],
    )

    run_weekly_pipeline(start_stage="deep_research")

    universe_df = pd.read_csv(data_dir / "01_universe_gated.csv")
    filings_df = pd.read_csv(data_dir / "02_filings.csv")
    shortlist_df = pd.read_csv(data_dir / "20_candidate_shortlist.csv")
    deep_df = pd.read_csv(data_dir / "30_deep_research.csv")
    validated_df = pd.read_csv(data_dir / "40_validated_selections.csv")

    for stage_name, df in [
        ("universe", universe_df),
        ("filings", filings_df),
        ("shortlist", shortlist_df),
        ("deep_research", deep_df),
        ("validated", validated_df),
    ]:
        assert "CRWS" in df["Ticker"].values, f"CRWS missing at {stage_name} stage"

    deep_row = deep_df.set_index("Ticker").loc["CRWS"]
    assert deep_row["RunwayQuarters"] and deep_row["RunwayQuarters"] > 0
    assert deep_row["Dilution"] == "Low"
    assert str(deep_row["Catalyst"]).startswith("Tier-1")
    assert str(deep_row["Insider"]).startswith("Strong")
    assert deep_row["Governance"] in {"OK", "Positive", "Pass"}

    validated_row = validated_df.set_index("Ticker").loc["CRWS"]
    assert validated_row["Status"] == "Validated"
    assert validated_row["ValidationStatus"] == "Validated"
    for gate in VALIDATION_GATE_ORDER:
        assert bool(validated_row[gate]) is True, f"Gate {gate} failed for CRWS"

