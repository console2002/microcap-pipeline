import types
from datetime import date, timedelta

import pandas as pd
import types
from datetime import date, timedelta

import app.edgar_adapter as edgar_adapter
from parse.router import _fetch_url


def _base_cfg(tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "FilingsWhitelist": ["10-Q", "8-K"],
        "FilingsGroups": {
            "Financials_Quarterly": {
                "forms": ["10-Q"],
                "lookback_days": 90,
            }
        },
        "RateLimitsPerMin": {"SEC": 0},
        "Universe": {
            "DropPatterns": [".U"],
            "DropWordPatterns": ["WARRANT"],
            "NormalizeTicker": True,
        },
        "Edgar": {"ThrottlePerMin": 0, "UserAgent": "test@example.com"},
        "Paths": {"data": str(data_dir), "logs": str(logs_dir)},
    }


def test_load_company_universe_applies_filters(monkeypatch, tmp_path):
    cfg = _base_cfg(tmp_path)

    df = pd.DataFrame(
        [
            {"ticker": "abc", "cik_str": "123", "title": "Alpha"},
            {"ticker": "bad.U", "cik_str": "456", "title": "Should Drop"},
            {"ticker": "warr", "cik_str": "789", "title": "Warrant Holdings"},
        ]
    )

    monkeypatch.setattr(edgar_adapter, "get_company_tickers", lambda: df)

    adapter = edgar_adapter.EdgarAdapter(cfg)
    universe = adapter.load_company_universe()

    assert universe == [
        {"Ticker": "ABC", "CIK": "0000000123", "Company": "Alpha"}
    ]


def test_fetch_recent_filings_respects_lookback(monkeypatch, tmp_path):
    cfg = _base_cfg(tmp_path)

    recent_date = (date.today() - timedelta(days=10)).isoformat()
    stale_date = (date.today() - timedelta(days=400)).isoformat()

    filings = [
        types.SimpleNamespace(
            form="10-Q",
            filing_date=recent_date,
            cik="1",
            company="Test Co",
            filing_url="http://example.com/q",
        ),
        types.SimpleNamespace(
            form="8-K",
            filing_date=stale_date,
            cik="1",
            company="Test Co",
            filing_url="http://example.com/old",
        ),
    ]

    class DummyCompany:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_filings(self, form=None, filing_date=None):
            return filings

    monkeypatch.setattr(edgar_adapter, "Company", DummyCompany)

    adapter = edgar_adapter.EdgarAdapter(cfg)
    results = adapter.fetch_recent_filings(["TEST"])

    assert len(results) == 1
    assert results[0]["Form"].upper() == "10-Q"
    assert results[0]["URL"] == "http://example.com/q"


def test_router_fetch_url_uses_adapter(monkeypatch):
    class DummyAdapter:
        def __init__(self):
            self.requested = []

        def download_filing_text(self, url: str):
            self.requested.append(url)
            return "payload"

    dummy = DummyAdapter()
    monkeypatch.setattr("parse.router.get_adapter", lambda: dummy)

    data = _fetch_url("http://example.com/test")

    assert data == b"payload"
    assert dummy.requested == ["http://example.com/test"]


def test_set_adapter_shares_singleton(monkeypatch, tmp_path):
    monkeypatch.setattr(edgar_adapter, "_ADAPTER", None)
    cfg = _base_cfg(tmp_path)

    seeded = edgar_adapter.EdgarAdapter(cfg)
    edgar_adapter.set_adapter(seeded)

    assert edgar_adapter.get_adapter() is seeded


def test_runway_from_financials_builds_from_statements(monkeypatch, tmp_path):
    cashflow_df = pd.DataFrame(
        {
            "concept": ["cash_ocf"],
            "label": ["Net cash used in operating activities"],
            "Three Months Ended": [-1200],
        }
    )
    balance_df = pd.DataFrame(
        {
            "concept": ["cash"],
            "label": ["Cash and cash equivalents"],
            "Three Months Ended": [5000],
        }
    )

    class DummyRendered:
        def __init__(self, df):
            self.df = df

        def to_dataframe(self):
            return self.df

    class DummyStatement:
        def __init__(self, df):
            self.df = df

        def render(self, standard=True):
            return DummyRendered(self.df)

    class DummyFinancials:
        def __init__(self):
            self.called = True

        def income_statement(self):
            return DummyStatement(pd.DataFrame())

        def balance_sheet(self):
            return DummyStatement(balance_df)

        def cashflow_statement(self):
            return DummyStatement(cashflow_df)

    monkeypatch.setattr(
        edgar_adapter.Financials,
        "extract",
        classmethod(lambda cls, filing: DummyFinancials()),
    )

    adapter = edgar_adapter.EdgarAdapter(_base_cfg(tmp_path))
    fake_filing = types.SimpleNamespace(form="10-Q")
    monkeypatch.setattr(adapter, "_resolve_filing", lambda url: fake_filing)

    result = adapter.runway_from_financials("http://example.com/filing", "10-Q")

    assert result["cash_raw"] == 5000
    assert result["ocf_raw"] == -1200
    assert result["ocf_quarterly_raw"] == -1200
    assert result["period_months"] == 3


def test_parse_accession_from_file_style_url(tmp_path):
    filing_path = tmp_path / "0001234567-23-000999.htm"
    filing_path.write_text("<html></html>")

    url = filing_path.as_uri()

    cik, accession = edgar_adapter._parse_accession_from_url(url)

    assert cik == "0001234567"
    assert accession == "0001234567-23-000999"
