import logging
from types import SimpleNamespace

import pandas as pd

from app import edgar_adapter
from app.runway_financials import (
    RUNWAY_REASON_NO_PERIODS,
    RUNWAY_REASON_NO_XBRL,
    RUNWAY_REASON_OK,
)


def _adapter():
    cfg = {"FilingsWhitelistByRole": {}, "FilingsLookbacks": {}, "Workers": {}}
    return edgar_adapter.EdgarAdapter(cfg)


def test_missing_xbrl_warning(monkeypatch, caplog):
    adapter = _adapter()
    filing = SimpleNamespace(
        form="6-K",
        filing_date="2024-09-30",
        company="TestCo",
        cik="1234567890",
        accession_no="0001234567-89-000123",
        ticker="TST",
    )

    monkeypatch.setattr(edgar_adapter.Financials, "extract", staticmethod(lambda _: None))
    monkeypatch.setattr(adapter, "_resolve_filing", lambda _: filing)

    with caplog.at_level(logging.WARNING):
        result = adapter.runway_from_financials(filing, form_hint=filing.form)

    assert result.get("runway_quarters") is None
    assert result["reason_code"] == RUNWAY_REASON_NO_XBRL
    warning_messages = [msg for msg in caplog.messages if "missing_xbrl" in msg]
    assert warning_messages, "expected missing_xbrl warning"
    assert "ticker=TST" in warning_messages[0]
    assert "form=6-K" in warning_messages[0]
    assert warning_messages[0].strip(), "warning message should not be empty"


def test_no_usable_periods_warning(monkeypatch, caplog):
    adapter = _adapter()
    filing = SimpleNamespace(
        form="10-Q",
        filing_date="2024-06-30",
        company="Periods Corp",
        cik="0000000001",
        accession_no="0000000001-24-000001",
        ticker="PRDS",
    )

    cashflow_df = pd.DataFrame(
        {
            "label": ["Net cash used in operating activities"],
            "Value": [-100.0],
        }
    )
    balance_df = pd.DataFrame(
        {
            "label": ["Cash and cash equivalents"],
            "Value": [400.0],
        }
    )

    class DummyStatement:
        def __init__(self, df):
            self._df = df

        def to_dataframe(self, presentation=True, include_unit=True):  # pragma: no cover - trivial
            return self._df

    class DummyFinancials:
        def income_statement(self):
            return DummyStatement(pd.DataFrame())

        def balance_sheet(self):
            return DummyStatement(balance_df)

        def cashflow_statement(self):
            return DummyStatement(cashflow_df)

    monkeypatch.setattr(
        edgar_adapter.Financials, "extract", staticmethod(lambda _: DummyFinancials())
    )
    monkeypatch.setattr(adapter, "_resolve_filing", lambda _: filing)

    with caplog.at_level(logging.WARNING):
        result = adapter.runway_from_financials(filing, form_hint=filing.form)

    assert result is not None
    assert result.get("runway_quarters") == 4.0
    assert result["reason_code"] == RUNWAY_REASON_OK
    runway_messages = [msg for msg in caplog.messages if "edgar_runway" in msg]
    assert not runway_messages, "no runway warnings expected when defaults cover periods"
