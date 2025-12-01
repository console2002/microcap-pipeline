import logging
from types import SimpleNamespace

import logging
from types import SimpleNamespace

import pandas as pd

from app import edgar_adapter


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
        result = adapter.extract_financial_sections(filing, form_hint=filing.form)

    assert result[0] is None
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

    empty_df = pd.DataFrame(columns=["label"])

    class DummyStatement:
        def __init__(self, df):
            self._df = df

        def render(self, standard=True):  # pragma: no cover - trivial
            class Rendered:
                def __init__(self, df):
                    self._df = df

                def to_dataframe(self):
                    return self._df

            return Rendered(self._df)

    class DummyFinancials:
        def income_statement(self):
            return DummyStatement(empty_df)

        def balance_sheet(self):
            return DummyStatement(empty_df)

        def cashflow_statement(self):
            return DummyStatement(empty_df)

    monkeypatch.setattr(
        edgar_adapter.Financials, "extract", staticmethod(lambda _: DummyFinancials())
    )
    monkeypatch.setattr(adapter, "_resolve_filing", lambda _: filing)

    with caplog.at_level(logging.WARNING):
        result = adapter.extract_financial_sections(filing, form_hint=filing.form)

    assert result is not None
    usable_msgs = [msg for msg in caplog.messages if "no_usable_periods" in msg]
    assert usable_msgs, "expected no_usable_periods warning"
    assert all("ticker=PRDS" in msg for msg in usable_msgs)
    assert all(msg.strip() for msg in usable_msgs)
