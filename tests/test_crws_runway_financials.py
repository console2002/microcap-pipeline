import pandas as pd

from app.edgar_adapter import EdgarAdapter
from app.runway_utils import compute_runway_quarters


class _StatementStub:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self, *_, **__):  # pragma: no cover - simple passthrough
        return self._df


class _FinancialsStub:
    def __init__(self, cashflow_df: pd.DataFrame, balance_df: pd.DataFrame, ocf_value):
        self._cashflow_df = cashflow_df
        self._balance_df = balance_df
        self._ocf_value = ocf_value
        self.form = "10-Q"

    def cashflow_statement(self):  # pragma: no cover - exercised indirectly
        return _StatementStub(self._cashflow_df)

    def balance_sheet(self):  # pragma: no cover - exercised indirectly
        return _StatementStub(self._balance_df)

    def get_operating_cash_flow(self):  # pragma: no cover - exercised indirectly
        return self._ocf_value


class _FilingStub:
    def __init__(self, form: str, accession: str, financials):
        self.form = form
        self.accession_no = accession
        self.filing_date = "2025-11-12"
        self.url = "https://www.sec.gov/Archives/example.html"
        self.financials = financials


def _build_financials(ocf_value: str, cash_value: str, label: str) -> _FinancialsStub:
    cashflow_df = pd.DataFrame(
        {"label": ["Net cash provided by operating activities"], label: [ocf_value]}
    )
    balance_df = pd.DataFrame(
        {"label": ["Cash and cash equivalents, at end of period"], label: [cash_value]}
    )
    return _FinancialsStub(cashflow_df, balance_df, ocf_value)


def test_crws_runway_from_financials_handles_string_values(monkeypatch):
    adapter = EdgarAdapter()

    financials = _build_financials("4,403", "810", "2025-09-28 - Six Months Ended")
    filing = _FilingStub("10-Q", "0001437749-25-034222", financials)

    monkeypatch.setattr(adapter, "_resolve_filing", lambda url: filing)

    result = adapter.runway_from_financials(
        "https://www.sec.gov/Archives/edgar/data/25895/000143774925034222/crws20250930_10q.htm",
        "10-Q",
    )

    assert result["reason_code"] == "OK"
    assert result["cash"] == 810
    assert result["ocf"] == 4403
    assert result["period_months"] == 6

    (
        quarters,
        used_primary,
        reason_code,
        reason_detail,
        reason_meta,
    ) = compute_runway_quarters(
        "https://www.sec.gov/Archives/edgar/data/25895/000143774925034222/crws20250930_10q.htm",
        adapter=adapter,
        return_reason=True,
        include_reason_meta=True,
    )

    assert used_primary is True
    assert reason_code == "OK"
    assert "positive" in reason_detail
    assert reason_meta == {"error_type": "", "error_message": "", "error_stage": ""}
    assert quarters == result["runway_quarters"] == 99
