import pandas as pd

from app.runway_financials import (
    RUNWAY_REASON_NO_CASHFLOW,
    RUNWAY_REASON_OK,
    compute_runway_from_financials,
)


class DummyStatement:
    def __init__(self, df: pd.DataFrame | None):
        self._df = df

    def to_dataframe(self, **_):
        return self._df


class DummyFinancials:
    def __init__(self, balance_df=None, cashflow_df=None, ocf=None):
        self._balance_df = balance_df
        self._cashflow_df = cashflow_df
        self._ocf = ocf

    def balance_sheet(self):
        return DummyStatement(self._balance_df) if self._balance_df is not None else None

    def cashflow_statement(self):
        return DummyStatement(self._cashflow_df) if self._cashflow_df is not None else None

    def get_operating_cash_flow(self, period_offset: int = 0):
        return self._ocf


def test_compute_runway_from_financials_basic():
    balance_df = pd.DataFrame(
        {"label": ["Cash and cash equivalents"], "2024-06-30": [120.0]}
    )
    cashflow_df = pd.DataFrame(
        {
            "label": ["Net cash used in operating activities"],
            "2024-06-30 (Three Months)": [-30.0],
        }
    )
    financials = DummyFinancials(balance_df=balance_df, cashflow_df=cashflow_df, ocf=None)

    result = compute_runway_from_financials(financials)

    assert result.runway_quarters == 4.0
    assert result.reason_code == RUNWAY_REASON_OK


def test_compute_runway_from_financials_positive_ocf():
    balance_df = pd.DataFrame({"label": ["Cash and cash equivalents"], "2024-06-30": [50.0]})
    cashflow_df = pd.DataFrame(
        {"label": ["Net cash provided by operating activities"], "2024-06-30 (Three Months)": [10.0]}
    )
    financials = DummyFinancials(balance_df=balance_df, cashflow_df=cashflow_df, ocf=10.0)

    result = compute_runway_from_financials(financials)

    assert result.runway_quarters > 0
    assert result.runway_quarters >= 99.0
    assert result.reason_code == RUNWAY_REASON_OK


def test_compute_runway_from_financials_missing_ocf():
    balance_df = pd.DataFrame({"label": ["Cash and cash equivalents"], "2024-06-30": [50.0]})
    cashflow_df = pd.DataFrame({"label": ["Unrelated"], "2024-06-30 (Three Months)": [5.0]})
    financials = DummyFinancials(balance_df=balance_df, cashflow_df=cashflow_df, ocf=None)

    result = compute_runway_from_financials(financials)

    assert result.runway_quarters is None
    assert result.reason_code == RUNWAY_REASON_NO_CASHFLOW
