"""Runway computation helpers using edgartools Financials objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


RUNWAY_REASON_OK = "OK"
RUNWAY_REASON_NO_XBRL = "NO_XBRL"
RUNWAY_REASON_NO_BALANCE = "NO_BALANCE_SHEET"
RUNWAY_REASON_NO_CASHFLOW = "NO_CASHFLOW"
RUNWAY_REASON_NO_PERIODS = "NO_PERIODS"
RUNWAY_REASON_UNSUPPORTED_FORM = "UNSUPPORTED_FORM"
RUNWAY_REASON_PARSER_ERROR = "PARSER_ERROR"


_METADATA_COLUMNS = {
    "concept",
    "label",
    "level",
    "abstract",
    "dimension",
    "balance",
    "weight",
    "preferred_sign",
    "unit",
    "point_in_time",
}

_OCF_LABELS = [
    "Net cash provided by operating activities",
    "Net cash used in operating activities",
    "Net cash provided by (used in) operating activities",
    "Operating cash flow",
]

_CASH_LABELS = [
    "Cash and cash equivalents",
    "Cash and cash equivalents, at end of period",
    "Cash and cash equivalents at carrying value",
]

_PERIOD_PATTERNS: dict[int, str] = {
    3: r"(?:THREE|3)\s+MONTH",
    6: r"(?:SIX|6)\s+MONTH",
    9: r"(?:NINE|9)\s+MONTH",
    12: r"(?:TWELVE|12)\s+MONTH|FISCAL YEAR|YEAR",
}

_POSITIVE_OCF_RUNWAY = 99.0


@dataclass
class RunwayComputation:
    runway_quarters: Optional[float]
    period_months: Optional[int]
    cash: Optional[float]
    ocf: Optional[float]
    reason_code: str
    reason_detail: str


def _coerce_numeric(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    normalized = text.replace(",", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = f"-{normalized[1:-1]}"
    try:
        return float(normalized)
    except ValueError:
        return None


def _value_columns(df: pd.DataFrame) -> list:
    cols = [col for col in df.columns if col not in _METADATA_COLUMNS]

    def _sort_key(col):
        try:
            return pd.to_datetime(col)
        except Exception:
            return col

    return sorted(cols, key=_sort_key, reverse=True)


def _find_value_by_label(df: pd.DataFrame, keywords: Iterable[str]) -> Optional[float]:
    if df is None or getattr(df, "empty", True):
        return None
    if "label" not in df.columns:
        return None

    value_cols = _value_columns(df)
    for keyword in keywords:
        matches = df[df["label"].str.contains(keyword, case=False, na=False, regex=False)]
        if matches.empty:
            continue
        for _, row in matches.iterrows():
            for col in value_cols:
                val = _coerce_numeric(row.get(col))
                if val is not None:
                    return val
    return None


def _infer_period_from_columns(df: pd.DataFrame) -> Optional[int]:
    if df is None or getattr(df, "empty", True):
        return None

    value_cols = _value_columns(df)
    parsed_dates = []
    for col in value_cols:
        try:
            parsed = pd.to_datetime(col)
        except Exception:
            continue
        parsed_dates.append(parsed)

    if len(parsed_dates) >= 2:
        parsed_dates.sort()
        delta_months = abs(parsed_dates[-1] - parsed_dates[-2]).days / 30.0
        for months in (3, 6, 9, 12):
            if abs(delta_months - months) < 1.5:
                return months

    for months, pattern in _PERIOD_PATTERNS.items():
        try:
            match = df[value_cols].columns.to_series().astype(str).str.contains(
                pattern, regex=True, case=False
            )
        except Exception:
            match = []
        if any(match):
            return months
    return None


def _normalize_burn_per_quarter(ocf_value: Optional[float], period_months: Optional[int]) -> Optional[float]:
    if ocf_value is None or period_months is None:
        return None
    if period_months <= 0:
        return None
    quarters = period_months / 3.0
    if quarters <= 0:
        return None
    return ocf_value / quarters


def compute_runway_from_financials(financials) -> RunwayComputation:
    """Derive runway using edgartools Financials as the only source."""

    if financials is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=None,
            cash=None,
            ocf=None,
            reason_code=RUNWAY_REASON_NO_XBRL,
            reason_detail="financials missing",
        )

    try:
        cashflow_stmt = financials.cashflow_statement()
    except Exception:
        cashflow_stmt = None

    try:
        balance_stmt = financials.balance_sheet()
    except Exception:
        balance_stmt = None

    if cashflow_stmt is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=None,
            cash=None,
            ocf=None,
            reason_code=RUNWAY_REASON_NO_CASHFLOW,
            reason_detail="cashflow statement missing",
        )

    if balance_stmt is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=None,
            cash=None,
            ocf=None,
            reason_code=RUNWAY_REASON_NO_BALANCE,
            reason_detail="balance sheet missing",
        )

    try:
        cashflow_df = cashflow_stmt.to_dataframe(presentation=True, include_unit=True)
    except Exception:
        cashflow_df = None

    try:
        balance_df = balance_stmt.to_dataframe(presentation=True, include_unit=True)
    except Exception:
        balance_df = None

    ocf_value = None
    try:
        ocf_value = financials.get_operating_cash_flow()
    except Exception:
        ocf_value = None

    if ocf_value is None:
        ocf_value = _find_value_by_label(cashflow_df, _OCF_LABELS)

    cash_value = _find_value_by_label(balance_df, _CASH_LABELS)
    period_months = _infer_period_from_columns(cashflow_df)

    if ocf_value is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=period_months,
            cash=cash_value,
            ocf=None,
            reason_code=RUNWAY_REASON_NO_CASHFLOW,
            reason_detail="operating cash flow missing",
        )

    if cash_value is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=period_months,
            cash=None,
            ocf=ocf_value,
            reason_code=RUNWAY_REASON_NO_BALANCE,
            reason_detail="cash and cash equivalents missing",
        )

    if period_months is None:
        return RunwayComputation(
            runway_quarters=None,
            period_months=None,
            cash=cash_value,
            ocf=ocf_value,
            reason_code=RUNWAY_REASON_NO_PERIODS,
            reason_detail="unable to infer reporting period",
        )

    ocf_quarterly = _normalize_burn_per_quarter(ocf_value, period_months)
    if ocf_quarterly is None or ocf_quarterly == 0:
        return RunwayComputation(
            runway_quarters=None,
            period_months=period_months,
            cash=cash_value,
            ocf=ocf_value,
            reason_code=RUNWAY_REASON_NO_CASHFLOW,
            reason_detail="operating cash flow not scalable to quarter",
        )

    if ocf_quarterly >= 0:
        return RunwayComputation(
            runway_quarters=_POSITIVE_OCF_RUNWAY,
            period_months=period_months,
            cash=cash_value,
            ocf=ocf_value,
            reason_code=RUNWAY_REASON_OK,
            reason_detail="positive operating cash flow",
        )

    runway_quarters = cash_value / abs(ocf_quarterly) if ocf_quarterly else None

    return RunwayComputation(
        runway_quarters=float(runway_quarters) if runway_quarters is not None else None,
        period_months=period_months,
        cash=cash_value,
        ocf=ocf_value,
        reason_code=RUNWAY_REASON_OK if runway_quarters is not None else RUNWAY_REASON_NO_CASHFLOW,
        reason_detail="" if runway_quarters is not None else "operating cash flow zero",
    )


__all__ = [
    "RunwayComputation",
    "compute_runway_from_financials",
    "RUNWAY_REASON_OK",
    "RUNWAY_REASON_NO_XBRL",
    "RUNWAY_REASON_NO_BALANCE",
    "RUNWAY_REASON_NO_CASHFLOW",
    "RUNWAY_REASON_NO_PERIODS",
    "RUNWAY_REASON_UNSUPPORTED_FORM",
    "RUNWAY_REASON_PARSER_ERROR",
]
