import pandas as pd

from app.runway_utils import compute_runway_quarters, write_runway_diagnostics


class DummyAdapter:
    def __init__(self, result):
        self._result = result

    def runway_from_financials(self, url, form_hint=None):
        return self._result

    def _resolve_filing(self, url):
        return None


def test_compute_runway_quarters_reason_ok():
    adapter = DummyAdapter({"runway_quarters": 4, "reason_code": "OK", "reason_detail": ""})
    quarters, used_primary, reason_code, reason_detail = compute_runway_quarters(
        "http://example.com", adapter=adapter, return_reason=True
    )

    assert quarters == 4
    assert used_primary is True
    assert reason_code == "OK"
    assert reason_detail == ""


def test_compute_runway_quarters_reason_no_xbrl():
    adapter = DummyAdapter({"runway_quarters": None, "reason_code": "NO_XBRL", "reason_detail": "missing"})
    quarters, used_primary, reason_code, reason_detail = compute_runway_quarters(
        "http://example.com", adapter=adapter, return_reason=True
    )

    assert quarters is None
    assert used_primary is False
    assert reason_code == "NO_XBRL"
    assert reason_detail == "missing"


def test_compute_runway_quarters_reason_no_cashflow():
    adapter = DummyAdapter({"runway_quarters": None, "reason_code": "NO_CASHFLOW", "reason_detail": "missing"})
    quarters, used_primary, reason_code, reason_detail = compute_runway_quarters(
        "http://example.com", adapter=adapter, return_reason=True
    )

    assert quarters is None
    assert used_primary is False
    assert reason_code == "NO_CASHFLOW"


def test_compute_runway_quarters_reason_no_periods():
    adapter = DummyAdapter({"runway_quarters": None, "reason_code": "NO_PERIODS", "reason_detail": "no periods"})
    quarters, used_primary, reason_code, reason_detail = compute_runway_quarters(
        "http://example.com", adapter=adapter, return_reason=True
    )

    assert quarters is None
    assert used_primary is False
    assert reason_code == "NO_PERIODS"


def test_write_runway_diagnostics(tmp_path):
    rows = [
        {
            "Ticker": "AAA",
            "CIK": "1",
            "Form": "10-Q",
            "FiledAt": "2024-01-01",
            "Accession": "0001",
            "RunwayQuarters": 4,
            "HasRunway": True,
            "RunwaySourceURL": "https://example.com/ok",
            "RunwayReasonCode": "OK",
            "RunwayReasonDetail": "",
        },
        {
            "Ticker": "BBB",
            "CIK": "2",
            "Form": "10-Q",
            "FiledAt": "2024-02-01",
            "Accession": "0002",
            "RunwayQuarters": None,
            "HasRunway": False,
            "RunwaySourceURL": "https://example.com/miss",
            "RunwayReasonCode": "NO_XBRL",
            "RunwayReasonDetail": "missing",
        },
        {
            "Ticker": "CCC",
            "CIK": "3",
            "Form": "10-Q",
            "FiledAt": "2024-03-01",
            "Accession": "0003",
            "RunwayQuarters": None,
            "HasRunway": False,
            "RunwaySourceURL": "https://example.com/error",
            "RunwayReasonCode": "PARSER_ERROR",
            "RunwayReasonDetail": "TypeError: bad value",
            "RunwayErrorType": "TypeError",
            "RunwayErrorMessage": "bad value",
            "RunwayErrorStage": "compute_runway",
        },
    ]

    output_path = tmp_path / "diag.csv"
    write_runway_diagnostics(rows, output_path)

    df = pd.read_csv(output_path)

    assert set(["Ticker", "CIK", "Form", "FiledAt", "Accession"]).issubset(df.columns)
    assert "RunwayReasonCode" in df.columns
    assert len(df) == 2
    assert df.iloc[0]["Ticker"] == "BBB"
    assert df.iloc[0]["RunwayReasonCode"] == "NO_XBRL"
    assert {"RunwayErrorType", "RunwayErrorMessage", "RunwayErrorStage", "Status"}.issubset(
        set(df.columns)
    )
