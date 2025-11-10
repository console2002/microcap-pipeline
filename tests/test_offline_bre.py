from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser_10q


def _load_offline_result(filename: str, form: str) -> dict:
    base_path = Path(__file__).resolve().parent / filename
    uri = base_path.resolve().as_uri()
    if form:
        uri = f"{uri}?form={form}"
    return parser_10q.get_runway_from_filing(uri)


def _assert_cash_value(cash_raw: float | None) -> None:
    assert cash_raw is not None
    expected_values = (1_531_994, 2_479_981)
    assert any(
        cash_raw == pytest.approx(expected, rel=0.02) for expected in expected_values
    ), f"cash_raw {cash_raw} not within expected bands"


def test_offline_brera_20f_original() -> None:
    result = _load_offline_result("ea0242119-20f_brerahold.htm", "20-F")

    assert result.get("form_type") == "20-F"
    assert result.get("period_months") == 12
    assert result.get("status") != "Missing OCF"

    ocf_raw = result.get("ocf_raw")
    assert ocf_raw is not None
    assert ocf_raw < 0
    assert abs(ocf_raw) == pytest.approx(3_121_362, rel=0.02)

    _assert_cash_value(result.get("cash_raw"))

    if ocf_raw < 0:
        runway_quarters = result.get("runway_quarters")
        assert runway_quarters is not None
        assert runway_quarters > 0
        assert result.get("estimate") == "annual_div4"


def test_offline_brera_20f_amendment() -> None:
    result = _load_offline_result("ea0242667-20fa1_brerahold.htm", "20-F")

    assert result.get("form_type") == "20-F"
    assert result.get("period_months") == 12
    assert result.get("status") != "Missing OCF"

    ocf_raw = result.get("ocf_raw")
    assert ocf_raw is not None
    assert ocf_raw < 0
    assert abs(ocf_raw) == pytest.approx(3_121_362, rel=0.02)

    _assert_cash_value(result.get("cash_raw"))

    if ocf_raw < 0:
        runway_quarters = result.get("runway_quarters")
        assert runway_quarters is not None
        assert runway_quarters > 0
        assert result.get("estimate") == "annual_div4"


def test_offline_brera_6k_exhibit() -> None:
    result = _load_offline_result("ea025948901ex99-1_brera.htm", "6-K")

    assert result.get("form_type") == "6-K"

    status = result.get("status")
    ocf_raw = result.get("ocf_raw")

    if ocf_raw is not None:
        assert status != "Missing OCF"
        period_months = result.get("period_months")
        assert period_months in {3, 6}
    else:
        # Some 6-K exhibits are PDFs or images; allow Missing OCF in that case.
        assert status == "Missing OCF"


@pytest.mark.parametrize(
    "doc_param",
    [
        "ea025948901ex99-1_brera.htm",
        "/Archives/edgar/data/1939965/000121390025102293/ea025948901ex99-1_brera.htm",
    ],
)
def test_offline_brera_ixviewer_redirect(doc_param: str) -> None:
    base_path = Path(__file__).resolve().parent / "BRERA Exhibit99_1.htm"
    doc_path = Path(__file__).resolve().parent / "ea025948901ex99-1_brera.htm"

    assert base_path.exists()
    assert doc_path.exists()

    query = f"doc={doc_param}&form=6-K"
    uri = f"{base_path.resolve().as_uri()}?{query}"

    result = parser_10q.get_runway_from_filing(uri)

    assert result.get("form_type") == "6-K"
    assert result.get("status") == "OK"
    assert result.get("period_months") == 6
    assert result.get("ocf_raw") == pytest.approx(-3_160_656, rel=0.02)
    assert result.get("cash_raw") == pytest.approx(658_136, rel=0.02)
    assert result.get("quarterly_burn") == pytest.approx(1_580_328, rel=0.02)
    assert result.get("runway_quarters") == pytest.approx(0.42, rel=0.02)


def test_offline_goldmining_6k_may_2025() -> None:
    result = _load_offline_result("ex_837171.htm", "6-K")

    assert result.get("form_type") == "6-K"
    status = result.get("status")
    assert status is not None
    assert status.startswith("OK")
    assert result.get("period_months") == 6

    assert result.get("ocf_raw") == pytest.approx(-8_619_000, rel=0.02)
    assert result.get("cash_raw") == pytest.approx(6_145_000, rel=0.02)
    assert result.get("quarterly_burn") == pytest.approx(4_309_500, rel=0.02)
    assert result.get("runway_quarters") == pytest.approx(1.43, rel=0.02)
    assert result.get("assumption") == "6m/2"


def test_offline_goldmining_6k_aug_2025() -> None:
    result = _load_offline_result("ex_866936.htm", "6-K")

    assert result.get("form_type") == "6-K"
    status = result.get("status")
    assert status is not None
    assert status.startswith("OK")
    assert result.get("period_months") == 9

    assert result.get("ocf_raw") == pytest.approx(-16_239_000, rel=0.02)
    assert result.get("cash_raw") == pytest.approx(6_521_000, rel=0.02)
    assert result.get("quarterly_burn") == pytest.approx(5_413_000, rel=0.02)
    assert result.get("runway_quarters") == pytest.approx(1.20, rel=0.02)
    assert result.get("assumption") == "9m/3"


def test_offline_snow_lake_20f_2025() -> None:
    result = _load_offline_result("000175392625001675/g084981_20f.htm", "20-F")

    assert result.get("form_type") == "20-F"
    status = result.get("status")
    assert status is not None
    assert status.startswith("OK")
    assert result.get("period_months") == 12

    assert result.get("ocf_raw") == pytest.approx(-9_390_622, rel=0.02)
    assert result.get("cash_raw") == pytest.approx(17_829_149, rel=0.02)
    assert result.get("quarterly_burn") == pytest.approx(2_347_655.5, rel=0.02)
    assert result.get("runway_quarters") == pytest.approx(7.59, rel=0.02)
    assert result.get("assumption") == "annual/4"
