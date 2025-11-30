import math

from parse.postproc import finalize_runway_result
from parse.units import normalize_ocf_value


def test_normalize_ocf_quarterly_periods():
    ocf_quarterly, period, assumption = normalize_ocf_value(-900_000, 9)
    assert ocf_quarterly == -300_000
    assert period == 9
    assert assumption == "9m/3"

    ocf_quarterly, period, assumption = normalize_ocf_value(-1_200_000, 12)
    assert ocf_quarterly == -300_000
    assert period == 12
    assert assumption == "annual/4"


def test_finalize_missing_ocf_and_missing_cash():
    result = finalize_runway_result(
        cash=1_000_000,
        ocf_raw=None,
        ocf_quarterly=None,
        period_months=6,
        assumption="6m/2",
        note="test",
        form_type="10-Q",
        units_scale=1,
        status=None,
        source_tags=None,
    )

    assert result["status"] == "Missing OCF"
    assert result["quarterly_burn"] is None
    assert result["runway_quarters"] is None
    assert result["complete"] is False

    result_missing_cash = finalize_runway_result(
        cash=None,
        ocf_raw=-300_000,
        ocf_quarterly=-300_000,
        period_months=3,
        assumption="",
        note="test",
        form_type="10-Q",
        units_scale=1,
        status=None,
        source_tags=None,
    )

    assert result_missing_cash["status"] == "Missing cash"
    assert result_missing_cash["runway_quarters"] is None


def test_finalize_positive_and_negative_ocf_paths():
    positive = finalize_runway_result(
        cash=2_000_000,
        ocf_raw=100_000,
        ocf_quarterly=100_000,
        period_months=3,
        assumption="",
        note="test",
        form_type="10-Q",
        units_scale=1,
        status=None,
        source_tags=None,
    )

    assert positive["status"] == "OCF positive (self-funding)"
    assert positive["quarterly_burn"] == 0.0
    assert positive["runway_quarters"] is None
    assert positive["complete"] is True

    negative = finalize_runway_result(
        cash=300_000,
        ocf_raw=-450_000,
        ocf_quarterly=-150_000,
        period_months=3,
        assumption="",
        note="test",
        form_type="10-Q",
        units_scale=1,
        status=None,
        source_tags=None,
    )

    assert negative["status"] == "OK"
    assert negative["quarterly_burn"] == 150_000
    assert math.isclose(negative["runway_quarters"], 2.0)
    assert math.isclose(negative["runway_months_display"], 6.0)
    assert negative["complete"] is True
