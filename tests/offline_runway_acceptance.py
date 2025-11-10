# Standard library dependencies required for fixture discovery and numeric checks.
import math
import sys
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parser_10q import get_runway_from_filing


TOLERANCE = 0.01


def _assert_close(actual: Optional[float], expected: float, *, tolerance: float = TOLERANCE) -> None:
    assert actual is not None, "value was None"
    if expected == 0:
        assert abs(actual - expected) <= tolerance
    else:
        assert math.isclose(actual, expected, rel_tol=tolerance), f"{actual} !~= {expected}"


CASES = [
    (
        "ex_837171.htm",
        "6-K",
        {
            "period": 6,
            "ocf": -8_619_000.0,
            "cash": 6_145_000.0,
            "runway_quarters": 1.43,
            "runway_months": 4.28,
        },
    ),
    (
        "ex_866936.htm",
        "6-K",
        {
            "period": 9,
            "ocf": -16_239_000.0,
            "cash": 6_521_000.0,
            "runway_quarters": 1.20,
            "runway_months": 3.61,
        },
    ),
    (
        "000175392625001675/g084981_20f.htm",
        "20-F",
        {
            "period": 12,
            "ocf": -9_390_622.0,
            "cash": 17_829_149.0,
            "runway_quarters": 7.59,
            "runway_months": 22.78,
        },
    ),
    (
        "ea025948901ex99-1_brera.htm",
        "6-K",
        {
            "period": 6,
            "ocf": -3_160_656.0,
            "cash": 658_136.0,
            "runway_quarters": 0.42,
            "runway_months": 1.25,
        },
    ),
]


@pytest.mark.parametrize("filename, form, expected", CASES)
def test_offline_runway_acceptance(
    filename: str, form: str, expected: dict[str, float]
) -> None:
    base_path = Path(__file__).resolve().parent
    target = base_path / filename
    assert target.exists(), f"missing test fixture: {target}"

    uri = target.as_uri()
    if form:
        uri = f"{uri}?form={form}"

    result = get_runway_from_filing(uri)

    period = result.get("period_months")
    ocf = result.get("ocf_raw")
    cash = result.get("cash_raw")
    runway_q = result.get("runway_quarters_raw")
    runway_q_display = result.get("runway_quarters_display")
    runway_m_display = result.get("runway_months_display")

    assert period == expected["period"]
    _assert_close(ocf, expected["ocf"])
    _assert_close(cash, expected["cash"])
    _assert_close(runway_q, expected["runway_quarters"])
    if runway_q_display is not None:
        _assert_close(runway_q_display, expected["runway_quarters"], tolerance=0.01)
    if runway_m_display is not None:
        _assert_close(runway_m_display, expected["runway_months"], tolerance=0.01)

    display_ocf = int(round(ocf)) if ocf is not None else None
    display_cash = int(round(cash)) if cash is not None else None
    display_runway_q = (
        round(runway_q_display, 2)
        if runway_q_display is not None
        else round(runway_q or 0.0, 2)
    )
    display_runway_m = (
        round(runway_m_display, 2)
        if runway_m_display is not None
        else round((runway_q or 0.0) * 3, 2)
    )

    print(
        "offline_accept: "
        f"file={filename} "
        f"period={period}m "
        f"ocf={display_ocf} "
        f"cash={display_cash} "
        f"runway_q={display_runway_q:.2f} "
        f"runway_m={display_runway_m:.2f} PASS"
    )
