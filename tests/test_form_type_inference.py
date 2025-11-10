from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse import k6
from parse.router import _infer_form_type_from_url, _normalize_form_type


def test_normalize_form_type_handles_x_separators() -> None:
    sample = "https://www.sec.gov/Archives/edgar/data/1833141/000183314125000023/cybininc-form6xk09x11x2025.htm"
    assert _normalize_form_type(sample) == "6-K"
    assert _infer_form_type_from_url(sample) == "6-K"


def test_k6_prefers_explicit_hint_over_xbrl(monkeypatch: pytest.MonkeyPatch) -> None:
    html_text = "<html><head><title>Form 6-K</title></head><body>FORM 6-K</body></html>"

    fake_xbrl_result = {
        "complete": False,
        "note": "values parsed from XBRL",
        "status": "Missing OCF",
        "form_type": "40-F",
        "cash_raw": None,
        "ocf_raw": None,
        "ocf_quarterly_raw": None,
        "period_months": None,
        "units_scale": 1,
    }

    monkeypatch.setattr(k6, "derive_from_xbrl", mock.Mock(return_value=(fake_xbrl_result, "40-F")))
    monkeypatch.setattr(k6, "follow_exhibits_and_parse", mock.Mock(return_value={}))
    monkeypatch.setattr(
        k6,
        "parse_html_cashflow_sections",
        mock.Mock(
            return_value={
                "cash_value": None,
                "ocf_value": None,
                "period_months_inferred": None,
                "units_scale": 1,
                "evidence": "",
            }
        ),
    )

    result = k6.parse(
        "https://www.sec.gov/Archives/edgar/data/1833141/000183314125000023/cybininc-form6xk09x11x2025.htm",
        html=html_text,
        form_hint="6-K",
    )

    assert result.get("form_type") == "6-K"
