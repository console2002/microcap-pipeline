from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse.router import _infer_form_type_from_url, _normalize_form_type


def test_normalize_form_type_handles_x_separators() -> None:
    sample = "https://www.sec.gov/Archives/edgar/data/1833141/000183314125000023/cybininc-form6xk09x11x2025.htm"
    assert _normalize_form_type(sample) == "6-K"
    assert _infer_form_type_from_url(sample) == "6-K"
def test_infer_form_type_handles_uppercase_hint() -> None:
    assert _normalize_form_type("FORM 10-Q") == "10-Q"
