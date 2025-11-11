"""Tests for debug OCF logging helpers."""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse import logging as parse_logging


def test_maybe_record_debug_ocf_allows_ok_prefix(monkeypatch):
    """Ensure statuses that begin with OK are treated as successes."""

    recorded = []

    def fake_record(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(parse_logging, "_record_debug_ocf", fake_record)

    result = {
        "status": "OK (from exhibit 99.1)",
        "ocf_raw": 123,
        "ocf_quarterly": 456,
        "ocf_quarterly_raw": "789",
        "complete": True,
    }

    parse_logging._maybe_record_debug_ocf(
        url="https://example.com/filing",
        form_type="10-K",
        result=result,
        html_info=None,
        extra=None,
    )

    assert recorded == []
