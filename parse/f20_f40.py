"""Parser for 20-F and 40-F filings."""
from __future__ import annotations

from .q10_k10 import parse as _parse_q10_k10


def parse(url: str, html: str | None = None, form_hint: str | None = None) -> dict:
    return _parse_q10_k10(url, html=html, form_hint=form_hint)


__all__ = ["parse"]
