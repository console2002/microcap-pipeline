"""Compatibility shim for legacy parser imports."""
from __future__ import annotations

from parse.router import get_runway_from_filing, url_matches_form

__all__ = ["get_runway_from_filing", "url_matches_form"]
