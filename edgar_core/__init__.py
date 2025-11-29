"""Thin wrapper package for edgartools data-object integrations."""

from .runway import extract_runway_from_filing
from .eight_k import extract_8k_event
from .offerings import extract_offering
from .governance import extract_governance
from .insider import extract_insider_events, aggregate_insider_events

__all__ = [
    "extract_runway_from_filing",
    "extract_8k_event",
    "extract_offering",
    "extract_governance",
    "extract_insider_events",
    "aggregate_insider_events",
]
