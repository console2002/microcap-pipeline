"""Helpers for biotech detection and peer read-through classification.

The helpers in this module are intentionally conservative and depend only on
existing EDGAR-derived CSV inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd

# Substrings that imply a biotech / biopharma business model.
_BIOTECH_KEYWORDS = {
    "biotech",
    "biotechnology",
    "biopharma",
    "biopharmaceutical",
    "pharma",
    "pharmaceutical",
    "drug manufacturers",
    "biological product",
}


def is_biotech(sector: str | None, industry: str | None) -> bool:
    """Return True if the sector/industry suggest a biotech ticker."""

    combined = f"{sector or ''} {industry or ''}".lower()
    return any(keyword in combined for keyword in _BIOTECH_KEYWORDS)


def get_biotech_peers(
    universe_df: pd.DataFrame,
    ticker: str,
    sector: str | None,
    industry: str | None,
    max_peers: int = 20,
) -> List[str]:
    """Return a list of peer biotech tickers from the supplied universe."""

    if universe_df is None or universe_df.empty:
        return []

    universe = universe_df.copy()
    universe["Sector"] = universe.get("Sector", pd.Series(dtype=str)).astype(str)
    universe["Industry"] = universe.get("Industry", pd.Series(dtype=str)).astype(str)

    biotech_mask = universe.apply(lambda r: is_biotech(r.get("Sector"), r.get("Industry")), axis=1)
    biotech_universe = universe[biotech_mask]
    if biotech_universe.empty:
        return []

    industry_lower = str(industry or "").lower()
    sector_lower = str(sector or "").lower()

    same_industry = biotech_universe[
        biotech_universe["Industry"].str.lower() == industry_lower
    ]

    if same_industry.empty:
        same_industry = biotech_universe[
            biotech_universe["Sector"].str.lower() == sector_lower
        ]

    peers = [str(tkr) for tkr in same_industry.get("Ticker", []) if str(tkr) != str(ticker)]
    return peers[:max_peers]


# Lightweight sentiment classification for peer events.
_POSITIVE_KEYWORDS = {
    "approval",
    "successful",
    "positive",
    "fast track",
    "orphan",
    "breakthrough",
    "license",
    "licensing",
    "partnership",
    "phase ii",
    "phase iii",
    "ph ii",
    "ph iii",
}

_NEGATIVE_KEYWORDS = {
    "crl",
    "clinical hold",
    "hold",
    "fail",
    "failure",
    "terminated",
    "termination",
    "discontinue",
    "discontinued",
    "delay",
    "negative",
}


@dataclass
class PeerEventClassification:
    code: str
    evidence: str


def classify_peer_events(events: pd.DataFrame) -> PeerEventClassification:
    """Classify peer events into POSITIVE/NEGATIVE/MIXED/NONE."""

    if events is None or events.empty:
        return PeerEventClassification("NONE", "")

    positive_rows = []
    negative_rows = []

    def _sentiment(text: str | None, tier: str | None) -> str | None:
        lower = str(text or "").lower()
        tier_lower = str(tier or "").lower()
        if any(key in lower for key in _NEGATIVE_KEYWORDS):
            return "NEGATIVE"
        if any(key in lower for key in _POSITIVE_KEYWORDS):
            return "POSITIVE"
        if "1" in tier_lower:
            return "POSITIVE"
        return None

    for _, rec in events.iterrows():
        text = rec.get("event_type") or rec.get("EventType") or rec.get("ItemsNormalized") or rec.get("ItemsPresent")
        tier = rec.get("Tier") or rec.get("event_tier") or rec.get("EventTier")
        sentiment = _sentiment(text, tier)
        if sentiment == "POSITIVE":
            positive_rows.append(rec)
        elif sentiment == "NEGATIVE":
            negative_rows.append(rec)

    if positive_rows and negative_rows:
        label = "MIXED"
        evidence_row = positive_rows[0]
    elif positive_rows:
        label = "POSITIVE"
        evidence_row = positive_rows[0]
    elif negative_rows:
        label = "NEGATIVE"
        evidence_row = negative_rows[0]
    else:
        return PeerEventClassification("NONE", "")

    ticker = evidence_row.get("Ticker", "")
    event_type = evidence_row.get("event_type") or evidence_row.get("EventType")
    date_val = (
        evidence_row.get("EventDate")
        or evidence_row.get("event_date")
        or evidence_row.get("FilingDate")
    )
    evidence = f"PEER={ticker}, EVENT={event_type}, DATE={date_val}".strip(", ")
    return PeerEventClassification(label, evidence)


__all__ = [
    "is_biotech",
    "get_biotech_peers",
    "classify_peer_events",
    "PeerEventClassification",
]
