"""Weekly W2 candidate shortlist builder.

The shortlist aligns with weekly.txt W2 Output B and is the canonical
``20_candidate_shortlist.csv`` used by W3/W4. Inputs:
    - 01_universe_gated.csv (identity + core gates)
    - 09_events.csv (normalized events from W2)
    - price/cap/ADV fields from hydrated/shortlist caches
Output columns preserve legacy fields used by W3 while adding W2.B aliases
such as Venue, Price($), Cap($M), ADV20(k), CatalystType, EventDate, and
Primary/Secondary sources.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from app.csv_names import csv_path

MAX_CANDIDATES = 40


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def _tier_rank(value: str) -> int:
    text = str(value or "").lower()
    if "1" in text:
        return 3
    if "2" in text:
        return 2
    return 1 if text else 0


def _select_primary_event(events: pd.DataFrame) -> dict:
    if events.empty:
        return {}
    events = events.copy()
    events["event_tier"] = events.get("event_tier", events.get("Tier", "Other"))
    events["event_date"] = events.get("event_date")
    events["event_type"] = events.get("event_type", events.get("EventType", ""))

    events["_tier_rank"] = events["event_tier"].apply(_tier_rank)
    events["_event_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    events.sort_values([
        "_tier_rank",
        "_event_dt",
        "event_date",
    ], ascending=[False, False, False], inplace=True)
    primary = events.iloc[0]
    return {
        "CatalystType": primary.get("event_type", ""),
        "EventDate": primary.get("event_date", ""),
        "EventTier": primary.get("event_tier", ""),
        "PrimarySource": primary.get("primary_source_url", primary.get("PrimarySource", "")),
        "SecondarySource": primary.get(
            "secondary_source_url", primary.get("SecondarySource", "")
        ),
    }


def _notes_status(price, cap, adv) -> str:
    try:
        float(price)
        float(cap)
        float(adv)
    except (TypeError, ValueError):
        return "TBD — exclude"
    return "Pass"


def build_candidate_shortlist(data_dir: str) -> pd.DataFrame:
    """Build the W2.B candidate shortlist and write 20_candidate_shortlist.csv."""

    universe = _load_csv(os.path.join(data_dir, "01_universe_gated.csv"))
    events = _load_csv(os.path.join(data_dir, "09_events.csv"))
    hydrated = _load_csv(csv_path(data_dir, "hydrated_candidates"))
    legacy_short = _load_csv(csv_path(data_dir, "shortlist_candidates"))

    if universe.empty or events.empty:
        shortlist = pd.DataFrame()
        shortlist.to_csv(os.path.join(data_dir, "20_candidate_shortlist.csv"), index=False)
        return shortlist

    base_cols = ["Ticker", "Company", "CIK", "Sector", "Industry", "Exchange", "MarketCap", "ADV20", "Close", "Price"]
    base = universe.copy()
    for col in base_cols:
        if col not in base.columns:
            base[col] = None

    if not hydrated.empty:
        base = base.merge(
            hydrated[[c for c in ["Ticker", "Close", "ADV20", "MarketCap"] if c in hydrated.columns]],
            on="Ticker",
            how="left",
            suffixes=("", "_hyd"),
        )
    if not legacy_short.empty:
        base = base.merge(
            legacy_short[[c for c in ["Ticker", "Close", "ADV20", "MarketCap"] if c in legacy_short.columns]],
            on="Ticker",
            how="left",
            suffixes=("", "_legacy"),
        )

    def _coalesce(row, fields: Iterable[str]):
        for field in fields:
            val = row.get(field)
            if pd.notna(val):
                return val
        return None

    event_groups = events.groupby("Ticker")
    rows: list[dict] = []
    for _, base_row in base.iterrows():
        ticker = base_row.get("Ticker")
        if pd.isna(ticker):
            continue
        ticker_events = event_groups.get_group(ticker) if ticker in event_groups.groups else pd.DataFrame()
        if ticker_events.empty:
            continue
        event_info = _select_primary_event(ticker_events)
        price_val = _coalesce(base_row, ["Price", "Close_hyd", "Close_legacy", "Close"])
        cap_val_raw = _coalesce(base_row, ["MarketCap", "MarketCap_hyd", "MarketCap_legacy"])
        adv_val_raw = _coalesce(base_row, ["ADV20", "ADV20_hyd", "ADV20_legacy"])
        cap_val = pd.to_numeric(cap_val_raw, errors="coerce") if cap_val_raw is not None else None
        adv_val = pd.to_numeric(adv_val_raw, errors="coerce") if adv_val_raw is not None else None
        cap_musd = cap_val / 1_000_000 if pd.notna(cap_val) else None
        adv_k = adv_val / 1_000 if pd.notna(adv_val) else None
        notes_status = _notes_status(price_val, cap_val, adv_val)

        row = {
            "Ticker": ticker,
            "Company": base_row.get("Company", ""),
            "CIK": base_row.get("CIK", ""),
            "Sector": base_row.get("Sector", ""),
            "Industry": base_row.get("Industry", ""),
            "Venue": base_row.get("Exchange", ""),
            "Price": price_val,
            "Price($)": price_val,
            "MarketCap": cap_val,
            "Cap($M)": cap_musd,
            "Cap_Musd": cap_musd,
            "ADV20": adv_val,
            "ADV20(k)": adv_k,
            "ADV20_k": adv_k,
            "PrimaryCatalystType": event_info.get("CatalystType", ""),
            "CatalystType": event_info.get("CatalystType", ""),
            "PrimaryCatalystDate": event_info.get("EventDate", ""),
            "EventDate": event_info.get("EventDate", ""),
            "EventTier": event_info.get("EventTier", ""),
            "PrimaryFilingURL": event_info.get("PrimarySource", ""),
            "PrimarySource": event_info.get("PrimarySource", ""),
            "SecondarySource": event_info.get("SecondarySource", ""),
            "NotesStatus": notes_status,
            "Notes/Status": notes_status,
        }
        rows.append(row)

    shortlist = pd.DataFrame(rows)
    if shortlist.empty:
        shortlist.to_csv(os.path.join(data_dir, "20_candidate_shortlist.csv"), index=False)
        return shortlist

    shortlist["_tier_order"] = shortlist["EventTier"].apply(_tier_rank)
    shortlist.sort_values(
        by=["_tier_order", "EventDate", "Cap($M)", "Ticker"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    shortlist = shortlist.drop(columns=["_tier_order"])
    shortlist = shortlist.head(MAX_CANDIDATES)
    shortlist.to_csv(os.path.join(data_dir, "20_candidate_shortlist.csv"), index=False)
    return shortlist

