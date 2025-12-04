from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def _tier_rank(tier_value: str) -> int:
    """Rank tiers with Tier-1 highest, then Tier-2, then everything else."""

    text = str(tier_value or "").lower()
    if "1" in text:
        return 2
    if "2" in text:
        return 1
    return 0


def first_non_na(*values):
    """Return the first value that is not None and not NaN."""

    for value in values:
        if value is None:
            continue
        if pd.isna(value):
            continue
        return value
    return None


def select_primary_catalyst(events: Sequence) -> Optional[pd.Series]:
    """Select the primary catalyst according to shared W2/W3 policy.

    Policy:
    - Prefer highest tier (Tier-1 > Tier-2 > Tier-None).
    - Within the same tier, pick the most recent event date.
    - If no events are provided, return None.
    """

    if events is None:
        return None

    if isinstance(events, pd.DataFrame):
        df = events.copy()
    else:
        df = pd.DataFrame(list(events))

    if df.empty:
        return None

    if "event_tier" in df.columns:
        df["event_tier"] = df["event_tier"]
    elif "Tier" in df.columns:
        df["event_tier"] = df["Tier"]
    elif "EventTier" in df.columns:
        df["event_tier"] = df["EventTier"]
    else:
        df["event_tier"] = ""

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"]
    elif "EventType" in df.columns:
        df["event_type"] = df["EventType"]
    else:
        df["event_type"] = ""

    date_series = pd.Series(pd.NaT, index=df.index, dtype=object)
    for col in ["event_date", "EventDate", "FilingDate"]:
        if col in df.columns:
            missing = date_series.isna()
            if missing.any():
                date_series = date_series.where(~missing, df[col]).infer_objects(copy=False)

    df["_event_date_value"] = date_series
    df["_event_dt"] = pd.to_datetime(date_series, errors="coerce")
    df["_tier_rank"] = df["event_tier"].apply(_tier_rank)

    sorted_df = df.sort_values(
        by=["_tier_rank", "_event_dt", "_event_date_value"],
        ascending=[False, False, False],
        na_position="last",
    )

    return sorted_df.iloc[0]

