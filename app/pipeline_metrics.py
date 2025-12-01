"""Lightweight helpers for computing pipeline stage metrics."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is available in runtime/tests
    pd = None


def _coerce_to_records(records) -> Sequence:
    if records is None:
        return []
    if pd is not None and isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        if isinstance(records, list):
            return records
        return list(records)
    return []


def _distinct_count(records, key_fields: tuple[str, ...]) -> int:
    if pd is not None and isinstance(records, pd.DataFrame):
        if records.empty:
            return 0
        if not set(key_fields).issubset(records.columns):
            return 0
        subset = records[list(key_fields)].dropna()
        if subset.empty:
            return 0
        # Normalize to string to prevent duplicates due to type variance.
        for field in key_fields:
            subset[field] = subset[field].astype(str)
        return int(len(subset.drop_duplicates()))

    if not records:
        return 0

    combos: set[tuple] = set()
    for row in records:
        if not isinstance(row, Mapping):
            continue
        try:
            values = tuple(row.get(field) for field in key_fields)
        except Exception:
            continue
        if any(val is None for val in values):
            continue
        combos.add(tuple(str(val) for val in values))
    return len(combos)


def stage_stats(records, key_fields: tuple[str, ...] = ("Ticker", "CIK")) -> dict:
    """Return basic metrics for a stage payload.

    The function accepts either a pandas DataFrame or an iterable of mapping
    objects (e.g., list of dicts).
    """

    normalized = _coerce_to_records(records)
    row_count = len(normalized) if normalized is not None else 0
    distinct_count = _distinct_count(normalized, key_fields)
    return {"row_count": row_count, "distinct_count": distinct_count}


def dropoff_stats(
    from_records, to_records, key_fields: tuple[str, ...] = ("Ticker", "CIK")
) -> dict:
    """Compute drop-off statistics between two stages."""

    from_stats = stage_stats(from_records, key_fields=key_fields)
    to_stats = stage_stats(to_records, key_fields=key_fields)

    from_count = from_stats.get("distinct_count", 0)
    to_count = to_stats.get("distinct_count", 0)
    dropped_count = max(from_count - to_count, 0)
    dropped_pct = 0.0
    if from_count > 0:
        dropped_pct = (dropped_count / from_count) * 100.0

    return {
        "from_count": from_count,
        "to_count": to_count,
        "dropped_count": dropped_count,
        "dropped_pct": dropped_pct,
    }


__all__ = ["stage_stats", "dropoff_stats"]
