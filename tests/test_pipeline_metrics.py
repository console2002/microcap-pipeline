import pandas as pd
import pytest

from app.pipeline_metrics import dropoff_stats, stage_stats


def test_stage_stats_basic_dicts():
    records = [
        {"Ticker": "AAA", "CIK": "1"},
        {"Ticker": "BBB", "CIK": "2"},
        {"Ticker": "CCC", "CIK": "3"},
    ]
    stats = stage_stats(records)
    assert stats["row_count"] == 3
    assert stats["distinct_count"] == 3


def test_stage_stats_deduplicates_pairs():
    records = [
        {"Ticker": "AAA", "CIK": "1"},
        {"Ticker": "AAA", "CIK": "1"},
        {"Ticker": "AAA", "CIK": "2"},
    ]
    stats = stage_stats(records)
    assert stats["row_count"] == 3
    assert stats["distinct_count"] == 2


def test_stage_stats_dataframe():
    df = pd.DataFrame(
        [
            {"Ticker": "AAA", "CIK": "1"},
            {"Ticker": "BBB", "CIK": "2"},
            {"Ticker": "AAA", "CIK": "1"},
        ]
    )
    stats = stage_stats(df)
    assert stats["row_count"] == 3
    assert stats["distinct_count"] == 2


def test_dropoff_stats_basic_case():
    from_records = [
        {"Ticker": "AAA", "CIK": "1"},
        {"Ticker": "BBB", "CIK": "2"},
        {"Ticker": "CCC", "CIK": "3"},
    ]
    to_records = [
        {"Ticker": "AAA", "CIK": "1"},
        {"Ticker": "BBB", "CIK": "2"},
    ]
    stats = dropoff_stats(from_records, to_records)
    assert stats["from_count"] == 3
    assert stats["to_count"] == 2
    assert stats["dropped_count"] == 1
    assert stats["dropped_pct"] == pytest.approx(33.3333, rel=1e-3)


def test_dropoff_stats_empty_from():
    stats = dropoff_stats([], [])
    assert stats == {"from_count": 0, "to_count": 0, "dropped_count": 0, "dropped_pct": 0.0}
