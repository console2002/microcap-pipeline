import pandas as pd

from app.config import weekly_allowed_forms
from app.pipeline import prune_weekly_filings


def test_prune_weekly_filings_limits_per_form():
    base_date = pd.Timestamp("2024-01-01")
    rows = []

    for i in range(15):
        rows.append({"Ticker": "AAA", "Form": "8-K", "FiledAt": base_date + pd.Timedelta(days=i)})

    for i in range(8):
        rows.append({"Ticker": "AAA", "Form": "S-3", "FiledAt": base_date + pd.Timedelta(days=i)})

    for i in range(5):
        rows.append({"Ticker": "BBB", "Form": "10-Q", "FiledAt": base_date + pd.Timedelta(days=i)})

    df = pd.DataFrame(rows)

    pruned = prune_weekly_filings(df)

    assert len(pruned[pruned["Form"] == "8-K"]) <= 10
    assert len(pruned[pruned["Form"] == "S-3"]) <= 5
    assert len(pruned[pruned["Form"] == "10-Q"]) <= 3

    allowed = weekly_allowed_forms()
    assert set(pruned["Form"].unique()).issubset(allowed)
