import pandas as pd

from app.pipeline import log_weekly_filings_stats


def test_weekly_filings_logging_messages():
    df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Form": "8-K", "FiledAt": "2024-02-01"},
            {"Ticker": "AAA", "Form": "10-Q", "FiledAt": "2024-02-05"},
            {"Ticker": "BBB", "Form": "S-3", "FiledAt": "2024-02-03"},
        ]
    )

    messages: list[str] = []
    log_weekly_filings_stats(df, messages.append)

    assert any("WEEKLY_FILINGS_TICKER" in msg for msg in messages)
    summary_msgs = [msg for msg in messages if "WEEKLY_FILINGS_SUMMARY" in msg]
    assert summary_msgs
    assert any("rows=" in msg and "tickers=" in msg for msg in summary_msgs)
