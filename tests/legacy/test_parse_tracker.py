import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parse_progress import ParseProgressTracker


def test_parse_tracker_compute_counts_once_without_date() -> None:
    tracker = ParseProgressTracker(on_change=None)
    tracker.reset()

    tracker.process_message(
        "parse_q10 [INFO] compute_runway: TEST cash=0 ocf=0 months=6 scale=1 est=interim form=10-Q"
    )
    assert "10-Q" in tracker.form_stats
    assert tracker.form_stats["10-Q"].parsed == 1

    # Duplicate compute for the same ticker without a date should not double count
    tracker.process_message(
        "parse_q10 [INFO] compute_runway: TEST cash=0 ocf=0 months=6 scale=1 est=interim form=10-Q"
    )
    assert tracker.form_stats["10-Q"].parsed == 1


def test_parse_tracker_exhibit_tracking() -> None:
    tracker = ParseProgressTracker(on_change=None)
    tracker.reset()

    tracker.process_message(
        "parse_q10 [INFO] RECT fetching 6-K filed 2025-07-31T00:00:00 url https://example.com/exhibit.htm"
    )
    assert "6-K" in tracker.form_stats
    assert tracker.form_stats["6-K"].parsed == 0

    tracker.process_message(
        "parse_q10 [INFO] compute_runway: RECT cash=3094839.000000 ocf=37213.500000 months=12 scale=1 est=interim form=6-K date=2025-07-31"
    )
    assert tracker.form_stats["6-K"].parsed == 1

    tracker.process_message("parse_q10 [WARN] RECT runway status Missing OCF (from exhibit)")
    assert tracker.form_stats["6-K"].missing == 1

    tracker.process_message(
        "parse_q10 [INFO] compute_runway: RECT cash=3094839.000000 ocf=37213.500000 months=12 scale=1 est=interim form=6-K date=2025-09-30"
    )
    assert tracker.form_stats["6-K"].parsed == 2

    tracker.process_message("parse_q10 [OK] RECT runway status OK (from exhibit)")
    assert tracker.form_stats["6-K"].valid == 1

    # A repeated status for the same filing should not change counts
    tracker.process_message("parse_q10 [OK] RECT runway status OK (from exhibit)")
    assert tracker.form_stats["6-K"].valid == 1


def test_parse_tracker_incomplete_counts_for_filings() -> None:
    tracker = ParseProgressTracker(on_change=None)
    tracker.reset()

    tracker.process_message(
        "parse_q10 [INFO] CYBN fetching 6-K filed 2025-10-31T00:00:00 url https://www.sec.gov/ix?doc=/example.htm"
    )
    tracker.process_message("parse_q10 [WARN] CYBN 6-K incomplete: Missing OCF")

    assert tracker.form_stats["6-K"].missing == 1
    assert tracker.form_stats["6-K"].parsed == 1

    # Duplicate message for the same filing should not double count
    tracker.process_message("parse_q10 [WARN] CYBN 6-K incomplete: Missing OCF")
    assert tracker.form_stats["6-K"].missing == 1
    assert tracker.form_stats["6-K"].parsed == 1

    # The subsequent compute message for the same filing should not increment parsed again
    tracker.process_message(
        "parse_q10 [INFO] compute_runway: CYBN cash=0 ocf=0 months=6 scale=1 est=interim form=6-K date=2025-10-31"
    )
    assert tracker.form_stats["6-K"].missing == 1
    assert tracker.form_stats["6-K"].parsed == 1

    # A new filing with a different date should be counted separately
    tracker.process_message(
        "parse_q10 [INFO] CYBN fetching 6-K filed 2025-12-31T00:00:00 url https://www.sec.gov/ix?doc=/example2.htm"
    )
    tracker.process_message("parse_q10 [WARN] CYBN 6-K incomplete: Missing OCF")

    assert tracker.form_stats["6-K"].missing == 2
    assert tracker.form_stats["6-K"].parsed == 2


def test_parse_tracker_handles_placeholders_and_unspecified() -> None:
    tracker = ParseProgressTracker(on_change=None)
    tracker.reset()

    assert tracker.form_stats["S-3"].note == "NI"
    assert tracker.form_stats["FORM 4"].note == "NI"

    tracker.process_message(
        "parse_q10 [INFO] compute_runway: NOFORM cash=0 ocf=0 months=6 scale=1 est=interim date=2024-01-01"
    )

    assert tracker.form_stats["Unspecified"].parsed == 1
    assert tracker.form_stats["Unspecified"].note == "Missing form detail"

    tracker.process_message(
        "parse_q10 [INFO] SAMPLE fetching S-3 filed 2024-01-02 url https://example.com"
    )
    assert tracker.form_stats["S-3"].note == ""
