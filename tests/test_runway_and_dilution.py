import os

from app.weekly_deep_research import compute_runway_from_html, run_weekly_deep_research


def test_compute_runway_from_html_fixture():
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "runway_sample.html")
    with open(fixture, "r", encoding="utf-8") as handle:
        html = handle.read()
    quarters = compute_runway_from_html(html)
    # cash 12,000,000 burn 3,000,000 => 4.0 quarters
    assert quarters == 4.0
