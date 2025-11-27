import os

from app.weekly_deep_research import compute_runway_from_html, run_weekly_deep_research


def test_compute_runway_from_html_fixture():
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "runway_sample.html")
    with open(fixture, "r", encoding="utf-8") as handle:
        html = handle.read()
    quarters = compute_runway_from_html(html)
    # cash 12,000,000 burn 3,000,000 => 4.0 quarters
    assert quarters == 4.0


def test_runway_fetches_http(monkeypatch, tmp_path):
    class FakeAdapter:
        def download_filing_text(self, url: str) -> str:  # pragma: no cover - simple stub
            assert url == "https://example.com/filing"
            return "Cash and cash equivalents: $10,000\nNet cash used in operating activities: $(2,000)"

    monkeypatch.setattr("app.weekly_deep_research.get_adapter", lambda cfg=None: FakeAdapter())

    data_dir = tmp_path
    filings_csv = data_dir / "02_filings.csv"
    shortlist_csv = data_dir / "20_candidate_shortlist.csv"

    filings_csv.write_text(
        "Ticker,CIK,FormType,FilingURL\nAAA,0000000003,10-Q,https://example.com/filing",
        encoding="utf-8",
    )
    shortlist_csv.write_text(
        "Ticker,Company,CIK\nAAA,Alpha,0000000003",
        encoding="utf-8",
    )

    df = run_weekly_deep_research(str(data_dir))
    assert not df.empty
    assert df.iloc[0]["RunwayQuarters"] == 5.0
