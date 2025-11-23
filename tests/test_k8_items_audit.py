from types import SimpleNamespace

from parse import k8 as parse_k8

from app import build_watchlist


def _make_row(url: str) -> SimpleNamespace:
    return SimpleNamespace(CIK="0000000000", Ticker="TEST", FiledAt="2024-12-31", URL=url)


def test_items_single_502_ignore_reason_populates_items():
    html = "<html><body>Item 5.02 Departure of directors.</body></html>"
    url = "file:///tmp/item502.htm"
    result = parse_k8.parse(url, html=html)

    event = build_watchlist._build_eight_k_event(_make_row(url), html, result)

    assert event is None
    items_present = build_watchlist._join_items(item.get("item", "") for item in result.get("items", []))
    assert items_present == "5.02"
    assert result["classification"]["ignore_reason"] == "Director/officer change only (Item 5.02)"


def test_items_508_801_combination_preserved():
    html = "<html><body>Item 5.08 Shareholder meeting. Item 8.01 Other Events.</body></html>"
    url = "file:///tmp/item508_801.htm"
    result = parse_k8.parse(url, html=html)

    event = build_watchlist._build_eight_k_event(_make_row(url), html, result)

    assert event is None
    items_present = build_watchlist._join_items(item.get("item", "") for item in result.get("items", []))
    assert items_present == "5.08; 8.01"
    assert result["classification"]["ignore_reason"] == "Annual meeting / procedural disclosure only (Item 5.08/8.01)"


def test_items_202_901_earnings_ignore_reason():
    html = "<html><body>Item 2.02 Results of Operations. Item 9.01 Financial Statements.</body></html>"
    url = "file:///tmp/item202_901.htm"
    result = parse_k8.parse(url, html=html)

    event = build_watchlist._build_eight_k_event(_make_row(url), html, result)

    assert event is None
    items_present = build_watchlist._join_items(item.get("item", "") for item in result.get("items", []))
    assert items_present == "2.02; 9.01"
    assert result["classification"]["ignore_reason"] == "Earnings release only (Item 2.02/9.01)"
