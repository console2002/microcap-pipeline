import types

import pytest

from app import build_watchlist
from app.eight_k_parser import EdgarEightKParser
from parse.router import MissingAdapterError, _fetch_url


class DummyAdapter:
    def __init__(self):
        self.requested = []

    def download_filing_text(self, url: str):
        self.requested.append(url)
        return "dummy"


def test_fetch_url_missing_adapter(monkeypatch):
    monkeypatch.setattr("parse.router.get_adapter", lambda: None)

    with pytest.raises(MissingAdapterError) as exc:
        _fetch_url("https://example.com/test")

    assert "adapter is None" in str(exc.value)
    assert "example.com/test" in str(exc.value)


def test_eight_k_parser_uses_adapter(monkeypatch):
    adapter = DummyAdapter()
    parser = EdgarEightKParser(adapter=adapter)

    monkeypatch.setattr(
        "app.eight_k_parser._parse_accession_from_url",
        lambda url: ("0000001234", "0000001234-24-000001"),
    )

    expected_url = parser._build_text_url("0000001234", "0000001234-24-000001")

    class DummyPress:
        def text(self):
            return "Press"

        def url(self):
            return "http://example.com/press"

    class DummyEightK:
        def __init__(self):
            self.items = ["Item 1.01", "Item 9.01"]
            self.press_releases = [DummyPress()]

        def __getitem__(self, key):
            if str(key).startswith("Item 1"):
                return "Event text"
            return ""

    class DummyFiling:
        def __init__(self, text_url):
            self.form = "8-K"
            self.exhibits = []
            self.filing_url = text_url
            self.text_url = text_url
            self.full_text_submission = "RAW"

        def obj(self):
            return DummyEightK()

        def xbrl(self):
            return None

    monkeypatch.setattr(
        "app.eight_k_parser.Filing.from_sgml_text", lambda raw: DummyFiling(expected_url)
    )

    result, error = parser.parse("https://example.com/8k")

    assert error == ""
    assert result is not None
    assert adapter.requested == [expected_url]


def test_process_eight_k_row_handles_missing_adapter(monkeypatch):
    def fake_parser():
        class DummyParser:
            def parse(self, url, form_hint=""):
                raise MissingAdapterError("adapter missing")

        return DummyParser()

    monkeypatch.setattr(build_watchlist, "_get_eight_k_parser", fake_parser)

    row = types.SimpleNamespace(
        URL="http://example.com/8k",
        Ticker="ABC",
        CIK="1234567890",
        AccessionNo="0000001234-24-000001",
    )

    result = build_watchlist._process_eight_k_row(row)

    assert result.event is None
    assert result.debug_entry[2] == "missing_adapter"
    assert "missing_adapter" in " ".join(result.log_messages)
