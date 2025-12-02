import pandas as pd
import pytest

from app.weekly_deep_research import (
    DILUTION_CREATION,
    DILUTION_TERMINATION,
    DILUTION_UNKNOWN,
    classify_dilution_filing,
)


class _StubExhibit:
    def __init__(self, document_type: str, text: str):
        self.document_type = document_type
        self._text = text

    def text(self) -> str:
        return self._text


class _StubFiling:
    def __init__(self, text: str = "", exhibits: list[_StubExhibit] | None = None):
        self._text = text
        self.exhibits = exhibits or []

    def text(self) -> str:
        return self._text

    def html(self) -> str:
        return ""


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            "This prospectus covers an at-the-market offering and we may sell shares from time to time.",
            DILUTION_CREATION,
        ),
        (
            "The sales agreement has been terminated and no further sales will occur.",
            DILUTION_TERMINATION,
        ),
        (
            "We may offer up to 10,000,000 shares under this sales agreement, which may be terminated at any time.",
            DILUTION_CREATION,
        ),
        (
            "We may offer and sell up to 5,000,000 shares under this program. The sales agreement has been terminated.",
            DILUTION_TERMINATION,
        ),
        (
            "General corporate update without offering language.",
            DILUTION_UNKNOWN,
        ),
    ],
)
def test_classify_dilution_filing_main_text(payload: str, expected: str) -> None:
    filing = _StubFiling(text=payload)
    assert classify_dilution_filing(filing) == expected


def test_classify_dilution_filing_reads_exhibits() -> None:
    exhibit = _StubExhibit("EX-1.1", "We may offer and sell up to 10,000,000 shares.")
    filing = _StubFiling(text="No primary offering text.", exhibits=[exhibit])
    assert classify_dilution_filing(filing) == DILUTION_CREATION


def test_dilution_details_prefers_most_recent_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import weekly_deep_research as wdr

    class _EventFiling:
        def __init__(self, classification: str):
            self.classification = classification

    def fake_resolve(record: pd.Series) -> _EventFiling:
        return _EventFiling(record["classification_override"])

    def fake_classify(filing: _EventFiling) -> str:
        return filing.classification

    monkeypatch.setattr(wdr, "_resolve_filing_from_record", fake_resolve)
    monkeypatch.setattr(wdr, "classify_dilution_filing", fake_classify)

    filings = pd.DataFrame(
        [
            {
                "Form": "S-3",
                "FilingURL": "https://example.com/termination",
                "FilingDate": "2024-01-01",
                "classification_override": wdr.DILUTION_TERMINATION,
            },
            {
                "Form": "S-3",
                "FilingURL": "https://example.com/unknown",
                "FilingDate": "2024-02-01",
                "classification_override": wdr.DILUTION_UNKNOWN,
            },
        ]
    )

    result = wdr._dilution_details(filings, "Form")

    assert result["score"] == "High"
    assert result["key_filing_url"] == "https://example.com/unknown"
    assert result["last_event_date"] == "2024-02-01"
    assert result["overhang_removed"] is False
