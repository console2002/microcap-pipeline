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
