import pytest

from edgar_core.eight_k import classify_eight_k_event


class StubEightK:
    def __init__(self, items, item_texts=None):
        self.items = list(items)
        self._item_texts = item_texts or {}

    def __getitem__(self, key):
        return self._item_texts.get(key, "")


class StubFiling:
    def __init__(self, exhibits=None):
        self.exhibits = exhibits or []


def test_listing_change_excludes_annual_meeting_boilerplate():
    eight_k = StubEightK(
        items=["5.07"],
        item_texts={
            "5.07": "The Company held its annual meeting. Our common stock is listed on the Nasdaq Global Select Market.",
        },
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] != "ListingChange"
    assert result["event_tier"] != "Tier-1"


def test_listing_change_tier1_requires_listing_items_and_context():
    eight_k = StubEightK(
        items=["3.01"],
        item_texts={
            "3.01": "The Company announced the transfer of listing of its common stock to the Nasdaq Capital Market.",
        },
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] == "ListingChange"
    assert result["event_tier"] == "Tier-1"


def test_listing_change_allows_deficiency_notice_without_listing_word():
    eight_k = StubEightK(
        items=["3.01"],
        item_texts={
            "3.01": (
                "Nasdaq has notified the Company that its bid price does not meet the minimum bid price "
                "requirement under Nasdaq rules. The Company intends to submit a compliance plan to Nasdaq."
            )
        },
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] == "ListingChange"
    assert result["event_tier"] == "Tier-1"


@pytest.mark.parametrize(
    "items",
    [
        ["2.02"],
        ["2.02", "7.01"],
    ],
)
def test_earnings_items_with_exchange_boilerplate_are_not_listing_change(items):
    eight_k = StubEightK(
        items=items,
        item_texts={
            items[0]: "Press release announcing financial results. Our common stock is listed on Nasdaq under the symbol XYZ.",
        },
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] != "ListingChange"
    assert result["event_tier"] != "Tier-1"


def test_contract_award_classification_unchanged():
    eight_k = StubEightK(
        items=["8.01"],
        item_texts={"8.01": "The Company received a new contract award from the Department of Defense."},
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] == "ContractAward"
    assert result["event_tier"] == "Tier-1"


def test_spinoff_with_soft_item_classifies_as_listing_change():
    eight_k = StubEightK(
        items=["8.01"],
        item_texts={
            "8.01": (
                "The Company announced a planned spin-off of its XYZ business into a separately traded public company "
                "on the Nasdaq Capital Market."
            )
        },
    )
    filing = StubFiling()

    result = classify_eight_k_event(eight_k, filing)

    assert result["event_type"] == "ListingChange"
    assert result["event_tier"] == "Tier-1"
