"""8-K extraction helpers using edgartools data objects.

The classifier consumes :class:`edgar.company_reports.EightK` and :class:`edgar.Filing`
objects (both produced by ``Filing.obj()``) and emits the normalized ``event_type``
and ``event_tier`` fields written to ``09_events.csv``. Downstream stages reuse
the same labels to populate ``PrimaryCatalystType/Date/URL`` in
``20_candidate_shortlist.csv`` and later W3/W4 CSVs.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

from edgar import Filing
from edgar.company_reports import EightK

logger = logging.getLogger(__name__)


TIER1_WHITELIST = {
    "ATM_TERMINATION",
    "SHELF_TERMINATION",
    "SALES_AGREEMENT_TERMINATION",
    "FDA",
    "ContractAward",
    "GuidanceUp",
    "ListingChange",
}


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in keywords)


def _gather_item_text(eight_k: EightK) -> list[str]:
    texts: list[str] = []
    for label in getattr(eight_k, "items", []) or []:
        try:
            value = eight_k[label] or ""
        except Exception:
            value = ""
        if value and str(value).strip():
            texts.append(str(value))
    return texts


def _gather_exhibit_texts(filing: Filing) -> list[str]:
    snippets: list[str] = []
    for exhibit in getattr(filing, "exhibits", []) or []:
        parts = [
            getattr(exhibit, "document_type", "") or "",
            getattr(exhibit, "document", "") or "",
            getattr(exhibit, "description", "") or "",
        ]
        snippet = " ".join(part for part in parts if part).strip()
        if snippet:
            snippets.append(snippet)
        try:
            text_val = exhibit.text()  # type: ignore[attr-defined]
            if text_val and str(text_val).strip():
                snippets.append(str(text_val))
        except Exception:
            continue
    return snippets


def _normalize_item_label(label: str) -> str:
    """Normalize an item heading such as ``"Item 2.02 Results"`` to ``"2.02"``."""

    lowered = label.lower().strip()
    if lowered.startswith("item"):
        lowered = lowered[len("item") :].strip()
    return lowered.lstrip(" .:-")


def classify_eight_k_event(
    eight_k: EightK, filing: Filing, press_text: str | None = None
) -> Dict[str, object]:
    """Classify an 8-K into a normalized event bucket and tier.

    Inputs are the parsed edgartools objects so that item headings, item text,
    and exhibit descriptions drive the labels. The output is a dictionary with
    ``event_type``, ``event_tier``, ``is_dilution``, ``dilution_tags``,
    ``tier1_type``, and ``tier1_trigger`` keys.
    """

    items = [
        _normalize_item_label(str(item)) for item in getattr(eight_k, "items", []) or []
    ]
    item_text = "\n\n".join(_gather_item_text(eight_k))
    exhibits_text = "\n\n".join(_gather_exhibit_texts(filing))
    combined_text = "\n\n".join(
        part for part in [item_text, exhibits_text, press_text or ""] if part
    ).lower()

    event_type = "OtherEvent"
    event_tier = "Other"
    tier1_type = ""
    tier1_trigger = ""
    dilution_tags: list[str] = []
    is_dilution = False

    atm_terms = [
        "at-the-market",
        "atm",
        "equity distribution",
        "equity offering",
        "sales agreement",
        "equity line",
    ]
    atm_creation_terms = [
        "enter into a sales agreement",
        "entered into a sales agreement",
        "at-the-market offering",
        "at-the-market equity offering",
        "equity distribution agreement",
        "sales agreement with",
    ]
    atm_termination_terms = [
        "terminate",
        "termination",
        "suspend",
        "expires",
        "no further sales",
        "terminated",
    ]
    financing_terms = [
        "registered direct",
        "underwritten offering",
        "subscription agreement",
        "securities purchase agreement",
        "convertible",
        "warrant",
        "offering",
    ]

    atm_context = _contains_any(combined_text, atm_terms + atm_creation_terms)
    atm_creation = atm_context and _contains_any(combined_text, atm_creation_terms)
    atm_termination = atm_context and _contains_any(combined_text, atm_termination_terms)

    if atm_termination:
        event_type = "ATM_TERMINATION"
        event_tier = "Tier-1"
        tier1_type = "ATM/Shelf Termination"
        tier1_trigger = "Termination language"
    elif atm_creation:
        event_type = "ATM_CREATION"
        event_tier = "Tier-2"
        is_dilution = True
        dilution_tags.append("atm")
    elif _contains_any(combined_text, financing_terms):
        event_type = "GENERIC_FINANCING"
        event_tier = "Tier-2"
        is_dilution = True
        dilution_tags.append("financing")
    elif _contains_any(
        combined_text, ["fda", "clearance", "approval", "510(k)", "pdufa", "de novo"]
    ):
        event_type = "FDA"
        event_tier = "Tier-1"
    elif _contains_any(
        combined_text, ["guidance", "raise guidance", "increase guidance", "upward guidance"]
    ):
        event_type = "GuidanceUp"
        event_tier = "Tier-1"
    elif _contains_any(
        combined_text, ["contract", "award", "purchase order", "task order", "funded"]
    ):
        event_type = "ContractAward"
        event_tier = "Tier-1"
    elif _contains_any(combined_text, ["uplist", "listing", "nyse", "nasdaq", "spinoff", "spin-off"]):
        event_type = "ListingChange"
        event_tier = "Tier-1"
    elif any(str(item).startswith("2.02") for item in items):
        event_type = "Earnings"
        event_tier = "Tier-2"
    elif any(str(item).startswith("5.02") for item in items):
        event_type = "ManagementChange"
        event_tier = "Tier-2"
    elif items:
        event_tier = "Tier-2"

    # Enforce a narrow Tier-1 whitelist to avoid false positives from generic ATM filings.
    if event_tier == "Tier-1" and event_type not in TIER1_WHITELIST:
        event_tier = "Tier-2"
    if event_type in {"ATM_TERMINATION", "SHELF_TERMINATION", "SALES_AGREEMENT_TERMINATION"}:
        event_tier = "Tier-1"

    return {
        "event_type": event_type,
        "event_tier": event_tier,
        "is_dilution": is_dilution,
        "dilution_tags": dilution_tags,
        "tier1_type": tier1_type,
        "tier1_trigger": tier1_trigger,
    }


def classify_event(eight_k: EightK, filing: Filing, press_text: str | None = None) -> Tuple[str, str]:
    """Return the ``(event_type, event_tier)`` tuple for legacy callers."""

    result = classify_eight_k_event(eight_k, filing, press_text)
    return result["event_type"], result["event_tier"]


def extract_8k_event(filing: Filing) -> Dict:
    logger.debug(
        "extract_8k_event entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    parse_status = "no_data"
    items_raw = []
    items_normalized = []
    has_press_release = False
    has_exhibits = False
    event_date = getattr(filing, "filing_date", None)
    event_text_summary = ""

    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed", exc_info=True)

    if obj is not None:
        parse_status = "edgar_obj_ok"
        items_raw = list(getattr(obj, "items", []) or [])
        items_normalized = [str(item) for item in items_raw]
        has_press_release = bool(getattr(obj, "has_press_release", False))
        has_exhibits = bool(getattr(obj, "press_releases", []))
        event_date = getattr(obj, "date_of_report", event_date)
        first_item = items_raw[0] if items_raw else None
        if isinstance(first_item, str):
            event_text_summary = first_item

    result = {
        "form": getattr(filing, "form", "8-K"),
        "cik": getattr(filing, "cik", ""),
        "ticker": getattr(filing, "ticker", ""),
        "filing_date": getattr(filing, "filing_date", ""),
        "event_date": event_date,
        "items_raw": items_raw,
        "items_normalized": items_normalized,
        "has_press_release": has_press_release,
        "has_exhibits": has_exhibits,
        "event_text_summary": event_text_summary,
        "parse_status": parse_status,
    }

    logger.debug(
        "extract_8k_event exit", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", ""), "parse_status": parse_status}
    )
    return result
