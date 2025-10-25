"""Mapping between FDA entity names and internal market identifiers.

This module exists because openFDA queries must use the FDA's applicant or
manufacturer name strings, not our internal tickers or CIKs. The
``FDA_COMPANY_MAP`` list links the FDA-facing names to our identifiers so we can
look up filings by the correct company name. The ``Confidence`` field gates
whether we query a mapping upstream: only entries marked "high" or "med" are
queried.
"""

FDA_COMPANY_MAP = [
    {
        "FDA_EntityName": "TODO COMPANY NAME INC",
        "Ticker": "TODO",
        "CIK": "0000000000",
        "Confidence": "high",
    },
    {
        "FDA_EntityName": "ANOTHER TODO MANUFACTURER LLC",
        "Ticker": "ANOT",
        "CIK": "0000000001",
        "Confidence": "med",
    },
]


def get_fda_entities_for_tickers(tickers: list[str]) -> list[dict]:
    """Return unique FDA entity mappings for the provided tickers."""
    if not tickers:
        return []

    ticker_set = {ticker.upper() for ticker in tickers}
    seen_entities: set[str] = set()
    results: list[dict] = []

    for entry in FDA_COMPANY_MAP:
        if entry.get("Confidence") == "low":
            continue

        entry_ticker = entry.get("Ticker", "").upper()
        if entry_ticker not in ticker_set:
            continue

        entity_name = entry.get("FDA_EntityName")
        if entity_name in seen_entities:
            continue

        results.append(entry)
        seen_entities.add(entity_name)

    return results
