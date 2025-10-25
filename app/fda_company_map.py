"""Mapping between FDA entity names and internal market identifiers.

This module exists because openFDA queries must use the FDA's applicant or
manufacturer name strings, not our internal tickers or CIKs. The
``FDA_COMPANY_MAP`` list links the FDA-facing names to our identifiers so we can
look up filings by the correct company name. The ``Confidence`` field gates
whether we query a mapping upstream: only entries marked "high" or "med" are
queried.
"""

import os

import pandas as pd

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


def ensure_csv(csv_path: str = "data/maps/FDA_Company_Map.csv") -> None:
    """Ensure the FDA company map CSV exists and is up to date."""

    directory = os.path.dirname(csv_path) or "."
    os.makedirs(directory, exist_ok=True)

    columns = ["FDA_EntityName", "Ticker", "CIK", "Confidence"]

    if not os.path.exists(csv_path):
        df = pd.DataFrame(FDA_COMPANY_MAP, columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"Added {len(df)} new rows to {csv_path}.")
        return

    existing_df = pd.read_csv(csv_path, dtype=str)
    existing_df = existing_df.fillna("")
    existing_keys = set(zip(existing_df.get("FDA_EntityName", []), existing_df.get("Ticker", [])))

    new_entries: list[dict] = []
    for entry in FDA_COMPANY_MAP:
        key = (entry.get("FDA_EntityName", ""), entry.get("Ticker", ""))
        if key in existing_keys:
            continue
        new_entries.append({column: entry.get(column, "") for column in columns})
        existing_keys.add(key)

    if not new_entries:
        print(f"Added 0 new rows to {csv_path}.")
        return

    new_df = pd.DataFrame(new_entries, columns=columns)
    new_df.to_csv(csv_path, mode="a", header=False, index=False)
    print(f"Added {len(new_entries)} new rows to {csv_path}.")


if __name__ == "__main__":
    ensure_csv()
