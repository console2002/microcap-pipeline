"""Centralized CSV filenames for pipeline outputs."""
import os

CSV_FILENAMES: dict[str, str] = {
    "profiles": "01_profiles.csv",
    "filings": "02_filings.csv",
    "prices": "03_prices.csv",
    "fda": "04_fda.csv",
    "hydrated_candidates": "05_hydrated_candidates.csv",
    "shortlist_candidates": "06_shortlist_candidates.csv",
    "deep_research_results": "07_deep_research_results.csv",
    "runway_extract_results": "08_runway_extract_results.csv",
    "eight_k_events": "09_8k_events.csv",
    "dr_populate_results": "10_dr_populate_results.csv",
    "validated_watchlist": "11_validated_watchlist.csv",
}


def csv_filename(key: str) -> str:
    """Return the numbered CSV filename for a logical dataset key."""
    try:
        return CSV_FILENAMES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown CSV key '{key}'") from exc


def csv_path(base_dir: str, key: str) -> str:
    """Return the absolute path for a CSV given its dataset key."""
    return os.path.join(base_dir, csv_filename(key))
