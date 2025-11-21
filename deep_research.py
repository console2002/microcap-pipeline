"""Deep research pipeline stub for catalysts, dilution, and runway analysis."""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from typing import Iterable, List, Sequence

from app.config import load_config
from app.csv_names import csv_filename, csv_path
from app.utils import ensure_csv, log_line, utc_now_iso


_PROGRESS_LOG_PATH: str | None = None
_CSV_FIELD_SIZE_LIMIT_SET = False
_ECHO_PROGRESS = False


def set_progress_echo(enabled: bool) -> None:
    """Enable or disable echoing progress rows to stdout."""
    global _ECHO_PROGRESS
    _ECHO_PROGRESS = enabled


def _ensure_csv_field_size_limit() -> None:
    """Raise the CSV field size limit to accommodate large research notes."""
    global _CSV_FIELD_SIZE_LIMIT_SET
    if _CSV_FIELD_SIZE_LIMIT_SET:
        return

    max_limit = sys.maxsize
    while max_limit > 0:
        try:
            csv.field_size_limit(max_limit)
            _CSV_FIELD_SIZE_LIMIT_SET = True
            break
        except (OverflowError, ValueError):
            max_limit //= 2
    if not _CSV_FIELD_SIZE_LIMIT_SET:
        # Fallback to a reasonable manual limit if we exhausted the loop.
        csv.field_size_limit(10_000_000)
        _CSV_FIELD_SIZE_LIMIT_SET = True


def _progress_log_path() -> str:
    global _PROGRESS_LOG_PATH
    if _PROGRESS_LOG_PATH is None:
        cfg = load_config()
        logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, "progress.csv")
        ensure_csv(path, ["timestamp", "status", "message"])
        _PROGRESS_LOG_PATH = path
    return _PROGRESS_LOG_PATH


def progress(status: str, message: str) -> None:
    """Append a progress row for the GUI tail (optionally echo to stdout)."""
    path = _progress_log_path()
    timestamp = utc_now_iso()
    log_line(path, [timestamp, status, message])
    if _ECHO_PROGRESS:
        print(f"{timestamp} | {status} | {message}")


DILUTION_FORMS = {
    "S-3",
    "S-3ASR",
    "S-8",
    "424B",
    "424B1",
    "424B2",
    "424B3",
    "424B4",
    "424B5",
    "424B7",
    "424B8",
}

DILUTION_KEYWORDS = [
    "offering",
    "atm",
    "at-the-market",
    "sale of common stock",
    "equity line",
]


def normalize_text(value: str | float | None) -> str:
    """Normalize arbitrary CSV cell values into clean strings."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(value).strip()
    text = str(value).strip()
    if not text:
        return ""
    # Many upstream CSV exports use Windows line endings (``\r\n``). If we only
    # split on ``\n`` later, the stray ``\r`` characters survive and end up
    # embedded inside URLs (e.g. ``https:\r//``).  When pandas writes those
    # values back out, the carriage return causes downstream CSV readers to
    # misalign the columns.  Normalize all carriage returns here so subsequent
    # parsing logic can safely split on ``\n``.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


_EVIDENCE_SPLIT_RE = re.compile(r"[;\r\n]+")


def _split_evidence_entries(text: str) -> list[str]:
    parts = [segment.strip() for segment in _EVIDENCE_SPLIT_RE.split(text)]
    cleaned = [seg for seg in parts if seg and seg.lower() != "nan"]
    return cleaned


def split_semicolon_list(raw: str | None) -> list[str]:
    """Split evidence fields that may be delimited by semicolons or newlines."""
    text = normalize_text(raw)
    if not text:
        return []
    return _split_evidence_entries(text)


def dedupe_preserve_order(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_dilution_info(filings_summary: str | None) -> dict:
    """Identify dilution-related filings and supporting links."""
    forms: list[str] = []
    links: list[str] = []

    summary_text = normalize_text(filings_summary)
    if not summary_text:
        return {
            "forms": forms,
            "links": links,
            "has_flag": False,
            "notes": "",
        }

    chunks = _split_evidence_entries(summary_text)
    for chunk in chunks:
        parts = [part.strip() for part in chunk.split("|")]
        form = parts[1] if len(parts) > 1 else ""
        form_clean = normalize_text(form)
        form_upper = form_clean.upper()
        chunk_lower = chunk.lower()

        url = ""
        for part in reversed(parts):
            candidate = part.strip()
            if candidate.lower().startswith("http"):
                url = candidate
                break
        if not url:
            for token in reversed(chunk.replace("|", " ").split()):
                token_strip = token.strip()
                if token_strip.lower().startswith("http"):
                    url = token_strip
                    break

        is_dilution_form = any(trigger in form_upper for trigger in DILUTION_FORMS)
        is_keyword_hit = False
        if form_upper == "8-K":
            is_keyword_hit = any(keyword in chunk_lower for keyword in DILUTION_KEYWORDS)

        if is_dilution_form or is_keyword_hit:
            if form_clean:
                forms.append(form_clean)
            if url:
                links.append(url)

    forms = dedupe_preserve_order(forms)
    links = dedupe_preserve_order(links)
    has_flag = bool(forms)
    notes = f"Recent dilution-related filings: {', '.join(forms)}" if has_flag else ""

    return {
        "forms": forms,
        "links": links,
        "has_flag": has_flag,
        "notes": notes,
    }


def load_research_rows(path: str | None = None) -> list[dict]:
    """Load shortlist rows from CSV into a list of dictionaries."""
    _ensure_csv_field_size_limit()

    if path is None:
        cfg = load_config()
        data_dir = cfg.get("Paths", {}).get("data", ".")
        path = csv_path(data_dir, "shortlist_candidates")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _determine_catalyst_type(latest_form: str, fda_event: str) -> str:
    if fda_event:
        return fda_event

    form_upper = latest_form.upper()
    if "8-K" in form_upper:
        return "Corporate event / guidance / financing"
    if "10-Q" in form_upper or "10-K" in form_upper:
        return "Earnings / financial update"
    return "TBD"


def build_bundle_for_row(row: dict) -> dict:
    ticker = normalize_text(row.get("Ticker"))
    company = normalize_text(row.get("Company"))
    cik = normalize_text(row.get("CIK"))
    sector = normalize_text(row.get("Sector"))

    fda_event = normalize_text(row.get("FDA_EventType"))
    fda_date = normalize_text(row.get("FDA_Date"))
    fda_url = normalize_text(row.get("FDA_URL"))
    fda_urls_all = split_semicolon_list(row.get("FDA_URLsAll"))
    fda_summary = normalize_text(row.get("FDA_Summary"))

    filing_form = normalize_text(row.get("LatestForm"))
    filed_at = normalize_text(row.get("FiledAt"))
    filing_url = normalize_text(row.get("FilingURL"))
    filing_urls_all = split_semicolon_list(row.get("FilingURLsAll"))
    filings_summary = normalize_text(row.get("FilingsSummary"))

    primary_link = fda_url if fda_event else filing_url
    catalyst_type = _determine_catalyst_type(filing_form, fda_event)
    catalyst_date = fda_date or filed_at

    evidence_links: list[str] = []
    if primary_link:
        evidence_links.append(primary_link)
    evidence_links.extend(fda_urls_all)
    evidence_links.extend(filing_urls_all)
    evidence_links = dedupe_preserve_order(evidence_links)

    dilution_info = extract_dilution_info(row.get("FilingsSummary"))

    return {
        "ticker": ticker,
        "company": company,
        "cik": cik,
        "sector": sector,
        "catalyst": {
            "catalyst_type": catalyst_type,
            "catalyst_date": catalyst_date,
            "primary_link": primary_link,
            "evidence_primary_links": evidence_links,
            "evidence_secondary_links": [],
        },
        "dilution": {
            "has_dilution_flag": dilution_info["has_flag"],
            "dilution_forms": dilution_info["forms"],
            "dilution_links": dilution_info["links"],
            "notes": dilution_info["notes"],
        },
        "runway": {
            "cash": None,
            "quarterly_burn": None,
            "runway_quarters": None,
            "notes": "TODO: parse 10-Q/10-K to compute runway",
        },
        "latest_form": filing_form,
        "filed_at": filed_at,
        "filing_url": filing_url,
        "filings_summary": filings_summary,
        "filing_urls_all": filing_urls_all,
        "fda_event_type": fda_event,
        "fda_date": fda_date,
        "fda_url": fda_url,
        "fda_urls_all": fda_urls_all,
        "fda_summary": fda_summary,
    }


def write_research_results(bundles: Sequence[dict], path: str = csv_filename("deep_research_results")) -> None:
    """Flatten bundles into CSV output for downstream steps."""
    fieldnames = [
        "Ticker",
        "Company",
        "CIK",
        "Sector",
        "CatalystType",
        "CatalystDate",
        "CatalystPrimaryLink",
        "CatalystEvidencePrimaryLinks",
        "CatalystEvidenceSecondaryLinks",
        "HasDilutionFlag",
        "DilutionForms",
        "DilutionLinks",
        "DilutionNotes",
        "RunwayCash",
        "RunwayQuarterlyBurn",
        "RunwayQuarters",
        "RunwayNotes",
        "LatestForm",
        "FiledAt",
        "FilingURL",
        "FilingsSummary",
        "FilingURLsAll",
        "FDA_EventType",
        "FDA_Date",
        "FDA_URL",
        "FDA_URLsAll",
        "FDA_Summary",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bundle in bundles:
            catalyst = bundle["catalyst"]
            dilution = bundle["dilution"]
            runway = bundle["runway"]

            writer.writerow({
                "Ticker": bundle["ticker"],
                "Company": bundle["company"],
                "CIK": bundle["cik"],
                "Sector": bundle["sector"],
                "CatalystType": catalyst["catalyst_type"],
                "CatalystDate": catalyst["catalyst_date"],
                "CatalystPrimaryLink": catalyst["primary_link"],
                "CatalystEvidencePrimaryLinks": "; ".join(catalyst["evidence_primary_links"]),
                "CatalystEvidenceSecondaryLinks": "; ".join(catalyst["evidence_secondary_links"]),
                "HasDilutionFlag": "TRUE" if dilution["has_dilution_flag"] else "FALSE",
                "DilutionForms": "; ".join(dilution["dilution_forms"]),
                "DilutionLinks": "; ".join(dilution["dilution_links"]),
                "DilutionNotes": dilution["notes"],
                "RunwayCash": runway["cash"] if runway["cash"] is not None else "",
                "RunwayQuarterlyBurn": runway["quarterly_burn"] if runway["quarterly_burn"] is not None else "",
                "RunwayQuarters": runway["runway_quarters"] if runway["runway_quarters"] is not None else "",
                "RunwayNotes": runway["notes"],
                "LatestForm": bundle.get("latest_form", ""),
                "FiledAt": bundle.get("filed_at", ""),
                "FilingURL": bundle.get("filing_url", ""),
                "FilingsSummary": bundle.get("filings_summary", ""),
                "FilingURLsAll": "; ".join(bundle.get("filing_urls_all", [])),
                "FDA_EventType": bundle.get("fda_event_type", ""),
                "FDA_Date": bundle.get("fda_date", ""),
                "FDA_URL": bundle.get("fda_url", ""),
                "FDA_URLsAll": "; ".join(bundle.get("fda_urls_all", [])),
                "FDA_Summary": bundle.get("fda_summary", ""),
            })


def run(data_dir: str | None = None, *, echo: bool = False) -> None:
    """Execute the deep research pipeline using the provided data directory."""

    prev_echo = _ECHO_PROGRESS
    set_progress_echo(echo)
    try:
        if data_dir is None:
            cfg = load_config()
            data_dir = cfg.get("Paths", {}).get("data", ".")

        shortlist_path = csv_path(data_dir, "shortlist_candidates")
        results_path = csv_path(data_dir, "deep_research_results")

        shortlist_rows = load_research_rows(shortlist_path)
        bundles: List[dict] = []

        for row in shortlist_rows:
            ticker = normalize_text(row.get("Ticker")) or "?"
            cik = normalize_text(row.get("CIK")) or "?"
            progress("RUN", f"DeepResearch {ticker} ({cik})")
            bundle = build_bundle_for_row(row)
            bundles.append(bundle)

            if bundle["dilution"]["has_dilution_flag"]:
                forms = bundle["dilution"]["dilution_forms"]
                progress("OK", f"{ticker}: dilution risk {forms}")
            else:
                progress("OK", f"{ticker}: no dilution flag")

        write_research_results(bundles, results_path)
        progress("OK", f"DeepResearch complete -> {csv_filename('deep_research_results')}")
    finally:
        set_progress_echo(prev_echo)


if __name__ == "__main__":
    run(echo=True)
