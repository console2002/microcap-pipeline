"""Populate cash runway metrics for research results using SEC filings."""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from app.config import load_config
from app.utils import ensure_csv, log_line, utc_now_iso

import parser_10q


_PROGRESS_LOG_PATH: str | None = None


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
    path = _progress_log_path()
    timestamp = utc_now_iso()
    log_line(path, [timestamp, status, message])


_RELEVANT_FORM_PREFIXES = ("10-Q", "10-K")
_DATE_FORMATS = [
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
]


_ISO_TZ_RE = re.compile(r"([+-]\d{2})(\d{2})$")

def _resolve_path(filename: str, base_dir: str | None = None) -> str:
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    candidates.extend([filename, os.path.join("data", filename)])
    for path in candidates:
        if os.path.exists(path):
            return path
    if base_dir:
        return os.path.join(base_dir, filename)
    return filename


def _read_csv_rows(path: str) -> tuple[List[dict], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _write_csv_rows(path: str, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_filed_at(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None

    normalized = text
    if normalized.endswith("Z") or normalized.endswith("z"):
        normalized = normalized[:-1] + "+00:00"

    normalized = _ISO_TZ_RE.sub(lambda m: f"{m.group(1)}:{m.group(2)}", normalized)
    normalized = re.sub(r"\s+[A-Za-z]{2,5}$", "", normalized)

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_relevant_form(form: str | None) -> bool:
    if not form:
        return False
    upper = form.upper()
    return upper.startswith(_RELEVANT_FORM_PREFIXES)


def _normalize_cik_value(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = re.sub(r"\.0+$", "", text)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""

    return digits.zfill(10)


def _build_latest_filing_map(filings: Iterable[dict]) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for row in filings:
        cik_normalized = _normalize_cik_value(row.get("CIK"))
        if not cik_normalized:
            continue
        cik_keys = {cik_normalized}
        trimmed = cik_normalized.lstrip("0")
        if trimmed:
            cik_keys.add(trimmed)
        form = (row.get("Form") or "").strip()
        if not _is_relevant_form(form):
            continue

        filed_at_raw = row.get("FiledAt")
        filed_at = _parse_filed_at(filed_at_raw)
        if filed_at is None:
            continue

        filing_url = (
            row.get("FilingURL")
            or row.get("FilingUrl")
            or row.get("URL")
            or row.get("Url")
            or ""
        )
        filing_url = str(filing_url).strip()

        info = {
            "filed_at": filed_at,
            "filing_url": filing_url,
            "form": form,
        }

        for key in cik_keys:
            current = latest.get(key)
            if current is None or filed_at > current["filed_at"]:
                latest[key] = info
    return latest


def _format_numeric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return str(value)


def run(data_dir: str | None = None) -> None:
    research_path = _resolve_path("research_results.csv", data_dir)
    filings_path = _resolve_path("filings.csv", data_dir)

    research_rows, fieldnames = _read_csv_rows(research_path)
    filings_rows, _ = _read_csv_rows(filings_path)

    latest_filing_map = _build_latest_filing_map(filings_rows)

    output_rows: List[dict] = []

    for row in research_rows:
        cik = _normalize_cik_value(row.get("CIK"))
        ticker = (row.get("Ticker") or "").strip() or cik or "?"

        cash: Optional[float] = None
        quarterly_burn: Optional[float] = None
        runway_quarters: Optional[float] = None
        note: Optional[str] = None

        filing_info = latest_filing_map.get(cik)
        if filing_info and filing_info.get("filing_url"):
            filing_url = filing_info["filing_url"]
            try:
                result = parser_10q.get_runway_from_filing(filing_url)
            except Exception:
                progress("ERROR", f"{ticker} fetch failed")
                result = None
                note = "failed to fetch 10-Q/10-K"
            else:
                cash = result.get("cash") if result else None
                quarterly_burn = result.get("quarterly_burn") if result else None
                runway_quarters = result.get("runway_quarters") if result else None
                note = (result.get("note") if result else None) or ""
        else:
            note = None

        if not note:
            note = "no 10-Q/10-K found"

        row["RunwayCash"] = _format_numeric(cash)
        row["RunwayQuarterlyBurn"] = _format_numeric(quarterly_burn)
        row["RunwayQuarters"] = _format_numeric(runway_quarters)
        row["RunwayNotes"] = note

        if runway_quarters is not None:
            progress("OK", f"{ticker} runway {runway_quarters:.2f} qtrs")
        else:
            progress("WARN", f"{ticker} runway missing")

        output_rows.append(row)

    output_dir = data_dir or os.path.dirname(research_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "research_results_runway.csv") if output_dir else "research_results_runway.csv"

    fieldnames_out = list(fieldnames)
    for col in [
        "RunwayCash",
        "RunwayQuarterlyBurn",
        "RunwayQuarters",
        "RunwayNotes",
    ]:
        if col not in fieldnames_out:
            fieldnames_out.append(col)

    _write_csv_rows(output_path, fieldnames_out, output_rows)
    progress("OK", f"Runway extraction complete -> {output_path}")


if __name__ == "__main__":
    run()
