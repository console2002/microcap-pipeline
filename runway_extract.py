"""Populate cash runway metrics for research results using SEC filings."""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from app.config import load_config
from app.utils import ensure_csv, log_line, utc_now_iso

import parser_10q


_PROGRESS_CALLBACK: Callable[[str, str], None] | None = None


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


def set_progress_callback(callback: Callable[[str, str], None] | None) -> None:
    """Register an optional in-memory progress callback."""
    global _PROGRESS_CALLBACK
    _PROGRESS_CALLBACK = callback


def progress(status: str, message: str) -> None:
    path = _progress_log_path()
    timestamp = utc_now_iso()
    log_line(path, [timestamp, status, message])
    if _PROGRESS_CALLBACK is not None:
        try:
            _PROGRESS_CALLBACK(status, message)
        except Exception:
            # never allow GUI callbacks to break disk logging
            pass


_RELEVANT_FORM_PREFIXES = (
    "10-Q",
    "10-QT",
    "10-K",
    "10-KT",
    "20-F",
    "40-F",
    "6-K",
)
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

    total_rows = len(research_rows)
    last_progress_pct = -1

    def emit_progress(current: int, ticker: str | None = None) -> None:
        nonlocal last_progress_pct
        if total_rows <= 0:
            return
        pct = int(round((current / total_rows) * 100))
        pct = max(0, min(100, pct))
        if pct == last_progress_pct:
            return
        last_progress_pct = pct
        detail = f" processed {ticker}" if ticker else ""
        progress("PROGRESS", f"({pct}%) parse_q10{detail}")

    emit_progress(0)

    for index, row in enumerate(research_rows, start=1):
        cik = _normalize_cik_value(row.get("CIK"))
        ticker = (row.get("Ticker") or "").strip() or cik or "?"
        catalyst_link = (row.get("CatalystPrimaryLink") or "").strip()

        cash: Optional[float] = None
        cash_raw: Optional[float] = None
        quarterly_burn: Optional[float] = None
        quarterly_burn_raw: Optional[float] = None
        runway_quarters: Optional[float] = None
        runway_quarters_raw: Optional[float] = None
        runway_estimate: str = ""
        note: Optional[str] = None
        source_form = ""
        source_date = ""
        source_url = ""

        filing_info = latest_filing_map.get(cik)
        if filing_info and filing_info.get("filing_url"):
            filing_url = filing_info["filing_url"]
            filed_at_dt = filing_info.get("filed_at")
            filed_at_text = filed_at_dt.isoformat() if filed_at_dt else "?"
            form_name = filing_info.get("form") or "?"
            source_form = form_name
            if filed_at_dt:
                source_date = filed_at_dt.date().isoformat()
            source_url = filing_url
            progress(
                "INFO",
                f"{ticker} fetching {form_name} filed {filed_at_text} url {filing_url}",
            )
            try:
                result = parser_10q.get_runway_from_filing(filing_url)
            except Exception as exc:
                error_msg = (
                    f"{ticker} fetch failed: {exc.__class__.__name__}: {exc} "
                    f"(url: {filing_url}; catalyst_link: {catalyst_link or 'n/a'})"
                )
                progress("ERROR", error_msg)
                result = None
                note = (
                    "failed to fetch 10-Q/10-K "
                    f"({exc.__class__.__name__}: {exc}) – {filing_url}"
                )
            else:
                if result:
                    cash = result.get("cash")
                    cash_raw = result.get("cash_raw")
                    quarterly_burn = result.get("quarterly_burn")
                    quarterly_burn_raw = result.get("quarterly_burn_raw")
                    runway_quarters = result.get("runway_quarters")
                    runway_quarters_raw = result.get("runway_quarters_raw")
                    runway_estimate = result.get("estimate", "") or ""
                    note = (result.get("note") or "")
                    detected_form = result.get("form_type")
                    if detected_form:
                        source_form = detected_form
                else:
                    note = ""
        elif filing_info:
            note = None
            filed_at_dt = filing_info.get("filed_at")
            filed_at_text = filed_at_dt.isoformat() if filed_at_dt else "?"
            form_name = filing_info.get("form") or "?"
            progress(
                "WARN",
                f"{ticker} missing filing URL for {form_name} filed {filed_at_text}",
            )
        else:
            note = None
            progress(
                "WARN",
                f"{ticker} no recent 10-Q/10-K filing in filings.csv (CIK {cik or 'n/a'})",
            )

        if not note:
            note = "no 10-Q/10-K found"

        evidence_stub_parts = [part for part in [source_date, source_form, source_url] if part]
        evidence_stub = " ".join(evidence_stub_parts)
        if evidence_stub:
            if note:
                note = f"{evidence_stub} | {note}"
            else:
                note = evidence_stub

        row["RunwayCash"] = _format_numeric(cash)
        row["RunwayCashRaw"] = _format_numeric(cash_raw)
        row["RunwayQuarterlyBurn"] = _format_numeric(quarterly_burn)
        row["RunwayQuarterlyBurnRaw"] = _format_numeric(quarterly_burn_raw)
        row["RunwayQuarters"] = _format_numeric(runway_quarters)
        row["RunwayQuartersRaw"] = _format_numeric(runway_quarters_raw)
        row["RunwayEstimate"] = runway_estimate
        row["RunwayNotes"] = note
        row["RunwaySourceForm"] = source_form
        row["RunwaySourceDate"] = source_date
        row["RunwaySourceURL"] = source_url

        if runway_quarters is not None:
            progress("OK", f"{ticker} runway {runway_quarters:.2f} qtrs")
        else:
            progress("WARN", f"{ticker} runway missing")

        output_rows.append(row)

        emit_progress(index, ticker)

    output_dir = data_dir or os.path.dirname(research_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "research_results_runway.csv") if output_dir else "research_results_runway.csv"

    fieldnames_out = list(fieldnames)
    for col in [
        "RunwayCash",
        "RunwayCashRaw",
        "RunwayQuarterlyBurn",
        "RunwayQuarterlyBurnRaw",
        "RunwayQuarters",
        "RunwayQuartersRaw",
        "RunwayEstimate",
        "RunwayNotes",
        "RunwaySourceForm",
        "RunwaySourceDate",
        "RunwaySourceURL",
    ]:
        if col not in fieldnames_out:
            fieldnames_out.append(col)

    _write_csv_rows(output_path, fieldnames_out, output_rows)
    progress("OK", f"Runway extraction complete -> {output_path}")
    emit_progress(total_rows or 0)


if __name__ == "__main__":
    run()
