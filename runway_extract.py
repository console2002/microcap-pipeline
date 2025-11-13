"""Populate cash runway metrics for research results using SEC filings."""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import load_config
from app.utils import ensure_csv, log_line, utc_now_iso
from app.cancel import CancelledRun

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


def _canonical_form_name(form: str | None) -> str:
    if not form:
        return ""
    upper = (form or "").strip().upper()
    if upper.endswith("/A"):
        upper = upper[:-2]
    for prefix in (
        "10-QT",
        "10-Q",
        "10-KT",
        "10-K",
        "20-F",
        "40-F",
        "6-K",
    ):
        if upper.startswith(prefix):
            if prefix == "10-QT":
                return "10-Q"
            if prefix == "10-KT":
                return "10-K"
            return prefix
    return upper


def _form_priority(form: str) -> int:
    priorities = {
        "10-Q": 0,
        "10-K": 1,
        "20-F": 2,
        "40-F": 2,
        "6-K": 3,
    }
    return priorities.get(form, 9)


def _normalize_filing_url(url: object) -> str:
    """Return a canonical SEC filing URL with an explicit scheme."""
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("//"):
        return "https:" + text
    if "://" in text:
        return text
    if text.startswith("/"):
        return "https://www.sec.gov" + text
    parsed = urlparse("https://" + text)
    if parsed.netloc:
        return "https://" + text
    return text


def _group_filings_by_cik(filings: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in filings:
        cik_normalized = _normalize_cik_value(row.get("CIK"))
        if not cik_normalized:
            continue
        form_original = (row.get("Form") or "").strip()
        if not _is_relevant_form(form_original):
            continue

        canonical_form = _canonical_form_name(form_original)

        filed_at_raw = row.get("FiledAt")
        filed_at = _parse_filed_at(filed_at_raw)
        if filed_at is None:
            continue

        filing_url = _normalize_filing_url(
            row.get("FilingURL")
            or row.get("FilingUrl")
            or row.get("URL")
            or row.get("Url")
            or ""
        )

        info = {
            "filed_at": filed_at,
            "filing_url": filing_url,
            "form": canonical_form,
            "original_form": form_original,
            "priority": _form_priority(canonical_form),
        }

        cik_keys = {cik_normalized}
        trimmed = cik_normalized.lstrip("0")
        if trimmed:
            cik_keys.add(trimmed)

        for key in cik_keys:
            grouped.setdefault(key, []).append(info)

    for key, items in grouped.items():
        def _sort_key(entry: dict) -> tuple:
            filed_at = entry.get("filed_at")
            priority = entry.get("priority")
            try:
                priority_val = int(priority)
            except (TypeError, ValueError):
                priority_val = 9
            return (filed_at or datetime.min, -priority_val)

        items.sort(key=_sort_key, reverse=True)
    return grouped


def _round_half_up(value: float, digits: int = 2) -> float:
    try:
        quant = Decimal("1").scaleb(-digits)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return value
    return float(rounded)


def _format_display_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    rounded = _round_half_up(value, 2)
    return f"{rounded:.2f}"


def _format_raw_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return format(value, "f")


def run(data_dir: str | None = None, stop_flag: dict | None = None) -> None:
    research_path = _resolve_path("03_deep_research_results.csv", data_dir)
    filings_path = _resolve_path("filings.csv", data_dir)

    research_rows, fieldnames = _read_csv_rows(research_path)
    filings_rows, _ = _read_csv_rows(filings_path)

    filings_by_cik = _group_filings_by_cik(filings_rows)

    # NEW: optional hard-drop gate (default False, enable in config)
    cfg = load_config()
    drop_if_incomplete = bool(cfg.get("Runway", {}).get("DropIfNoRunway", False))

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

    cancel_pending = False
    cancel_announced = False
    cancel_after_write = False
    last_index = 0

    def check_cancel_request() -> None:
        nonlocal cancel_pending, cancel_announced
        if stop_flag and stop_flag.get("stop"):
            if not cancel_announced:
                progress(
                    "CANCEL",
                    "Cancellation requested – finishing current ticker before stopping",
                )
                cancel_announced = True
            cancel_pending = True

    for index, row in enumerate(research_rows, start=1):
        last_index = index
        check_cancel_request()
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
        source_form = ""
        source_date = ""
        source_url = ""
        runway_status = "No eligible filing"
        runway_assumption = ""
        ocf_period_months: Optional[int] = None
        ocf_raw: Optional[float] = None
        units_scale: Optional[int] = None
        source_days_ago = ""

        candidate_infos: List[dict] = []
        if cik:
            keys = [cik]
            trimmed_key = cik.lstrip("0")
            if trimmed_key and trimmed_key != cik:
                keys.append(trimmed_key)
            seen_filing_keys: set[tuple] = set()
            for key in keys:
                for info in filings_by_cik.get(key, []):
                    filing_key = (info.get("filing_url"), info.get("filed_at"))
                    if filing_key in seen_filing_keys:
                        continue
                    seen_filing_keys.add(filing_key)
                    candidate_infos.append(info)

        final_result: Optional[dict] = None
        final_info: Optional[dict] = None
        partial_result: Optional[dict] = None
        partial_info: Optional[dict] = None
        partial_rank: Optional[Tuple[int, datetime]] = None
        status_history: List[str] = []
        note_history: List[str] = []

        if not candidate_infos:
            status_history.append("No eligible filing")
            note_history.append("No eligible filing")
            progress("WARN", f"{ticker} no eligible filing in filings.csv (CIK {cik or 'n/a'})")
        else:
            for info in candidate_infos:
                check_cancel_request()
                form_name = info.get("form") or info.get("original_form") or "?"
                filing_url = info.get("filing_url") or ""
                filed_at_dt = info.get("filed_at")
                filed_at_text = filed_at_dt.isoformat() if filed_at_dt else "?"

                if not filing_url:
                    status_history.append("Fetch error")
                    note_history.append(f"{form_name} {filed_at_text} missing filing URL")
                    progress("WARN", f"{ticker} missing filing URL for {form_name} filed {filed_at_text}")
                    continue

                if not parser_10q.url_matches_form(filing_url, form_name):
                    status_history.append("Form/URL mismatch")
                    note_history.append(f"{form_name} URL mismatch: {filing_url}")
                    progress("WARN", f"{ticker} form/url mismatch for {form_name} {filing_url}")
                    continue

                progress("INFO", f"{ticker} fetching {form_name} filed {filed_at_text} url {filing_url}")

                form_hint = _canonical_form_name(form_name)
                hinted_url = filing_url
                if form_hint:
                    try:
                        parsed_url = urlparse(filing_url)
                        query_params = [(k, v) for k, v in parse_qsl(parsed_url.query, keep_blank_values=True) if k.lower() != "form"]
                        query_params.append(("form", form_hint))
                        hinted_url = urlunparse(
                            (
                                parsed_url.scheme,
                                parsed_url.netloc,
                                parsed_url.path,
                                parsed_url.params,
                                urlencode(query_params, doseq=True),
                                parsed_url.fragment,
                            )
                        )
                    except Exception:
                        hinted_url = filing_url
                try:
                    result = parser_10q.get_runway_from_filing(hinted_url)
                except Exception as exc:
                    status_history.append("Fetch error")
                    detail = f"{form_name} fetch failed: {exc.__class__.__name__}: {exc}"
                    note_history.append(detail)
                    progress(
                        "ERROR",
                        f"{ticker} fetch failed: {exc.__class__.__name__}: {exc} "
                        f"(url: {filing_url}; catalyst_link: {catalyst_link or 'n/a'})",
                    )
                    continue

                if not result:
                    status_history.append("Fetch error")
                    note_history.append(f"{form_name} parser returned no data")
                    continue

                if not result.get("form_type") and form_name:
                    result["form_type"] = form_name

                result_status = result.get("status") or "OK"
                note_text = result.get("note") or ""

                if result.get("complete"):
                    final_result = result
                    final_info = info
                    status_history.append(result_status)
                    if note_text:
                        note_history.append(note_text)
                    else:
                        note_history.append(result_status)
                    break

                candidate_priority = info.get("priority", 9)
                filed_at_rank = filed_at_dt or datetime.min
                candidate_rank: Tuple[int, datetime] = (candidate_priority, filed_at_rank)
                if partial_rank is None or candidate_priority < partial_rank[0] or (
                    candidate_priority == partial_rank[0] and filed_at_rank > partial_rank[1]
                ):
                    partial_result = result
                    partial_info = info
                    partial_rank = candidate_rank
                status_history.append(result_status)
                if note_text:
                    note_history.append(note_text)
                else:
                    note_history.append(result_status)
                progress("WARN", f"{ticker} {form_name} incomplete: {result_status}")

        result_to_use = final_result or partial_result
        info_to_use = final_info or partial_info

        # NEW: optional hard-drop — only when computation is incomplete (no runway).
        if drop_if_incomplete:
            if not result_to_use or not result_to_use.get("complete"):
                drop_reason = (result_to_use.get("status") if result_to_use else "no parsable filing")
                progress("DROP", f"{ticker} dropped (no runway): {drop_reason}")
                emit_progress(index, ticker)
                continue

        if result_to_use:
            cash = result_to_use.get("cash")
            cash_raw = result_to_use.get("cash_raw")
            quarterly_burn = result_to_use.get("quarterly_burn")
            quarterly_burn_raw = result_to_use.get("quarterly_burn_raw")
            runway_quarters = result_to_use.get("runway_quarters")
            runway_quarters_raw = result_to_use.get("runway_quarters_raw")
            runway_estimate = result_to_use.get("estimate", "") or ""
            source_form = result_to_use.get("form_type") or source_form
            runway_status = result_to_use.get("status") or "OK"
            runway_assumption = result_to_use.get("assumption") or ""
            ocf_period_months = result_to_use.get("period_months")
            ocf_raw = result_to_use.get("ocf_raw")
            units_scale = result_to_use.get("units_scale")
        else:
            runway_estimate = ""
            runway_status = status_history[-1] if status_history else "No eligible filing"
            runway_assumption = ""
            ocf_period_months = None
            ocf_raw = None
            units_scale = None

        if info_to_use:
            filed_at_dt = info_to_use.get("filed_at")
            if filed_at_dt:
                source_date = filed_at_dt.date().isoformat()
                days_delta = datetime.utcnow().date() - filed_at_dt.date()
                source_days_ago = str(days_delta.days)
            source_form = result_to_use.get("form_type") if result_to_use else (info_to_use.get("form") or source_form)
            source_url = info_to_use.get("filing_url") or source_url
        else:
            source_days_ago = ""

        note_parts: List[str] = []
        seen_notes: set[str] = set()
        for entry in note_history:
            if entry and entry not in seen_notes:
                note_parts.append(entry)
                seen_notes.add(entry)

        evidence_stub_parts = [part for part in [source_date, source_form, source_url] if part]
        evidence_stub = " ".join(evidence_stub_parts)
        if evidence_stub and evidence_stub not in seen_notes:
            note_parts.insert(0, evidence_stub)
            seen_notes.add(evidence_stub)

        note = " | ".join(part for part in note_parts if part)

        row["RunwayCash"] = _format_display_value(cash)
        row["RunwayCashRaw"] = _format_raw_value(cash_raw)
        row["RunwayQuarterlyBurn"] = _format_display_value(quarterly_burn)
        row["RunwayQuarterlyBurnRaw"] = _format_raw_value(quarterly_burn_raw)
        row["RunwayQuarters"] = _format_display_value(runway_quarters)
        row["RunwayQuartersRaw"] = _format_raw_value(runway_quarters_raw)
        row["RunwayEstimate"] = runway_estimate
        row["RunwayNotes"] = note
        row["RunwaySourceForm"] = source_form
        row["RunwaySourceDate"] = source_date
        row["RunwaySourceURL"] = source_url
        row["RunwayStatus"] = runway_status
        row["RunwayAssumption"] = runway_assumption
        row["RunwayOCFPeriodMonths"] = str(ocf_period_months or "")
        row["RunwayOCFRaw"] = _format_raw_value(ocf_raw)
        row["RunwayUnitsScale"] = str(units_scale or "")
        row["RunwaySourceDaysAgo"] = source_days_ago

        telemetry_cash_str = _format_raw_value(result_to_use.get("cash_raw") if result_to_use else None)
        telemetry_ocf_str = _format_raw_value(result_to_use.get("ocf_raw") if result_to_use else None)
        telemetry_months = (
            str(result_to_use.get("period_months")) if result_to_use and result_to_use.get("period_months") is not None else ""
        )
        telemetry_scale = (
            str(result_to_use.get("units_scale")) if result_to_use and result_to_use.get("units_scale") is not None else ""
        )
        telemetry_estimate = result_to_use.get("estimate") if result_to_use else ""
        telemetry_form = row.get("RunwaySourceForm", "")
        telemetry_date = row.get("RunwaySourceDate", "")
        merged_tag = ""
        if result_to_use:
            tags = result_to_use.get("source_tags") or []
            if "XBRL" in tags and "HTML" in tags:
                merged_tag = " merged=XBRL+HTML"
        progress(
            "INFO",
            "compute_runway: "
            f"{ticker} cash={telemetry_cash_str} ocf={telemetry_ocf_str} "
            f"months={telemetry_months} scale={telemetry_scale} "
            f"est={telemetry_estimate} form={telemetry_form} date={telemetry_date}{merged_tag}",
        )

        if runway_status == "OK" and runway_quarters is not None:
            progress("OK", f"{ticker} runway {runway_quarters:.2f} qtrs")
        else:
            progress("WARN", f"{ticker} runway status {runway_status}")

        output_rows.append(row)
        emit_progress(index, ticker)

        if cancel_pending:
            cancel_after_write = True
            break

    output_dir = data_dir or os.path.dirname(research_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "04_runway_extract_results.csv") if output_dir else "04_runway_extract_results.csv"

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
        "RunwayStatus",
        "RunwayAssumption",
        "RunwayOCFPeriodMonths",
        "RunwayOCFRaw",
        "RunwayUnitsScale",
        "RunwaySourceForm",
        "RunwaySourceDate",
        "RunwaySourceURL",
        "RunwaySourceDaysAgo",
    ]:
        if col not in fieldnames_out:
            fieldnames_out.append(col)

    _write_csv_rows(output_path, fieldnames_out, output_rows)

    if cancel_after_write:
        rows_written = len(output_rows)
        progress(
            "CANCEL",
            f"Runway extraction cancelled -> {output_path} ({rows_written} rows written)",
        )
        if last_index and total_rows:
            emit_progress(last_index)
        raise CancelledRun("cancel during parse_q10")

    progress("OK", f"Runway extraction complete -> {output_path}")
    emit_progress(total_rows or 0)


if __name__ == "__main__":
    run()
