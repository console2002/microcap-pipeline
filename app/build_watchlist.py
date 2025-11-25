"""Build a validated watchlist from deep research results and market data."""
from __future__ import annotations

import csv
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlsplit
from pathlib import Path

import pandas as pd

from app.config import load_config
from app.csv_names import csv_filename, csv_path
from app.utils import ensure_csv, log_line, utc_now_iso
from parse.htmlutil import preview_text
from parse.router import _fetch_url

try:  # pragma: no cover - optional for legacy parsing
    from parse import k8 as parse_k8
except Exception:  # pragma: no cover - keep pipeline alive without legacy module
    parse_k8 = None


ProgressFn = Optional[Callable[[str], None]]


_OUTPUT_COLUMNS = [
    "Ticker",
    "Company",
    "Sector",
    "CIK",
    "Price",
    "MarketCap",
    "ADV20",
    "LatestForm",
    "LatestFiledAt",
    "LatestFiledAgeDays",
    "CatalystType",
    "Catalyst",
    "CatalystStatus",
    "CatalystPrimaryForm",
    "CatalystPrimaryDate",
    "CatalystPrimaryDaysAgo",
    "CatalystPrimaryURL",
    "RunwayQuartersRaw",
    "RunwayQuarters",
    "RunwayCashRaw",
    "RunwayCash",
    "RunwayQuarterlyBurnRaw",
    "RunwayQuarterlyBurn",
    "RunwayEstimate",
    "RunwayNotes",
    "RunwayDays",
    "RunwayBucket",
    "RunwaySourceForm",
    "RunwaySourceDate",
    "RunwaySourceDaysAgo",
    "RunwaySourceURL",
    "RunwayStatus",
    "DilutionFlag",
    "Dilution",
    "DilutionStatus",
    "DilutionFormsHit",
    "DilutionKeywordsHit",
    "DilutionPrimaryForm",
    "DilutionPrimaryDate",
    "DilutionPrimaryDaysAgo",
    "DilutionPrimaryURL",
    "DilutionEvidenceAll",
    "Governance",
    "GovernanceStatus",
    "GovernancePrimaryForm",
    "GovernancePrimaryDate",
    "GovernancePrimaryDaysAgo",
    "GovernancePrimaryURL",
    "GovernanceEvidenceAll",
    "GovernanceNotes",
    "HasGoingConcern",
    "HasMaterialWeakness",
    "Auditor",
    "Insider",
    "InsiderStatus",
    "InsiderForms345Links",
    "InsiderBuyCount",
    "HasInsiderCluster",
    "Ownership",
    "OwnershipStatus",
    "OwnershipLinks",
    "Materiality",
    "SubscoresEvidenced",
    "SubscoresEvidencedCount",
    "Status",
    "ChecklistPassed",
    "ChecklistCatalyst",
    "ChecklistDilution",
    "ChecklistRunway",
    "ChecklistGovernance",
    "ChecklistInsider",
    "ChecklistOwnership",
    "RiskFlag",
    "Tier1Type",
    "Tier1Trigger",
]

_EIGHT_K_EVENTS_COLUMNS = [
    "CIK",
    "Ticker",
    "FilingDate",
    "FilingURL",
    "ItemsPresent",
    "IsCatalyst",
    "CatalystType",
    "Tier1Type",
    "Tier1Trigger",
    "IsDilution",
    "DilutionTags",
    "IgnoreReason",
]


def _join_items(items: Iterable[str]) -> str:
    cleaned = [item.strip() for item in items if item and str(item).strip()]
    return "; ".join(cleaned)

_EIGHT_K_DEBUG_HEADER = [
    "ts",
    "url",
    "reason",
    "items_detected",
    "bytes_html",
    "bytes_exhibits",
    "sample_text",
]

_NEGATIVE_DILUTION_TERMS = (
    "high risk",
    "toxic",
    "at-the-market",
)

_SUBSCORE_SPLIT_RE = re.compile(r"[;\n]+")
_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_URL_RE = re.compile(r"https?://\S+")

_MANAGER_FORM_PREFIXES = (
    "13F",
    "13F-HR",
    "SC 13D",
    "SC 13G",
)

_DILUTION_FORM_PREFIXES = (
    "S-1",
    "S-3",
    "F-1",
    "F-3",
    "424B",
    "S-8",
)

_DILUTION_OFFERING_PREFIXES = (
    "S-1",
    "S-3",
    "F-1",
    "F-3",
    "424B",
)

_DILUTION_KEYWORDS = [
    "equity distribution agreement",
    "atm",
    "at-the-market",
    "sales agreement",
    "registered direct",
    "underwritten",
    "follow-on",
    "prospectus supplement",
    "rule 415",
    "shelf",
    "pipe",
    "subscription agreement",
    "convertible note",
    "convertible debenture",
    "convertible",
    "warrant",
    "pre-funded",
]

_FORM_URL_PATTERNS = {
    "10q": re.compile(r"/[^/]*10-?q"),
    "10k": re.compile(r"/[^/]*10-?k"),
    "8k": re.compile(r"/[^/]*8-?k"),
    "6k": re.compile(r"/[^/]*6-?k"),
    "20f": re.compile(r"/[^/]*20-?f"),
    "40f": re.compile(r"/[^/]*40-?f"),
    "def14a": re.compile(r"/[^/]*def14a"),
    "424b": re.compile(r"/[^/]*424b"),
    "s1": re.compile(r"/[^/]*s-?1"),
    "s3": re.compile(r"/[^/]*s-?3"),
    "f1": re.compile(r"/[^/]*f-?1"),
    "f3": re.compile(r"/[^/]*f-?3"),
    "8a12b": re.compile(r"/[^/]*8a12b"),
}

_GOVERNANCE_VALID_FORMS = {"DEF 14A", "10-K", "20-F", "40-F"}
_INSIDER_LINK_RE = re.compile(r"(form[345]|xslf345)", re.IGNORECASE)
_OWNERSHIP_LINK_RE = re.compile(r"(13f|sc13d|sc13g)", re.IGNORECASE)

_CATALYST_DILUTION_FORM_GUARDS = {"s3", "424b5", "424b3"}

_CATALYST_ITEM_LABELS = {
    "1.01": "Material Agreement",
    "2.02": "Earnings Results",
    "3.02": "Unregistered Sales",
    "7.01": "Reg FD",
    "8.01": "Other Events",
}

_CATALYST_KEYWORD_LABELS = [
    (re.compile(r"guidance"), "Guidance"),
    (re.compile(r"contract|award|partnership"), "Commercial Update"),
    (re.compile(r"regulatory|approval|clearance|adcom|pdufa"), "Regulatory"),
    (re.compile(r"bankruptcy|restructuring"), "Restructuring"),
    (re.compile(r"buyback"), "Buyback"),
    (re.compile(r"uplist|downlist"), "Listing Change"),
]


def _progress_log_path() -> str:
    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "progress.csv")
    ensure_csv(path, ["timestamp", "status", "message"])
    return path


def _emit(status: str, message: str, progress_fn: ProgressFn) -> None:
    path = _progress_log_path()
    timestamp = utc_now_iso()
    log_line(path, [timestamp, status, message])
    if progress_fn is not None:
        try:
            progress_fn(f"{status} {message}")
        except Exception:
            pass


def _resolve_path(filename: str, base_dir: str | None) -> str:
    candidates: list[str] = []
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    else:
        candidates.extend([filename, os.path.join("data", filename)])
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, filename) if base_dir else filename


def _normalize_ticker(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return text


def _normalize_cik(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(10)


def _count_subscores(raw: object) -> int:
    if raw is None:
        return 0
    text = str(raw).strip()
    if not text:
        return 0
    parts = [segment.strip() for segment in _SUBSCORE_SPLIT_RE.split(text)]
    return sum(1 for part in parts if part)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "no", "n"}:
            return False
        if text in {"1", "true", "yes", "y", "t"}:
            return True
        return False
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return False
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    return url.strip().rstrip(";.,")


def _extract_accession_token(url: str) -> str:
    if not url:
        return ""

    match = re.search(r"(?i)/edgar/data/\d+/([^/?#]+)/", url)
    if not match:
        return ""

    return match.group(1).lower()


@dataclass
class EightKEvent:
    cik: str
    ticker: str
    filing_date: Optional[str]
    filing_url: str
    items_present: str
    is_catalyst: bool
    catalyst_type: str
    catalyst_label: str
    tier1_type: str
    tier1_trigger: str
    is_dilution: bool
    dilution_tags: list[str]
    ignore_reason: str

    def dilution_tags_joined(self) -> str:
        return ";".join(self.dilution_tags)


class EightKLookup:
    def __init__(self, events: Iterable[EightKEvent]):
        self.by_url: dict[str, EightKEvent] = {}
        self.by_cik: dict[str, list[EightKEvent]] = defaultdict(list)
        self.by_ticker: dict[str, list[EightKEvent]] = defaultdict(list)
        self.by_accession: dict[str, EightKEvent] = {}

        for event in events:
            normalized_url = _normalize_url(event.filing_url)
            if normalized_url and normalized_url not in self.by_url:
                self.by_url[normalized_url] = event
            if event.cik:
                self.by_cik[event.cik].append(event)
            if event.ticker:
                self.by_ticker[event.ticker].append(event)
            accession_token = _extract_accession_token(event.filing_url)
            if accession_token:
                existing = self.by_accession.get(accession_token)
                if not existing or (event.filing_date or "") > (existing.filing_date or ""):
                    self.by_accession[accession_token] = event

        def _sort_events(mapping: dict[str, list[EightKEvent]]) -> None:
            for key, seq in mapping.items():
                seq.sort(
                    key=lambda evt: (
                        evt.filing_date or "",
                        evt.filing_url,
                    ),
                    reverse=True,
                )

        _sort_events(self.by_cik)
        _sort_events(self.by_ticker)

    def match(self, ticker: str, cik: str, candidate_urls: Iterable[str]) -> Optional[EightKEvent]:
        urls = [url for url in (candidate_urls or [])]

        for raw_url in urls:
            normalized = _normalize_url(raw_url)
            if not normalized:
                continue
            event = self.by_url.get(normalized)
            if event:
                return event

        normalized_cik = _normalize_cik(cik)
        if normalized_cik and self.by_cik.get(normalized_cik):
            return self.by_cik[normalized_cik][0]

        normalized_ticker = _normalize_ticker(ticker)
        if normalized_ticker and self.by_ticker.get(normalized_ticker):
            return self.by_ticker[normalized_ticker][0]

        for raw_url in urls:
            accession_token = _extract_accession_token(raw_url)
            if accession_token and accession_token in self.by_accession:
                return self.by_accession[accession_token]

        return None


def _eight_k_debug_path() -> str:
    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "debug8k.csv")
    ensure_csv(path, _EIGHT_K_DEBUG_HEADER)
    return path


def _write_eight_k_debug(entries: list[list[object]]) -> None:
    if not entries:
        return
    path = _eight_k_debug_path()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(entries)


def _read_eight_k_html(url: str) -> tuple[Optional[str], Optional[str]]:
    if not url:
        return None, "empty_url"
    parsed = urlsplit(url)
    if parsed.scheme == "file":
        path_text = unquote(parsed.path or "")
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        candidate = Path(path_text)
        if not candidate.exists():
            return None, "file_missing"
        try:
            return candidate.read_text(encoding="utf-8", errors="ignore"), None
        except OSError as exc:
            return None, str(exc)
    try:
        raw = _fetch_url(url)
    except Exception as exc:  # pragma: no cover - network errors
        return None, str(exc)
    return raw.decode("utf-8", errors="ignore"), None


@dataclass
class _EightKProcessResult:
    url: str
    event: Optional[EightKEvent]
    csv_row: Optional[dict[str, object]]
    debug_entry: Optional[list[object]]
    log_messages: list[str]


def _process_eight_k_row(row) -> _EightKProcessResult:
    url = _clean_text(getattr(row, "URL", ""))
    if not url:
        return _EightKProcessResult(url="", event=None, csv_row=None, debug_entry=None, log_messages=[])

    log_messages: list[str] = []

    ticker = _normalize_ticker(getattr(row, "Ticker", ""))
    cik = _normalize_cik(getattr(row, "CIK", ""))
    identifier = ticker or cik or "unknown"

    fetch_started = time.time()
    html_text, fetch_error = _read_eight_k_html(url)
    fetch_elapsed = time.time() - fetch_started
    if fetch_elapsed > 10:
        log_messages.append(f"eight_k: slow fetch {fetch_elapsed:.1f}s for {url}")

    if html_text is None:
        reason = f"fetch_failed:{fetch_error}"
        log_messages.append(
            f"eight_k: {identifier} fetch failed url {url} reason {reason}"
        )
        debug_entry = [
            utc_now_iso(),
            url,
            reason,
            "",
            0,
            0,
            "",
        ]
        return _EightKProcessResult(url=url, event=None, csv_row=None, debug_entry=debug_entry, log_messages=log_messages)

    form_hint = _clean_text(getattr(row, "Form", "")) or "8-K"
    parse_started = time.time()
    if parse_k8 is None:
        reason = "parse_error:legacy_parser_missing"
        log_messages.append(
            f"eight_k: {identifier} parse skipped url {url} reason {reason}"
        )
        debug_entry = [
            utc_now_iso(),
            url,
            reason,
            "",
            len(html_text.encode("utf-8", errors="ignore")),
            0,
            preview_text(html_text),
        ]
        return _EightKProcessResult(url=url, event=None, csv_row=None, debug_entry=debug_entry, log_messages=log_messages)

    try:
        result = parse_k8.parse(url, html=html_text, form_hint=form_hint)
    except Exception as exc:  # pragma: no cover - unexpected parser failures
        reason = f"parse_error:{exc}"
        log_messages.append(
            f"eight_k: {identifier} parse failed url {url} reason {reason}"
        )
        debug_entry = [
            utc_now_iso(),
            url,
            reason,
            "",
            len(html_text.encode("utf-8", errors="ignore")),
            0,
            preview_text(html_text),
        ]
        return _EightKProcessResult(url=url, event=None, csv_row=None, debug_entry=debug_entry, log_messages=log_messages)

    parse_elapsed = time.time() - parse_started
    if parse_elapsed > 10:
        log_messages.append(f"eight_k: slow parse {parse_elapsed:.1f}s for {url}")

    event = _build_eight_k_event(row, html_text, result)
    exhibits = result.get("exhibits") or []
    exhibit_bytes = sum(len(_clean_text(exhibit.get("text", ""))) for exhibit in exhibits)

    if event is None:
        items = result.get("items") or []
        items_present = _join_items(item.get("item", "") for item in items)
        ignore_reason = _clean_text((result.get("classification") or {}).get("ignore_reason"))
        reason_suffix = f" ({ignore_reason})" if ignore_reason else ""
        log_messages.append(
            f"eight_k: {identifier} no actionable items{reason_suffix} url {url} items {items_present or 'none'}"
        )
        debug_entry = [
            utc_now_iso(),
            url,
            "no_actionable_items",
            items_present,
            len(html_text.encode("utf-8", errors="ignore")),
            exhibit_bytes,
            preview_text(html_text),
        ]
        return _EightKProcessResult(url=url, event=None, csv_row=None, debug_entry=debug_entry, log_messages=log_messages)

    csv_row = {
        "CIK": event.cik,
        "Ticker": event.ticker,
        "FilingDate": event.filing_date,
        "FilingURL": event.filing_url,
        "ItemsPresent": event.items_present,
        "IsCatalyst": event.is_catalyst,
        "CatalystType": event.catalyst_type,
        "Tier1Type": event.tier1_type,
        "Tier1Trigger": event.tier1_trigger,
        "IsDilution": event.is_dilution,
        "DilutionTags": event.dilution_tags_joined(),
        "IgnoreReason": event.ignore_reason,
    }

    catalyst_display = event.catalyst_type if event.is_catalyst else "NONE"
    dilution_display = "yes" if event.is_dilution else "no"
    log_messages.append(
        " ".join(
            [
                f"eight_k: {identifier} parsed url {url}",
                f"items {event.items_present or 'none'}",
                f"catalyst {catalyst_display}",
                f"tier1_type {event.tier1_type or 'none'}",
                f"tier1_trigger {event.tier1_trigger or 'none'}",
                f"dilution {dilution_display}",
                f"dilution_tags {event.dilution_tags_joined() or 'none'}",
                f"ignore_reason {event.ignore_reason or 'none'}",
            ]
        )
    )

    return _EightKProcessResult(
        url=url,
        event=event,
        csv_row=csv_row,
        debug_entry=None,
        log_messages=log_messages,
    )


def _format_event_label(event: EightKEvent) -> str:
    if not event.is_catalyst:
        return ""
    if event.catalyst_label:
        return event.catalyst_label
    return event.catalyst_type


def _build_eight_k_event(
    row,
    html_text: str,
    result: dict,
) -> Optional[EightKEvent]:
    items = result.get("items") or []
    classification = result.get("classification") or {}
    exhibits = result.get("exhibits") or []
    items_present = _join_items(item.get("item", "") for item in items)
    if not items_present:
        return None

    is_catalyst = bool(classification.get("is_catalyst"))
    is_dilution = bool(classification.get("is_dilution"))
    if not (is_catalyst or is_dilution):
        return None

    ignore_reason = _clean_text(classification.get("ignore_reason"))

    tier_raw = _clean_text(classification.get("tier"))
    catalyst_type = tier_raw if tier_raw in {"Tier-1", "Tier-2"} else "NONE"
    tier1_type = _clean_text(classification.get("tier1_type"))
    tier1_trigger = _clean_text(classification.get("tier1_trigger"))

    dilution_tags = []
    for tag in classification.get("dilution_tags") or []:
        text = _clean_text(tag)
        if text and text not in dilution_tags:
            dilution_tags.append(text)

    filing_date = _clean_text(result.get("filing_date"))
    if not filing_date:
        filed_at = getattr(row, "FiledAt", "")
        parsed_date = _parse_date_value(filed_at)
        filing_date = _format_date(parsed_date)

    cik = _normalize_cik(getattr(row, "CIK", ""))
    ticker = _normalize_ticker(getattr(row, "Ticker", ""))
    filing_url = _clean_text(getattr(row, "URL", "")) or _clean_text(result.get("url", ""))

    label_parts: list[str] = []
    if is_catalyst:
        label_parts.append(catalyst_type)
        if catalyst_type == "Tier-1" and tier1_type:
            label_parts.append(tier1_type)
    catalyst_label = " ".join(part for part in label_parts if part)

    return EightKEvent(
        cik=cik,
        ticker=ticker,
        filing_date=filing_date or "",
        filing_url=filing_url,
        items_present=items_present,
        is_catalyst=is_catalyst,
        catalyst_type=catalyst_type or "NONE",
        catalyst_label=catalyst_label,
        tier1_type=tier1_type,
        tier1_trigger=tier1_trigger,
        is_dilution=is_dilution,
        dilution_tags=dilution_tags,
        ignore_reason=ignore_reason,
    )


def _generate_eight_k_events(
    data_dir: str,
    progress_fn: ProgressFn,
) -> tuple[pd.DataFrame, EightKLookup]:
    filings_path = _resolve_path(csv_filename("filings"), data_dir)
    if not os.path.exists(filings_path):
        events_df = pd.DataFrame(columns=_EIGHT_K_EVENTS_COLUMNS)
        events_df.to_csv(csv_path(data_dir, "eight_k_events"), index=False)
        _emit("INFO", "eight_k: parsed 0", progress_fn)
        _emit("INFO", "eight_k: failed 0", progress_fn)
        return events_df, EightKLookup([])

    _emit("INFO", f"eight_k: loading filings from {filings_path}", progress_fn)

    filings_df = pd.read_csv(filings_path, encoding="utf-8")
    if filings_df.empty or "Form" not in filings_df.columns or "URL" not in filings_df.columns:
        events_df = pd.DataFrame(columns=_EIGHT_K_EVENTS_COLUMNS)
        events_df.to_csv(csv_path(data_dir, "eight_k_events"), index=False)
        _emit("INFO", "eight_k: parsed 0", progress_fn)
        _emit("INFO", "eight_k: failed 0", progress_fn)
        return events_df, EightKLookup([])

    df = filings_df.copy()
    df["Form_norm"] = df["Form"].astype(str).str.upper()
    df = df[df["Form_norm"].str.startswith("8-K")]
    if df.empty:
        events_df = pd.DataFrame(columns=_EIGHT_K_EVENTS_COLUMNS)
        events_df.to_csv(csv_path(data_dir, "eight_k_events"), index=False)
        _emit("INFO", "eight_k: parsed 0", progress_fn)
        _emit("INFO", "eight_k: failed 0", progress_fn)
        return events_df, EightKLookup([])

    df = df.drop_duplicates(subset=["URL"], keep="last")

    total_filings = len(df.index)
    _emit("INFO", f"eight_k: {total_filings} unique 8-K filings queued", progress_fn)
    _emit("INFO", f"eight_k: (0.0%) processing {total_filings} filings", progress_fn)
    report_every = max(1, total_filings // 20)

    last_reported_pct_tenths = 0

    def _emit_progress(processed_count: int) -> None:
        nonlocal last_reported_pct_tenths
        if total_filings <= 0:
            return
        pct_tenths = int(round((processed_count / total_filings) * 1000))
        pct_tenths = min(max(pct_tenths, 0), 1000)
        if pct_tenths == last_reported_pct_tenths and processed_count != total_filings:
            return
        last_reported_pct_tenths = pct_tenths
        pct_display = pct_tenths / 10
        _emit(
            "INFO",
            f"eight_k: ({pct_display:.1f}%) processed {processed_count}/{total_filings} filings",
            progress_fn,
        )

    csv_rows: list[dict[str, object]] = []
    events: list[EightKEvent] = []
    debug_entries: list[list[object]] = []

    processed = 0
    last_reported_parsed = 0
    last_reported_failed = 0

    last_heartbeat = time.time()

    _emit("INFO", "eight_k: start", progress_fn)
    _emit_progress(0)

    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_eight_k_row, row) for row in df.itertuples(index=False)]

        for future in as_completed(futures):
            result = future.result()
            processed += 1

            _emit_progress(processed)

            if result.url and (processed <= 3 or processed % max(1, report_every * 2) == 0):
                _emit("INFO", f"eight_k: fetching {processed}/{total_filings} {result.url}", progress_fn)

            for log_message in result.log_messages:
                _emit("INFO", log_message, progress_fn)

            if result.debug_entry is not None:
                debug_entries.append(result.debug_entry)
                if len(debug_entries) != last_reported_failed:
                    last_reported_failed = len(debug_entries)
                    _emit(
                        "INFO",
                        f"eight_k: failed {last_reported_failed}",
                        progress_fn,
                    )

            if result.event is not None and result.csv_row is not None:
                events.append(result.event)
                csv_rows.append(result.csv_row)
                if len(events) != last_reported_parsed:
                    last_reported_parsed = len(events)
                    _emit(
                        "INFO",
                        f"eight_k: parsed {last_reported_parsed}",
                        progress_fn,
                    )

            now = time.time()
            if now - last_heartbeat > 30:
                _emit(
                    "INFO",
                    f"eight_k: heartbeat processed {processed}/{total_filings} (parsed {len(events)} failed {len(debug_entries)})",
                    progress_fn,
                )
                last_heartbeat = now

    events_df = pd.DataFrame(csv_rows, columns=_EIGHT_K_EVENTS_COLUMNS)
    events_df.to_csv(csv_path(data_dir, "eight_k_events"), index=False)

    _eight_k_debug_path()
    _write_eight_k_debug(debug_entries)

    _emit("INFO", f"eight_k: parsed {len(events_df)}", progress_fn)
    _emit("INFO", f"eight_k: failed {len(debug_entries)}", progress_fn)
    _emit("INFO", f"eight_k: complete – wrote {len(events_df)} rows", progress_fn)

    return events_df, EightKLookup(events)


def _load_eight_k_events_from_csv(
    data_dir: str,
) -> tuple[pd.DataFrame, EightKLookup] | None:
    events_path = csv_path(data_dir, "eight_k_events")
    if not os.path.exists(events_path):
        return None

    try:
        df = pd.read_csv(events_path, encoding="utf-8")
    except Exception:
        return None

    if df.empty:
        return df, EightKLookup([])

    events: list[EightKEvent] = []
    for row in df.itertuples(index=False):
        dilution_tags_text = _clean_text(getattr(row, "DilutionTags", ""))
        dilution_tags = [tag.strip() for tag in dilution_tags_text.split(";") if tag.strip()]
        event = EightKEvent(
            cik=_normalize_cik(getattr(row, "CIK", "")),
            ticker=_normalize_ticker(getattr(row, "Ticker", "")),
            filing_date=_clean_text(getattr(row, "FilingDate", "")),
            filing_url=_clean_text(getattr(row, "FilingURL", "")),
            items_present=_clean_text(getattr(row, "ItemsPresent", "")),
            is_catalyst=_coerce_bool(getattr(row, "IsCatalyst", False)),
            catalyst_type=_clean_text(getattr(row, "CatalystType", "")) or "NONE",
            catalyst_label="",
            tier1_type=_clean_text(getattr(row, "Tier1Type", "")),
            tier1_trigger=_clean_text(getattr(row, "Tier1Trigger", "")),
            is_dilution=_coerce_bool(getattr(row, "IsDilution", False)),
            dilution_tags=dilution_tags,
            ignore_reason=_clean_text(getattr(row, "IgnoreReason", "")),
        )
        events.append(event)

    return df, EightKLookup(events)


def generate_eight_k_events(
    data_dir: str | None = None,
    progress_fn: ProgressFn = None,
) -> tuple[pd.DataFrame, EightKLookup]:
    if data_dir is None:
        cfg = load_config()
        data_dir = cfg.get("Paths", {}).get("data", "data")

    os.makedirs(data_dir, exist_ok=True)
    return _generate_eight_k_events(data_dir, progress_fn)


def load_or_generate_eight_k_events(
    data_dir: str,
    progress_fn: ProgressFn,
) -> tuple[pd.DataFrame, EightKLookup]:
    os.makedirs(data_dir, exist_ok=True)
    loaded = _load_eight_k_events_from_csv(data_dir)
    if loaded is not None:
        df, lookup = loaded
        _emit("INFO", f"eight_k: loaded {len(df)} events from {csv_filename('eight_k_events')}", progress_fn)
        return df, lookup

    return _generate_eight_k_events(data_dir, progress_fn)


def _collect_candidate_urls(row) -> list[str]:
    texts = [
        _clean_text(getattr(row, "CatalystPrimaryLink", "")),
        _clean_text(getattr(row, "CatalystPrimaryURL", "")),
        _clean_text(getattr(row, "Catalyst", "")),
        _clean_text(getattr(row, "Dilution", "")),
        _clean_text(getattr(row, "DilutionLinks", "")),
        _clean_text(getattr(row, "FilingsSummary", "")),
    ]
    urls: list[str] = []
    for text in texts:
        urls.extend(_extract_urls_list(text))
    return urls


def _apply_event_to_catalyst_summary(
    summary: dict,
    event: EightKEvent,
    now: pd.Timestamp,
) -> dict:
    if not event.is_catalyst:
        return summary
    updated = dict(summary)
    updated["status"] = "Pass"
    updated["primary_form"] = "8-K"
    updated["primary_url"] = event.filing_url
    updated["primary_text"] = _format_event_label(event)
    updated["primary_raw"] = updated["primary_text"]
    updated["primary_entry"] = None
    updated["primary_date"] = event.filing_date or summary.get("primary_date", "")
    date_val = _parse_date_value(event.filing_date) if event.filing_date else None
    updated["primary_days_ago"] = _compute_days_ago(date_val, now)
    return updated


def _csv_cell_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    return value


def _round_half_up_number(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        quant = Decimal("1").scaleb(-digits)
        return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return float(value)


def _runway_bucket(quarters: Optional[float]) -> str:
    if quarters is None:
        return ""
    if quarters < 1:
        return "<1q"
    if quarters < 3:
        return "1-3q"
    if quarters < 6:
        return "3-6q"
    if quarters < 12:
        return "6-12q"
    return ">12q"


def _normalize_form_name(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"\s+", " ", text)
    return text


def _form_guard_key(value: str) -> str:
    text = _normalize_form_name(value)
    if text.startswith("FORM "):
        text = text[5:]
    text = text.replace(" ", "")
    if text.endswith("/A"):
        text = text[:-2]
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_manager_form(value: str) -> bool:
    text = _normalize_form_name(value)
    return any(text.startswith(prefix) for prefix in _MANAGER_FORM_PREFIXES)


def _is_dilution_only_form(value: str) -> bool:
    key = _form_guard_key(value)
    return key in _CATALYST_DILUTION_FORM_GUARDS


def _url_matches_form(form: str, url: str) -> bool:
    if not form or not url:
        return True

    key = _form_guard_key(form)
    pattern = next(
        (regex for prefix, regex in _FORM_URL_PATTERNS.items() if key.startswith(prefix)),
        None,
    )
    if pattern is None:
        return True

    parsed = urlsplit(url)
    lowered_url = url.lower()

    doc_candidates: list[str] = []
    if parsed.query:
        query = parse_qs(parsed.query)
        for candidate_key in ("doc", "filename", "file", "document"):
            for value in query.get(candidate_key, []):
                if value:
                    doc_candidates.append(unquote(value))

    text_to_check = " ".join([lowered_url, *[candidate.lower() for candidate in doc_candidates]])
    if pattern.search(text_to_check):
        return True

    for other_prefix, other_pattern in _FORM_URL_PATTERNS.items():
        if other_prefix == key:
            continue
        if other_pattern.search(text_to_check):
            return False

    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    doc_paths = [candidate.lower() for candidate in doc_candidates]

    if netloc.endswith("sec.gov"):
        if "/archives/edgar/data/" in path:
            return True
        if path.startswith(("/ix", "/cgi-bin/ix", "/cgi-bin/viewer")):
            return True
        if any("/archives/edgar/data/" in candidate for candidate in doc_paths):
            return True

    return False


def _parse_date_value(value: object) -> Optional[pd.Timestamp]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        if _is_iso_date_string(text):
            ts = pd.to_datetime(text, utc=True, errors="coerce", format="%Y-%m-%d")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
            ts = pd.to_datetime(text, utc=True, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        else:
            ts = pd.to_datetime(text, utc=True, errors="coerce", dayfirst=True)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        if ts.empty:
            return None
        ts = ts.iloc[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.normalize()


def _format_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d")


def _is_iso_date_string(value: str) -> bool:
    return bool(value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _compute_days_ago(ts: Optional[pd.Timestamp], now: pd.Timestamp) -> str:
    if ts is None:
        return ""
    delta = now - ts
    if pd.isna(delta):
        return ""
    try:
        return str(int(delta.days))
    except Exception:
        return ""


def _select_primary_entry(entries: list[dict]) -> Optional[dict]:
    if not entries:
        return None

    def sort_key(entry: dict) -> tuple[int, pd.Timestamp, int]:
        date_val = entry.get("date_value")
        has_date = 1 if isinstance(date_val, pd.Timestamp) else 0
        sortable = date_val if isinstance(date_val, pd.Timestamp) else pd.Timestamp.min.tz_localize("UTC")
        has_url = 1 if entry.get("url") else 0
        return (has_date, sortable, has_url)

    ordered = sorted(entries, key=sort_key, reverse=True)
    return ordered[0] if ordered else None


def _entry_core(entry: dict) -> Optional[tuple[str, str, str]]:
    url = _clean_text(entry.get("url"))
    if not url:
        return None

    date_value = entry.get("date_value")
    if isinstance(date_value, pd.Timestamp):
        date_str = _format_date(date_value)
    else:
        date_str = entry.get("date") or ""
    if date_str and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        parsed = _parse_date_value(date_str)
        date_str = _format_date(parsed)

    form = _normalize_form_name(entry.get("form", ""))

    if not (date_str and form):
        return None

    return date_str, form, url


def _extract_entry_description(entry: dict, url: str) -> str:
    raw_text = entry.get("raw") or ""
    if not raw_text:
        return ""

    idx = raw_text.find(url) if url else -1
    if idx != -1:
        remainder = raw_text[idx + len(url) :]
    else:
        match = _URL_RE.search(raw_text)
        if not match:
            return ""
        remainder = raw_text[match.end() :]

    desc_text = remainder.strip(" |")
    if not desc_text:
        return ""
    desc_text = re.sub(r"\s+", " ", desc_text)
    return desc_text


def _entry_to_text(entry: dict) -> str:
    core = _entry_core(entry)
    if core is None:
        return ""

    date_str, form, url = core
    base_text = f"{date_str} {form} {url}"
    desc_text = _extract_entry_description(entry, url)
    if desc_text:
        return f"{base_text} {desc_text}"
    return base_text


def _format_evidence_entries(entries: list[dict]) -> str:
    formatted: list[str] = []
    seen_core: set[str] = set()
    for entry in entries:
        core = _entry_core(entry)
        if core is None:
            continue
        core_text = f"{core[0]} {core[1]} {core[2]}"
        if core_text in seen_core:
            continue
        seen_core.add(core_text)
        text = _entry_to_text(entry)
        if text:
            formatted.append(text)
    return "; ".join(formatted)


def detect_tier1(evidence_text: str) -> tuple[str, bool]:
    text = (evidence_text or "").lower()
    patterns = [
        ("FDA", r"(510\(k\)|de ?novo|pdufa|approval|cleared|clearance|crl|complete response letter|adcom vote|ind (accepted|cleared))"),
        ("GuidanceUp", r"(raises|increases|hikes)\s+guidance|guidance.*(raised|increased)"),
        ("FundedAward", r"(award|contract).*(obligated|obligation).*\$[0-9]"),
        ("OverhangRemoval", r"(terminate(d|s)?|withdraw(n|s)?)\s.*(atm|shelf|sales agreement|equity distribution)"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            return label, True
    return "None", False


def _enforce_form_url_guard(
    ticker: str,
    subscore: str,
    summary: dict,
    entries: list[dict],
    now: pd.Timestamp,
    progress_fn: ProgressFn,
) -> dict:
    form = summary.get("primary_form", "")
    url = summary.get("primary_url", "")
    if not form or not url:
        return summary

    if _url_matches_form(form, url):
        return summary

    message = f"Fix(FormURLMismatch): {ticker} {subscore} expected={form} gotURL={url}"

    for entry in entries:
        candidate_form = entry.get("form") or ""
        candidate_url = entry.get("url") or ""
        if candidate_form and candidate_url and _url_matches_form(candidate_form, candidate_url):
            summary["primary_entry"] = entry
            summary["primary_form"] = candidate_form
            summary["primary_date"] = entry.get("date", "")
            summary["primary_days_ago"] = _compute_days_ago(entry.get("date_value"), now)
            summary["primary_url"] = candidate_url
            summary["primary_text"] = _entry_to_text(entry)
            summary["primary_raw"] = entry.get("raw", "")
            summary["status"] = "Pass"
            _emit("WARN", message, progress_fn)
            return summary

    _emit("WARN", message, progress_fn)
    return summary


def _extract_urls_list(text: str) -> list[str]:
    urls = _URL_RE.findall(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _infer_catalyst_type(entries: list[dict], fallback_form: str, fallback_text: str) -> str:
    for entry in entries:
        lower_text = entry.get("lower", "")
        for item, label in _CATALYST_ITEM_LABELS.items():
            if f"item {item.lower()}" in lower_text:
                return label
        for pattern, label in _CATALYST_KEYWORD_LABELS:
            if pattern.search(lower_text):
                return label

    form = _normalize_form_name(fallback_form)
    if form in {"10-Q", "10-K", "20-F", "40-F"}:
        return "Earnings/Financial update"

    lower_fallback = fallback_text.lower()
    for pattern, label in _CATALYST_KEYWORD_LABELS:
        if pattern.search(lower_fallback):
            return label
    return ""


def _is_status_pass(status: str) -> bool:
    text = _clean_text(status).lower()
    if not text:
        return False
    return text.startswith("pass") or text.startswith("overhang")


def _normalize_checklist_status(status_value: object) -> str:
    text = _clean_text(status_value)
    if not text:
        return "Missing"

    if _is_status_pass(text):
        return "Pass"

    lowered = text.lower()
    if lowered in {"missing", "placeholder", "tbd", "n/a", "na", "not available"}:
        return "Missing"

    return "Fail"


def _parse_evidence_entries(text: str) -> list[dict]:
    entries: list[dict] = []
    if not text:
        return entries

    for segment in _SUBSCORE_SPLIT_RE.split(text):
        raw = segment.strip()
        if not raw:
            continue

        url_match = _URL_RE.search(raw)
        url = url_match.group(0) if url_match else ""

        date_match = _DATE_RE.search(raw)
        date_val = _parse_date_value(date_match.group(1)) if date_match else None

        remainder = raw
        if date_match:
            remainder = raw[date_match.end() :].lstrip(" |") or remainder

        candidate_segment = remainder.split("|")[0].strip()
        if candidate_segment:
            http_pos = candidate_segment.lower().find("http")
            if http_pos != -1:
                candidate_segment = candidate_segment[:http_pos].rstrip()

        form = ""
        if candidate_segment:
            match = re.match(r"([A-Za-z0-9./-]+(?:\s+[A-Za-z0-9.-]+)?)", candidate_segment)
            if match:
                form = match.group(1)

        normalized_form = _normalize_form_name(form)

        if date_val is None:
            # attempt to parse the leading token if a date wasn't captured by regex
            leading_token = candidate_segment.split()[0] if candidate_segment else ""
            alt_date = _parse_date_value(leading_token)
            if alt_date is not None:
                date_val = alt_date

        entries.append(
            {
                "raw": raw,
                "date": _format_date(date_val),
                "date_value": date_val,
                "form": normalized_form,
                "url": url,
                "lower": raw.lower(),
            }
        )

    return entries


def _first_url(text: str) -> str:
    if not text:
        return ""
    match = _URL_RE.search(text)
    if match:
        return match.group(0)
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _URL_RE.search(chunk)
        if match:
            return match.group(0)
    return ""


def _summarize_category(entries: list[dict], now: pd.Timestamp) -> dict:
    summary = {
        "status": "Missing",
        "primary_text": "",
        "primary_raw": "",
        "primary_form": "",
        "primary_date": "",
        "primary_days_ago": "",
        "primary_url": "",
        "primary_entry": None,
    }

    primary = _select_primary_entry(entries)
    if not primary:
        return summary

    summary["status"] = "Pass"
    summary["primary_entry"] = primary
    summary["primary_form"] = primary.get("form", "")
    summary["primary_date"] = primary.get("date", "")
    summary["primary_days_ago"] = _compute_days_ago(primary.get("date_value"), now)
    summary["primary_url"] = primary.get("url", "")
    summary["primary_text"] = _entry_to_text(primary)
    summary["primary_raw"] = primary.get("raw", "")
    return summary


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and pd.isna(value):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(result):
            return None
        return result
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        result = float(text)
    except ValueError:
        return None
    if pd.isna(result):
        return None
    return result


def _select_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", col.lower()): col for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _load_market_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    ticker_col = _select_column(df, ["Ticker"])
    cik_col = _select_column(df, ["CIK"])
    price_col = _select_column(df, ["Price", "Close", "Last", "LastPrice"])
    mc_col = _select_column(df, ["MarketCap", "Market_Cap", "MarketCapitalization"])
    adv_col = _select_column(df, ["ADV20", "AverageVolume20", "AverageVolume20Day"])

    columns = {}
    if ticker_col:
        columns["Ticker"] = df[ticker_col].apply(_normalize_ticker)
    else:
        columns["Ticker"] = pd.Series(dtype="string")

    if cik_col:
        columns["CIK"] = df[cik_col].apply(_normalize_cik)
    else:
        columns["CIK"] = pd.Series(dtype="string")

    if price_col:
        columns["Price"] = df[price_col]
    else:
        columns["Price"] = pd.Series(dtype="float64")

    if mc_col:
        columns["MarketCap"] = df[mc_col]
    else:
        columns["MarketCap"] = pd.Series(dtype="float64")

    if adv_col:
        columns["ADV20"] = df[adv_col]
    else:
        columns["ADV20"] = pd.Series(dtype="float64")

    market = pd.DataFrame(columns)
    market["Ticker"] = market["Ticker"].fillna("").astype(str).str.strip().str.upper()
    market["CIK"] = market["CIK"].fillna("").astype(str).apply(_normalize_cik)
    return market


def _merge_market_info(df: pd.DataFrame, market: pd.DataFrame) -> None:
    if market.empty:
        return

    market = market.copy()
    market["Ticker_norm"] = market["Ticker"].apply(_normalize_ticker)
    market["CIK_norm"] = market["CIK"].apply(_normalize_cik)

    by_ticker = (
        market[market["Ticker_norm"] != ""]
        .drop_duplicates(subset=["Ticker_norm"], keep="last")
        .set_index("Ticker_norm")
    )

    by_cik = (
        market[market["CIK_norm"] != ""]
        .drop_duplicates(subset=["CIK_norm"], keep="last")
        .set_index("CIK_norm")
    )

    for column in ["Price", "MarketCap", "ADV20"]:
        if column not in df.columns:
            df[column] = pd.Series(dtype="float64")

        if not by_ticker.empty and column in by_ticker.columns:
            ticker_series = df["Ticker_norm"].map(by_ticker[column])
            df[column] = df[column].where(df[column].notna(), ticker_series)

        if not by_cik.empty and column in by_cik.columns:
            cik_series = df["CIK_norm"].map(by_cik[column])
            df[column] = df[column].where(df[column].notna(), cik_series)


def run(
    data_dir: str | None = None,
    progress_fn: ProgressFn = None,
    stop_flag: dict | None = None,
) -> tuple[int, str]:
    """Build the validated watchlist.

    Returns a tuple ``(rows_written, status)`` where ``status`` is one of
    ``"ok"``, ``"missing_source"``, or ``"stopped"``.
    """

    if stop_flag is None:
        stop_flag = {}

    if data_dir is None:
        cfg = load_config()
        data_dir = cfg.get("Paths", {}).get("data", "data")

    os.makedirs(data_dir, exist_ok=True)

    research_path = _resolve_path(csv_filename("dr_populate_results"), data_dir)
    if not os.path.exists(research_path):
        _emit("ERROR", f"build_watchlist: {csv_filename('dr_populate_results')} not found", progress_fn)
        return 0, "missing_source"

    _emit("INFO", f"build_watchlist: reading {csv_filename('dr_populate_results')}", progress_fn)
    research_df = pd.read_csv(research_path, encoding="utf-8")

    if research_df.empty:
        _emit("INFO", "build_watchlist: research results empty", progress_fn)

    research_df = research_df.copy()

    if "Ticker" not in research_df.columns:
        research_df["Ticker"] = ""
    if "CIK" not in research_df.columns:
        research_df["CIK"] = ""
    if "SubscoresEvidenced" not in research_df.columns:
        research_df["SubscoresEvidenced"] = ""

    research_df["Ticker_norm"] = research_df["Ticker"].apply(_normalize_ticker)
    research_df["CIK_norm"] = research_df["CIK"].apply(_normalize_cik)
    research_df["SubscoresEvidencedCount"] = research_df["SubscoresEvidenced"].apply(_count_subscores)

    candidates_path = _resolve_path(csv_filename("hydrated_candidates"), data_dir)
    prices_path = _resolve_path(csv_filename("prices"), data_dir)

    market_frames = []
    for path in (candidates_path, prices_path):
        market_df = _load_market_data(path)
        if not market_df.empty:
            _emit("INFO", f"build_watchlist: loaded market data from {os.path.basename(path)}", progress_fn)
            market_frames.append(market_df)
        else:
            if os.path.exists(path):
                _emit("INFO", f"build_watchlist: {os.path.basename(path)} has no usable rows", progress_fn)
            else:
                _emit("INFO", f"build_watchlist: {os.path.basename(path)} not found", progress_fn)

    combined_market = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
    _merge_market_info(research_df, combined_market)

    total_rows = len(research_df)
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    _, eight_k_lookup = load_or_generate_eight_k_events(data_dir, progress_fn)

    survivors: list[dict] = []
    status = "ok"

    for idx, row in enumerate(research_df.itertuples(index=False), start=1):
        if stop_flag.get("stop"):
            status = "stopped"
            _emit("INFO", "build_watchlist: stop requested", progress_fn)
            break

        if idx == 1 or idx % 25 == 0 or idx == total_rows:
            _emit("INFO", f"build_watchlist: processing row {idx} of {total_rows}", progress_fn)

        ticker_value = _clean_text(getattr(row, "Ticker", ""))
        cik_value = _normalize_cik(getattr(row, "CIK", ""))
        identifier = ticker_value or cik_value
        company_value = getattr(row, "Company", "")
        sector_text = _clean_text(getattr(row, "Sector", ""))

        row_urls = _collect_candidate_urls(row)
        eight_k_event = eight_k_lookup.match(ticker_value, cik_value, row_urls)

        price_val = _to_float(getattr(row, "Price", None))
        market_cap_val = _to_float(getattr(row, "MarketCap", None))
        adv_val = _to_float(getattr(row, "ADV20", None))

        if price_val is None or price_val < 1:
            continue
        if market_cap_val is None or market_cap_val >= 350_000_000:
            continue
        if adv_val is None or adv_val < 40_000:
            continue

        runway_quarters_raw = _to_float(getattr(row, "RunwayQuartersRaw", None))
        if runway_quarters_raw is None:
            runway_quarters_raw = _to_float(getattr(row, "RunwayQuarters", None))
        runway_quarters = (
            _round_half_up_number(runway_quarters_raw)
            if runway_quarters_raw is not None
            else None
        )

        runway_cash_raw = _to_float(getattr(row, "RunwayCashRaw", None))
        if runway_cash_raw is None:
            runway_cash_raw = _to_float(getattr(row, "RunwayCash", None))
        runway_cash = _round_half_up_number(
            _to_float(getattr(row, "RunwayCash", runway_cash_raw))
        )

        runway_burn_raw = _to_float(getattr(row, "RunwayQuarterlyBurnRaw", None))
        if runway_burn_raw is None:
            runway_burn_raw = _to_float(getattr(row, "RunwayQuarterlyBurn", None))
        runway_burn = _round_half_up_number(
            _to_float(getattr(row, "RunwayQuarterlyBurn", runway_burn_raw))
        )

        runway_estimate = _clean_text(getattr(row, "RunwayEstimate", ""))
        runway_notes = _clean_text(getattr(row, "RunwayNotes", ""))

        runway_burn_missing = (
            (runway_burn_raw is None or runway_burn_raw <= 0)
            and (runway_burn is None or runway_burn <= 0)
        )
        if runway_burn_missing or not runway_estimate:
            _emit("WARN", f"Dropped: Missing runway burn metrics ({identifier})", progress_fn)
            continue
        runway_source_form = _normalize_form_name(_clean_text(getattr(row, "RunwaySourceForm", "")))
        runway_source_date_ts = _parse_date_value(getattr(row, "RunwaySourceDate", ""))
        runway_source_url = _clean_text(getattr(row, "RunwaySourceURL", ""))

        has_runway_values = (
            runway_quarters is not None
            and runway_cash is not None
            and runway_burn is not None
        )

        runway_days_value: Optional[int] = None
        runway_bucket_label = ""

        if has_runway_values and runway_quarters is not None:
            runway_days_value = int(round(runway_quarters * 90))
            runway_bucket_label = _runway_bucket(runway_quarters)
            if not runway_source_form or runway_source_date_ts is None or not runway_source_url:
                _emit("WARN", f"Dropped: Missing runway source details ({identifier})", progress_fn)
                continue
            if (now_utc - runway_source_date_ts).days > 465:
                _emit("WARN", f"Dropped: Runway filing stale ({identifier})", progress_fn)
                continue
            runway_status = "Pass" if runway_quarters > 0 else "Fail"
        else:
            runway_status = "Missing"
            runway_estimate = ""
            runway_source_form = ""
            runway_source_date_ts = None
            runway_source_url = ""

        runway_source_date = _format_date(runway_source_date_ts)
        runway_source_days = _compute_days_ago(runway_source_date_ts, now_utc)

        catalyst_text = _clean_text(getattr(row, "Catalyst", ""))
        event_catalyst = bool(eight_k_event and eight_k_event.is_catalyst)
        if not catalyst_text and event_catalyst:
            catalyst_text = _format_event_label(eight_k_event)
        if (not catalyst_text or catalyst_text.upper() == "TBD") and not event_catalyst:
            _emit("WARN", f"Dropped: Catalyst evidence missing ({identifier})", progress_fn)
            continue

        dilution_text = _clean_text(getattr(row, "Dilution", ""))
        if any(term in dilution_text.lower() for term in _NEGATIVE_DILUTION_TERMS):
            continue

        governance_text = _clean_text(getattr(row, "Governance", ""))
        if not governance_text:
            continue

        filings_summary_text = _clean_text(getattr(row, "FilingsSummary", ""))
        filing_entries = [
            entry
            for entry in _parse_evidence_entries(filings_summary_text)
            if not _is_manager_form(entry.get("form", ""))
        ]

        latest_entry = _select_primary_entry(filing_entries)
        latest_form = _normalize_form_name(latest_entry.get("form", "")) if latest_entry else ""
        if not latest_form:
            fallback_form = _normalize_form_name(_clean_text(getattr(row, "LatestForm", "")))
            if fallback_form and not _is_manager_form(fallback_form):
                latest_form = fallback_form
        latest_filed_ts = latest_entry.get("date_value") if latest_entry else None
        filed_at_fallback = _parse_date_value(getattr(row, "FiledAt", None))
        if latest_filed_ts is None:
            latest_filed_ts = filed_at_fallback
        latest_filed_at = _format_date(latest_filed_ts)
        latest_filed_age_days = _compute_days_ago(latest_filed_ts, now_utc)

        catalyst_entries = _parse_evidence_entries(catalyst_text)
        catalyst_entries = [
            entry
            for entry in catalyst_entries
            if not _is_dilution_only_form(entry.get("form", ""))
        ]
        catalyst_summary = _summarize_category(catalyst_entries, now_utc)
        catalyst_summary = _enforce_form_url_guard(identifier, "Catalyst", catalyst_summary, catalyst_entries, now_utc, progress_fn)
        if eight_k_event and eight_k_event.is_catalyst:
            catalyst_summary = _apply_event_to_catalyst_summary(catalyst_summary, eight_k_event, now_utc)
        if catalyst_summary.get("status") != "Pass":
            _emit("WARN", f"Dropped: Catalyst evidence missing ({identifier})", progress_fn)
            continue
        catalyst_primary_url = catalyst_summary.get("primary_url") or _clean_text(getattr(row, "CatalystPrimaryLink", ""))
        if catalyst_primary_url:
            catalyst_summary["primary_url"] = catalyst_primary_url
        catalyst_field = catalyst_summary.get("primary_text") or catalyst_text
        catalyst_evidence = _format_evidence_entries(catalyst_entries) or catalyst_text
        catalyst_type = _infer_catalyst_type(
            catalyst_entries,
            catalyst_summary.get("primary_form", ""),
            catalyst_text,
        ) or _clean_text(getattr(row, "CatalystType", ""))
        if eight_k_event and eight_k_event.is_catalyst:
            event_label = _format_event_label(eight_k_event)
            if event_label:
                catalyst_field = event_label
                catalyst_evidence = event_label
            catalyst_type = eight_k_event.catalyst_type or catalyst_type

        dilution_entries = _parse_evidence_entries(dilution_text)
        dilution_summary = _summarize_category(dilution_entries, now_utc)
        dilution_summary = _enforce_form_url_guard(identifier, "Dilution", dilution_summary, dilution_entries, now_utc, progress_fn)
        dilution_primary_url = dilution_summary.get("primary_url") or _first_url(_clean_text(getattr(row, "DilutionLinks", "")))
        if dilution_primary_url:
            dilution_summary["primary_url"] = dilution_primary_url
        dilution_forms_hit = list(
            dict.fromkeys(
                [
                    entry.get("form", "")
                    for entry in dilution_entries
                    if entry.get("form")
                    and any(entry.get("form", "").startswith(prefix) for prefix in _DILUTION_FORM_PREFIXES)
                ]
            )
        )
        dilution_keywords_raw: list[str] = []
        has_keyword_filing = False
        for entry in dilution_entries:
            lower_text = entry.get("lower", "")
            form_name = entry.get("form", "") or ""
            if form_name.startswith(("8-K", "6-K")) and any(keyword in lower_text for keyword in _DILUTION_KEYWORDS):
                has_keyword_filing = True
            for keyword in _DILUTION_KEYWORDS:
                if keyword in lower_text:
                    dilution_keywords_raw.append(keyword)
        dilution_keywords_hit = list(dict.fromkeys(dilution_keywords_raw))
        has_offering_form = any(
            entry.get("form", "").startswith(prefix)
            for entry in dilution_entries
            for prefix in _DILUTION_OFFERING_PREFIXES
        )
        has_s8_only = bool(dilution_forms_hit) and all(form.startswith("S-8") for form in dilution_forms_hit)
        dilution_status = "Missing"
        dilution_flag = False
        if has_offering_form or has_keyword_filing:
            dilution_status = "Pass (Offering)"
            dilution_flag = True
        elif has_s8_only:
            dilution_status = "Overhang (S-8)"
            dilution_flag = True
        elif dilution_entries:
            dilution_status = "None"
            dilution_flag = False

        if eight_k_event and eight_k_event.is_dilution:
            dilution_flag = True
            dilution_status = "Pass (Offering)"
            dilution_summary["primary_form"] = "8-K"
            if eight_k_event.filing_url:
                dilution_summary["primary_url"] = eight_k_event.filing_url
            if eight_k_event.filing_date:
                dilution_summary["primary_date"] = eight_k_event.filing_date
            event_label = _format_event_label(eight_k_event)
            if event_label:
                dilution_summary["primary_text"] = event_label
            dilution_forms_hit = ["8-K"]
            if eight_k_event.dilution_tags:
                dilution_keywords_hit = list(dict.fromkeys(eight_k_event.dilution_tags))

        if not dilution_flag:
            _emit("WARN", f"Dropped: Dilution filing missing ({identifier})", progress_fn)
            continue
        dilution_evidence = _format_evidence_entries(dilution_entries) or dilution_text
        dilution_field = dilution_summary.get("primary_text") or dilution_text
        if eight_k_event and eight_k_event.is_dilution:
            dilution_field = "Pass (Offering)"
            dilution_evidence = f"8-K {eight_k_event.filing_url}".strip()

        governance_entries = _parse_evidence_entries(governance_text)
        governance_summary = _summarize_category(governance_entries, now_utc)
        governance_summary = _enforce_form_url_guard(
            identifier, "Governance", governance_summary, governance_entries, now_utc, progress_fn
        )
        governance_primary_url = governance_summary.get("primary_url") or _first_url(governance_text)
        if governance_primary_url:
            governance_summary["primary_url"] = governance_primary_url
        invalid_governance_form = False
        if (
            governance_summary.get("primary_form")
            and governance_summary["primary_form"] not in _GOVERNANCE_VALID_FORMS
        ):
            invalid_governance_form = True
            governance_summary["status"] = "Missing"
            governance_summary["primary_form"] = ""
            governance_summary["primary_date"] = ""
            governance_summary["primary_days_ago"] = ""
            governance_summary["primary_url"] = ""
            governance_summary["primary_text"] = ""
        governance_evidence = _format_evidence_entries(governance_entries) or governance_text
        governance_field = governance_summary.get("primary_text") or governance_text
        governance_notes = governance_summary.get("primary_raw") or governance_text
        gov_lower = governance_text.lower()
        has_going_concern = "going concern" in gov_lower
        has_material_weakness = "material weakness" in gov_lower
        auditor_text = _clean_text(getattr(row, "Auditor", ""))

        risk_flag = ""
        dilution_lower = dilution_text.lower()
        if "watch" in dilution_lower:
            risk_flag = "Dilution Watch"
        elif "watch" in gov_lower:
            risk_flag = "Governance Watch"

        insider_text = _clean_text(getattr(row, "Insider", ""))
        ownership_text = _clean_text(getattr(row, "Ownership", ""))
        insider_links = _extract_urls_list(insider_text)
        ownership_links = _extract_urls_list(ownership_text)
        insider_links = [url for url in insider_links if _INSIDER_LINK_RE.search(url)]
        ownership_links = [url for url in ownership_links if _OWNERSHIP_LINK_RE.search(url)]
        insider_buy_count_val = _to_float(getattr(row, "InsiderBuyCount", None))
        insider_buy_count = int(insider_buy_count_val) if insider_buy_count_val is not None else 0
        has_insider_cluster_raw = getattr(row, "HasInsiderCluster", False)
        if isinstance(has_insider_cluster_raw, str):
            has_insider_cluster = has_insider_cluster_raw.strip().lower() in {"true", "1", "yes"}
        else:
            has_insider_cluster = bool(has_insider_cluster_raw)

        materiality_text = _clean_text(getattr(row, "Materiality", ""))
        subscores_text = _clean_text(getattr(row, "SubscoresEvidenced", ""))
        subscores_count = int(getattr(row, "SubscoresEvidencedCount", 0) or 0)
        status_text = _clean_text(getattr(row, "Status", ""))
        status_text = status_text.replace("—", "-").replace("–", "-")

        insider_status_label = _normalize_checklist_status(getattr(row, "InsiderStatus", ""))
        ownership_status_label = _normalize_checklist_status(getattr(row, "OwnershipStatus", ""))

        checklist_statuses = {
            "Catalyst": _normalize_checklist_status(catalyst_summary.get("status", "")),
            "Dilution": _normalize_checklist_status(dilution_status),
            "Runway": _normalize_checklist_status(runway_status),
            "Governance": _normalize_checklist_status(governance_summary.get("status", "")),
            "Insider": insider_status_label,
            "Ownership": ownership_status_label,
        }

        evidence_text = f"{catalyst_evidence} {dilution_evidence}".strip()
        tier1_label_detected, tier1_flag = detect_tier1(evidence_text)
        tier1_type_value = tier1_label_detected
        tier1_trigger_value = "Detected" if tier1_flag else ""
        if eight_k_event and eight_k_event.is_catalyst:
            if eight_k_event.catalyst_type == "Tier-1":
                if eight_k_event.tier1_type:
                    tier1_type_value = eight_k_event.tier1_type
                if eight_k_event.tier1_trigger:
                    tier1_trigger_value = eight_k_event.tier1_trigger
            else:
                tier1_type_value = ""
                tier1_trigger_value = ""

        catalyst_primary_form = catalyst_summary.get("primary_form", "")
        catalyst_primary_date = catalyst_summary.get("primary_date", "")
        catalyst_primary_days = catalyst_summary.get("primary_days_ago", "")
        catalyst_primary_url = catalyst_summary.get("primary_url", "")

        catalyst_type_clean = _clean_text(catalyst_type).upper()
        if catalyst_type_clean not in {"TIER-1", "TIER-2"}:
            catalyst_type = "NONE"
            catalyst_summary["status"] = "Missing"
            checklist_statuses["Catalyst"] = "Missing"
            catalyst_field = ""
            catalyst_primary_form = ""
            catalyst_primary_date = ""
            catalyst_primary_days = ""
            catalyst_primary_url = ""
            catalyst_summary["primary_form"] = ""
            catalyst_summary["primary_date"] = ""
            catalyst_summary["primary_days_ago"] = ""
            catalyst_summary["primary_url"] = ""
            catalyst_summary["primary_text"] = ""
        else:
            catalyst_type = "Tier-1" if catalyst_type_clean == "TIER-1" else "Tier-2"
            if catalyst_summary.get("status") == "Pass" and eight_k_event and eight_k_event.is_catalyst:
                catalyst_primary_form = "8-K"

        passed_count = sum(1 for status in checklist_statuses.values() if status == "Pass")
        total_checks = len(checklist_statuses)
        checklist_passed_value = f"{passed_count}/{total_checks}"

        mandatory_pass = (
            checklist_statuses["Catalyst"] == "Pass"
            and checklist_statuses["Dilution"] == "Pass"
            and checklist_statuses["Runway"] == "Pass"
        )
        if not mandatory_pass:
            status_text = "TBD - exclude"

        dilution_primary_form = dilution_summary.get("primary_form", "")
        dilution_primary_date = dilution_summary.get("primary_date", "")
        dilution_primary_days = dilution_summary.get("primary_days_ago", "")
        dilution_primary_url = dilution_summary.get("primary_url", "")

        governance_primary_form = governance_summary.get("primary_form", "")
        governance_primary_date = governance_summary.get("primary_date", "")
        governance_primary_days = governance_summary.get("primary_days_ago", "")
        governance_primary_url = governance_summary.get("primary_url", "")

        validation_errors: list[str] = []
        if not _is_iso_date_string(catalyst_primary_date) or not _url_matches_form(
            catalyst_primary_form, catalyst_primary_url
        ):
            validation_errors.append("CatalystPrimary")

        if dilution_primary_form:
            if not _is_iso_date_string(dilution_primary_date):
                validation_errors.append("DilutionPrimaryDate")
            if not _url_matches_form(dilution_primary_form, dilution_primary_url):
                validation_errors.append("DilutionPrimaryURL")

        if invalid_governance_form or (
            governance_primary_form and governance_primary_form not in _GOVERNANCE_VALID_FORMS
        ):
            validation_errors.append("GovernancePrimaryForm")

        if dilution_flag and dilution_status not in {"Pass (Offering)", "Overhang (S-8)"}:
            validation_errors.append("DilutionStatus")

        if runway_quarters is None and (runway_status != "Missing" or runway_source_url):
            validation_errors.append("RunwayMissing")

        if validation_errors:
            _emit(
                "WARN",
                f"Dropped: Validation failed ({identifier}) {'; '.join(validation_errors)}",
                progress_fn,
            )
            continue

        record = {
            "Ticker": ticker_value,
            "Company": company_value,
            "Sector": sector_text,
            "CIK": cik_value,
            "Price": price_val,
            "MarketCap": market_cap_val,
            "ADV20": adv_val,
            "LatestForm": latest_form,
            "LatestFiledAt": latest_filed_at,
            "LatestFiledAgeDays": latest_filed_age_days,
            "CatalystType": catalyst_type,
            "Catalyst": catalyst_field,
            "CatalystStatus": catalyst_summary.get("status", ""),
            "CatalystPrimaryForm": catalyst_primary_form,
            "CatalystPrimaryDate": catalyst_primary_date,
            "CatalystPrimaryDaysAgo": catalyst_primary_days,
            "CatalystPrimaryURL": catalyst_primary_url,
            "RunwayQuartersRaw": runway_quarters_raw,
            "RunwayQuarters": runway_quarters,
            "RunwayCashRaw": runway_cash_raw,
            "RunwayCash": runway_cash,
            "RunwayQuarterlyBurnRaw": runway_burn_raw,
            "RunwayQuarterlyBurn": runway_burn,
            "RunwayEstimate": runway_estimate,
            "RunwayNotes": runway_notes,
            "RunwayDays": runway_days_value,
            "RunwayBucket": runway_bucket_label,
            "RunwaySourceForm": runway_source_form,
            "RunwaySourceDate": runway_source_date,
            "RunwaySourceDaysAgo": runway_source_days,
            "RunwaySourceURL": runway_source_url,
            "RunwayStatus": runway_status,
            "DilutionFlag": dilution_flag,
            "Dilution": dilution_field,
            "DilutionStatus": dilution_status,
            "DilutionFormsHit": "; ".join(dilution_forms_hit),
            "DilutionKeywordsHit": "; ".join(dilution_keywords_hit),
            "DilutionPrimaryForm": dilution_primary_form,
            "DilutionPrimaryDate": dilution_primary_date,
            "DilutionPrimaryDaysAgo": dilution_primary_days,
            "DilutionPrimaryURL": dilution_primary_url,
            "DilutionEvidenceAll": dilution_evidence,
            "Governance": governance_field,
            "GovernanceStatus": governance_summary.get("status", ""),
            "GovernancePrimaryForm": governance_primary_form,
            "GovernancePrimaryDate": governance_primary_date,
            "GovernancePrimaryDaysAgo": governance_primary_days,
            "GovernancePrimaryURL": governance_primary_url,
            "GovernanceEvidenceAll": governance_evidence,
            "GovernanceNotes": governance_notes,
            "HasGoingConcern": has_going_concern,
            "HasMaterialWeakness": has_material_weakness,
            "Auditor": auditor_text,
            "Insider": insider_text,
            "InsiderStatus": insider_status_label,
            "InsiderForms345Links": "; ".join(insider_links),
            "InsiderBuyCount": insider_buy_count,
            "HasInsiderCluster": has_insider_cluster,
            "Ownership": ownership_text,
            "OwnershipStatus": ownership_status_label,
            "OwnershipLinks": "; ".join(ownership_links),
            "Materiality": materiality_text,
            "SubscoresEvidenced": subscores_text,
            "SubscoresEvidencedCount": subscores_count,
            "Status": status_text,
            "ChecklistPassed": checklist_passed_value,
            "ChecklistCatalyst": checklist_statuses["Catalyst"],
            "ChecklistDilution": checklist_statuses["Dilution"],
            "ChecklistRunway": checklist_statuses["Runway"],
            "ChecklistGovernance": checklist_statuses["Governance"],
            "ChecklistInsider": checklist_statuses["Insider"],
            "ChecklistOwnership": checklist_statuses["Ownership"],
            "RiskFlag": risk_flag,
            "Tier1Type": tier1_type_value,
            "Tier1Trigger": tier1_trigger_value,
        }

        survivors.append(record)

    output_path = csv_path(data_dir, "validated_watchlist")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_OUTPUT_COLUMNS)
        for record in survivors:
            writer.writerow([_csv_cell_value(record.get(column)) for column in _OUTPUT_COLUMNS])

    rows_written = len(survivors)

    if status == "stopped":
        _emit(
            "INFO",
            f"build_watchlist: stop requested; wrote {csv_filename('validated_watchlist')} with {rows_written} rows",
            progress_fn,
        )
    else:
        _emit(
            "OK",
            f"build_watchlist: wrote {csv_filename('validated_watchlist')} with {rows_written} rows",
            progress_fn,
        )

    return rows_written, status
