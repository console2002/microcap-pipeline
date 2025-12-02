from __future__ import annotations

import csv
import logging
import os
import re
from typing import Callable, Iterable, List, Optional

import pandas as pd

from edgar import Filing

from app.biotech_utils import (
    classify_peer_events,
    get_biotech_peers,
    is_biotech,
    PeerEventClassification,
)
from app.config import load_config
from app.edgar_adapter import get_adapter
from app.runway_utils import compute_runway_from_html, compute_runway_quarters
from app.settings import BIOTECH_PEER_REQUIRED_FOR_VALIDATION
from app.utils import ensure_csv
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
RUNWAY_FORMS = ("10-Q", "10-K", "20-F", "6-K", "40-F")
PEER_EVENTS_LOOKBACK_DAYS = 180
DILUTION_CREATION = "OVERHANG_CREATION"
DILUTION_TERMINATION = "OVERHANG_TERMINATION"
DILUTION_UNKNOWN = "UNKNOWN"

logger = logging.getLogger(__name__)


def _normalize_form(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).strip().upper()


def _load_csv(path: str, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8")
    if required:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")
    return df


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()

    df = events.copy()
    df["event_date_canonical"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    date_cols = [
        "EventDate",
        "event_date",
        "FilingDate",
        "Date",
    ]
    for col in date_cols:
        if col in df.columns:
            df["event_date_canonical"] = df["event_date_canonical"].fillna(
                pd.to_datetime(df[col], errors="coerce")
            )
    df["event_date_canonical"] = df["event_date_canonical"].dt.tz_localize(None)
    return df


def _strip_html_tags(text: str) -> str:
    if not text:
        return ""
    # Cheap HTML tag stripper to keep dependencies minimal; sufficient for
    # keyword scanning of offering/termination language.
    return re.sub(r"<[^>]+>", " ", text)


def _extract_filing_text(filing: Filing, max_chars: int = 40_000) -> str:
    """Return a bounded blob of filing text for keyword heuristics.

    We favor ``Filing.text()`` and fall back to ``html()`` if necessary.
    Select exhibits (EX-1.*, EX-10.*) are appended to capture sales
    agreements or terminations that sometimes live in exhibits. The body is
    truncated first to preserve budget for exhibits so they are always
    scanned.
    """

    text_blob = ""
    try:
        text_blob = filing.text() or ""
    except Exception:
        logger.debug("filing.text() failed for dilution scan", exc_info=True)

    if not text_blob:
        try:
            text_blob = _strip_html_tags(filing.html() or "")
        except Exception:
            logger.debug("filing.html() failed for dilution scan", exc_info=True)

    exhibit_texts: list[str] = []
    for exhibit in getattr(filing, "exhibits", []) or []:
        doc_type = str(getattr(exhibit, "document_type", "") or "").upper()
        if not doc_type.startswith(("EX-1", "EX-10")):
            continue
        try:
            exhibit_texts.append(exhibit.text() or "")
        except Exception:
            continue

    max_body_chars = int(max_chars * 0.7)
    body_part = (text_blob or "")[:max_body_chars]
    remaining = max(max_chars - len(body_part), 0)

    parts: list[str] = [body_part] if body_part else []
    for exhibit_text in exhibit_texts:
        if remaining <= 0:
            break
        snippet = exhibit_text[:remaining]
        if snippet:
            parts.append(snippet)
            remaining -= len(snippet)

    combined = "\n\n".join(part for part in parts if part)
    return combined[:max_chars]


def classify_dilution_filing(filing: Filing) -> str:
    """Classify a dilution-related filing as overhang creation vs termination.

    Heuristics (documented for future tuning):
    - Creation/increase: phrases like "at-the-market offering", "equity distribution
      agreement", "we may sell from time to time", "up to [N] shares", "offering".
    - Termination/exhaustion: "terminate", "has been terminated", "sales agreement
      terminated", "no further sales", "suspend our offering", "program has expired".
    """

    text_blob = _extract_filing_text(filing)
    if not text_blob:
        return DILUTION_UNKNOWN

    lower_text = text_blob.lower()

    creation_keywords = [
        "at-the-market offering",
        "at the market offering",
        "equity distribution agreement",
        "equity distribution",
        "we may sell from time to time",
        "we may offer and sell",
        "up to",
        "registered direct offering",
        "sell shares",
    ]
    # Only treat strong termination phrases as overhang removal; boilerplate "may be terminated"
    # is deliberately excluded so creation language still wins.
    termination_keywords = [
        "has been terminated",
        "is hereby terminated",
        "we have terminated",
        "terminated effective",
        "terminated the sales agreement",
        "sales agreement terminated",
        "termination notice",
        "no further sales",
        "suspend our offering",
        "program has expired",
    ]

    if any(keyword in lower_text for keyword in termination_keywords):
        return DILUTION_TERMINATION
    if any(keyword in lower_text for keyword in creation_keywords):
        return DILUTION_CREATION
    return DILUTION_UNKNOWN


def _biotech_peer_read_from_events(
    ticker: str,
    sector: str,
    industry: str,
    events: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[str, str]:
    """Compute BiotechPeerRead string and evidence."""

    if not is_biotech(sector, industry):
        return "N:NonBiotech", ""

    peers = get_biotech_peers(universe, ticker, sector, industry)
    if not peers:
        return "N:NoPeers", ""

    now = pd.Timestamp.utcnow().tz_localize(None)
    lookback_start = now.normalize() - pd.Timedelta(days=PEER_EVENTS_LOOKBACK_DAYS)
    peer_events = events[events.get("Ticker", pd.Series(dtype=str)).isin(peers)].copy()
    if not peer_events.empty:
        peer_events = peer_events[
            (peer_events["event_date_canonical"] >= lookback_start)
            & (peer_events["event_date_canonical"] <= now)
        ]

    classification: PeerEventClassification = classify_peer_events(peer_events)

    mapping = {
        "NONE": "N:NoPeerEvent",
        "POSITIVE": "Y_POS",
        "NEGATIVE": "Y_NEG",
        "MIXED": "Y_MIXED",
    }
    base = mapping.get(classification.code, "N:NoPeerEvent")
    evidence = classification.evidence
    if base.startswith("Y") and evidence:
        base = f"{base}:{evidence}"
    return base, evidence


def _event_sort_key(event: dict) -> pd.Timestamp:
    ts = event.get("filed_ts")
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts
    try:
        parsed = pd.to_datetime(event.get("filed_at_raw"), errors="coerce")
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed
    except Exception:
        pass
    return pd.Timestamp.min


def _resolve_filing_from_record(record: pd.Series) -> Optional[Filing]:
    adapter = get_adapter()
    candidates = [
        record.get("MasterTxtURL"),
        record.get("FilingURL"),
        record.get("URL"),
        record.get("Accession"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            filing = adapter._resolve_filing(candidate)
        except Exception:
            logger.debug("dilution: filing resolve failed for %s", candidate, exc_info=True)
            continue
        if filing is not None:
            return filing
    return None


def _dilution_details(filings: pd.DataFrame, form_col: str) -> dict:
    events: list[dict] = []

    for _, record in filings.iterrows():
        form = _normalize_form(record.get(form_col))
        if not form:
            continue
        if not (form in DILUTION_FORMS or any(form.startswith(prefix) for prefix in DILUTION_FORMS)):
            continue

        url = record.get("FilingURL") or record.get("URL") or record.get("MasterTxtURL") or ""
        filed_at_raw = record.get("FilingDate") or record.get("Date") or record.get("FiledAt")
        filed_ts = pd.to_datetime(filed_at_raw, errors="coerce") if filed_at_raw is not None else pd.NaT

        filing_obj = _resolve_filing_from_record(record)
        classification = DILUTION_UNKNOWN
        if filing_obj is not None:
            classification = classify_dilution_filing(filing_obj)

        events.append(
            {
                "form": form,
                "url": str(url) if url else "",
                "filed_at_raw": filed_at_raw,
                "filed_ts": filed_ts,
                "classification": classification,
            }
        )

    if not events:
        return {
            "score": "TBD",
            "evidence": "",
            "last_event_date": None,
            "key_filing_url": None,
            "overhang_removed": None,
        }

    events_sorted = sorted(events, key=_event_sort_key, reverse=True)

    def _overhang_from_event(event: dict) -> bool:
        classification = event.get("classification")
        if classification == DILUTION_TERMINATION:
            return False
        if classification == DILUTION_CREATION:
            return True
        # Default to caution: unknown dilution filings usually introduce overhang.
        return True

    driver_event = events_sorted[0]
    overhang_present = _overhang_from_event(driver_event)

    evidence_candidates = [driver_event.get("url", ""), *(ev.get("url", "") for ev in events_sorted)]
    evidence = _aggregate_evidence(evidence_candidates)
    last_date = driver_event.get("filed_at_raw")
    if not last_date:
        ts_val = driver_event.get("filed_ts")
        if isinstance(ts_val, pd.Timestamp) and not pd.isna(ts_val):
            last_date = ts_val.date().isoformat()
    score = "High" if overhang_present else "Low"

    return {
        "score": score,
        "evidence": evidence,
        "last_event_date": last_date,
        "key_filing_url": driver_event.get("url", "") or None,
        "overhang_removed": not overhang_present,
    }


def _catalyst_details(events: pd.DataFrame) -> tuple[str, str | None, str | None, str | None]:
    if events is None or events.empty:
        return "None", None, None, None
    events = events.copy()
    events["Tier"] = events.get("Tier", events.get("event_tier", pd.Series(dtype=str))).astype(str)
    tier1 = events[events["Tier"].str.contains("1", case=False, na=False)]
    target = tier1 if not tier1.empty else events
    sort_cols = [col for col in ["EventDate", "event_date", "FilingDate"] if col in target.columns]
    if sort_cols:
        target = target.sort_values(by=sort_cols, ascending=True, na_position="last")
    row = target.iloc[0]
    score = "Tier-1" if not tier1.empty else "Tier-2"
    event_date = row.get("EventDate") or row.get("event_date") or row.get("FilingDate")
    event_type = (
        row.get("event_type")
        or row.get("EventType")
        or row.get("ItemsNormalized")
        or row.get("ItemsPresent")
    )
    url = row.get("PrimarySource") or row.get("primary_source_url") or row.get("FilingURL") or row.get("URL")
    return score, event_date, event_type, url


def _governance_details(filings: pd.DataFrame, form_col: str) -> tuple[str, str, str]:
    normalized = {_normalize_form(f) for f in filings.get(form_col, pd.Series(dtype=str)).astype(str) if f}
    evidence = []
    going_concern = "N"
    if not filings.empty:
        gov_mask = filings[form_col].astype(str).str.upper().str.contains("DEF 14A|10-K|10-Q|20-F|40-F", regex=True)
        gov_records = filings[gov_mask]
        for _, rec in gov_records.iterrows():
            url = rec.get("FilingURL") or rec.get("URL")
            if url:
                evidence.append(str(url))
            text = str(rec.get("FilingText", ""))
            if re.search(r"going concern", text, re.IGNORECASE):
                going_concern = "Y"
    if any("DEF 14A" in form for form in normalized):
        score = "OK"
    elif any(form.startswith(prefix) for form in normalized for prefix in RUNWAY_FORMS):
        score = "OK"
    elif normalized:
        score = "Concern"
    else:
        score = "TBD"
    return score, going_concern, _aggregate_evidence(evidence)


def _insider_details(filings: pd.DataFrame, form_col: str) -> tuple[str, str | None, str]:
    forms = filings.get(form_col, pd.Series(dtype=str)).astype(str)
    evidence = []
    dates = []
    score = "TBD"
    for _, rec in filings.iterrows():
        form = _normalize_form(rec.get(form_col))
        if form in {"3", "4", "5"} or form.startswith("4"):
            url = rec.get("FilingURL") or rec.get("URL")
            if url:
                evidence.append(str(url))
            date_val = rec.get("FilingDate") or rec.get("Date")
            if pd.notna(date_val):
                dates.append(date_val)
            score = "Strong" if form.startswith("4") else "Weak"
    last_date = max(dates) if dates else None
    return score, last_date, _aggregate_evidence(evidence)


def _materiality(subscore_count: int, catalyst: str, mandatory_ok: bool) -> str:
    """Return a PASS/FAIL Materiality string.

    Materiality must always start with PASS or FAIL for downstream validation.
    """

    catalyst_text = str(catalyst or "").strip()
    catalyst_lower = catalyst_text.lower()

    if not mandatory_ok:
        return "FAIL - mandatory subscore missing"

    strong_catalyst = catalyst_lower.startswith("tier-1")
    has_catalyst = catalyst_lower not in {"", "none", "tbd", "nan"}

    if strong_catalyst and subscore_count >= 4:
        return "PASS - Tier1 catalyst"
    if has_catalyst and subscore_count >= 4:
        tier2_flag = catalyst_lower.startswith("tier-2")
        return "PASS - Tier2" if tier2_flag else "PASS - catalyst"

    return "FAIL - weak profile"


def _materiality_passed(materiality: str) -> bool:
    return isinstance(materiality, str) and materiality.strip().upper().startswith("PASS")


def _aggregate_evidence(primary_links: list[str]) -> str:
    clean = [link for link in primary_links if link]
    deduped = list(dict.fromkeys(clean))
    return ";".join(deduped)


def _classify_evidence(links: list[str]) -> tuple[list[str], list[str]]:
    """Split evidence links into primary (SEC/registry) vs secondary."""

    primary: list[str] = []
    secondary: list[str] = []
    for link in links:
        if not link:
            continue
        normalized = str(link).strip()
        lower = normalized.lower()
        if lower.startswith("file://") or "sec.gov" in lower or "edgar" in lower:
            primary.append(normalized)
        else:
            secondary.append(normalized)
    return list(dict.fromkeys(primary)), list(dict.fromkeys(secondary))


def _conviction_from_subscores(subscore_count: int, catalyst: str, materiality: str) -> str:
    if not _materiality_passed(materiality):
        return "Low"

    catalyst_lower = str(catalyst or "").lower()
    strong_catalyst = catalyst_lower.startswith("tier-1")

    if subscore_count >= 4 and strong_catalyst:
        return "High"
    if subscore_count >= 4:
        return "Medium"
    return "Low"


def _status_from_row(
    mandatory_ok: bool,
    subscore_count: int,
    materiality: str,
    biotech_peer: str,
    is_biotech_candidate: bool,
    require_biotech_peer: bool,
) -> str:
    materiality_lower = materiality.lower()
    materiality_ok = materiality_lower.startswith("pass")
    biotech_ok = True
    if require_biotech_peer and is_biotech_candidate:
        biotech_ok = isinstance(biotech_peer, str) and biotech_peer.startswith("Y_")
    if mandatory_ok and subscore_count >= 4 and materiality_ok and biotech_ok:
        return "Validated"
    return "TBD — exclude"


def _progress_emit(progress_fn: Callable[[str], None] | None, status: str, message: str) -> None:
    if progress_fn is None:
        return
    try:
        progress_fn(f"dr_forms [{status}] {message}")
    except Exception:
        pass


def _emit_form_fetch(
    progress_fn: Callable[[str], None] | None,
    ticker: str,
    form: str,
    filed_at: str | None,
    url: str | None,
) -> None:
    parts = [ticker, "fetching", form]
    if filed_at:
        parts.append(f"filed {filed_at}")
    if url:
        parts.append(f"url {url}")
    _progress_emit(progress_fn, "INFO", " ".join(parts))


def _emit_form_status(
    progress_fn: Callable[[str], None] | None,
    ticker: str,
    form: str,
    status: str,
) -> None:
    _progress_emit(progress_fn, "OK", f"{ticker} {form} form status {status}")


def _emit_form_incomplete(
    progress_fn: Callable[[str], None] | None,
    ticker: str,
    form: str,
    reason: str,
) -> None:
    _progress_emit(progress_fn, "WARN", f"{ticker} {form} incomplete: {reason}")


def _iter_filings_for_forms(
    filings: pd.DataFrame,
    form_col: str,
    forms: set[str],
) -> Iterable[dict]:
    def _first_non_missing(record: pd.Series, keys: tuple[str, ...]) -> str:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            if pd.isna(value):
                continue
            return value
        return ""

    for _, record in filings.iterrows():
        form = _normalize_form(record.get(form_col))
        if not form:
            continue
        compact = form.replace(" ", "")
        if any(
            form == target
            or form.startswith(target)
            or compact == target.replace(" ", "")
            or compact.startswith(target.replace(" ", ""))
            for target in forms
        ):
            filed_at = _first_non_missing(record, ("FilingDate", "Date"))
            url = _first_non_missing(record, ("FilingURL", "URL"))

            yield {
                "form": form,
                "filed_at": filed_at,
                "url": url,
            }


def run_weekly_deep_research(
    data_dir: str | None = None, progress_fn: Callable[[str], None] | None = None
) -> pd.DataFrame:
    cfg = load_config()
    data_dir = data_dir or cfg.get("Paths", {}).get("data", "data")
    shortlist_path = os.path.join(data_dir, "20_candidate_shortlist.csv")
    filings_path = os.path.join(data_dir, "02_filings.csv")
    events_path = os.path.join(data_dir, "09_events.csv")
    universe_path = os.path.join(data_dir, "01_universe_gated.csv")

    shortlist = _load_csv(shortlist_path)
    filings = _load_csv(filings_path)
    events = _prepare_events(_load_csv(events_path))
    universe = _load_csv(universe_path)

    required_shortlist = ["Ticker", "Company", "CIK"]
    for col in required_shortlist:
        if col not in shortlist.columns:
            raise RuntimeError(f"{shortlist_path} missing required column {col}")

    output_rows: List[dict] = []
    runway_numeric_count = 0
    runway_evidence_only: list[str] = []

    def _select_runway_details(candidate_filings: pd.DataFrame, form_col: str):
        if candidate_filings.empty:
            return None, "", False

        filings_with_form = candidate_filings.copy()
        filings_with_form[form_col] = filings_with_form.get(form_col, pd.Series(dtype=str)).astype(str)
        mask = filings_with_form[form_col].str.upper().str.startswith(RUNWAY_FORMS)
        subset = filings_with_form[mask]
        if subset.empty:
            return None, "", False

        subset = subset.copy()
        subset["RunwayQuarters"] = pd.to_numeric(subset.get("RunwayQuarters"), errors="coerce")

        def _sort_key(rec: pd.Series):
            age_val = rec.get("Age")
            if pd.notna(age_val):
                return age_val
            filed_at = rec.get("FiledAt") or rec.get("FilingDate") or rec.get("Date")
            try:
                return (pd.Timestamp.utcnow() - pd.to_datetime(filed_at)).days
            except Exception:
                return 10**6

        subset["_runway_sort"] = subset.apply(_sort_key, axis=1)
        subset = subset.sort_values(by="_runway_sort", ascending=True, na_position="last")
        with_numeric = subset[subset["RunwayQuarters"].notna()]
        target = with_numeric if not with_numeric.empty else subset
        chosen = target.iloc[0]

        quarters = chosen.get("RunwayQuarters")
        try:
            quarters = float(quarters) if pd.notna(quarters) else None
        except Exception:
            quarters = None

        evidence_url = (
            chosen.get("RunwaySourceURL")
            or chosen.get("FilingURL")
            or chosen.get("URL")
            or ""
        )

        if quarters is None and evidence_url:
            adapter = get_adapter()
            quarters, _ = compute_runway_quarters(str(evidence_url), adapter=adapter)

        return quarters, evidence_url, pd.notna(quarters)

    for row in shortlist.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        cik = getattr(row, "CIK")
        sector = getattr(row, "Sector", "") if hasattr(row, "Sector") else ""
        industry = getattr(row, "Industry", "") if hasattr(row, "Industry") else ""
        candidate_filings = filings[(filings.get("Ticker", "").astype(str) == str(ticker)) | (filings.get("CIK", "").astype(str) == str(cik))]
        form_col = "FormType" if "FormType" in candidate_filings.columns else "Form"

        runway_link = ""
        runway_quarters = None
        runway_evidence: list[str] = []
        if not candidate_filings.empty:
            runway_quarters, runway_link, _ = _select_runway_details(
                candidate_filings, form_col
            )
            if runway_link:
                runway_evidence.append(str(runway_link))
            if runway_quarters is None and runway_link:
                runway_evidence_only.append(str(ticker))
                logger.info(
                    "weekly_w3: runway missing numeric value for %s (%s) despite evidence", ticker, cik
                )
            elif runway_quarters is not None:
                runway_numeric_count += 1

        dilution_details = _dilution_details(candidate_filings, form_col)
        dilution = dilution_details.get("score", "TBD")
        dilution_evidence = dilution_details.get("evidence", "")
        last_dilution_date = dilution_details.get("last_event_date")
        dilution_key_url = dilution_details.get("key_filing_url")
        dilution_forms = list(
            _iter_filings_for_forms(candidate_filings, form_col, set(DILUTION_FORMS))
        )
        for entry in dilution_forms:
            _emit_form_fetch(progress_fn, str(ticker), entry["form"], entry["filed_at"], entry["url"])
            if entry["url"]:
                _emit_form_status(progress_fn, str(ticker), entry["form"], "OK dilution evidence captured")
            else:
                _emit_form_incomplete(progress_fn, str(ticker), entry["form"], "missing filing URL")
        ticker_series = events.get("Ticker", pd.Series(dtype=str))
        cik_series = events.get("CIK", pd.Series(dtype=str))
        candidate_events = events[(ticker_series.astype(str) == str(ticker)) | (cik_series.astype(str) == str(cik))]
        catalyst, catalyst_date, catalyst_type, catalyst_url = _catalyst_details(candidate_events)
        governance, going_concern, governance_evidence = _governance_details(candidate_filings, form_col)
        governance_forms = list(
            _iter_filings_for_forms(
                candidate_filings,
                form_col,
                {"DEF 14A", "DEF14A", "DEFA14", "DEFM14", "DEFC14"},
            )
        )
        for entry in governance_forms:
            _emit_form_fetch(progress_fn, str(ticker), entry["form"], entry["filed_at"], entry["url"])
            if entry["url"]:
                status_text = "OK governance evidence captured"
                _emit_form_status(progress_fn, str(ticker), entry["form"], status_text)
            else:
                _emit_form_incomplete(progress_fn, str(ticker), entry["form"], "missing filing URL")

        insider, last_insider_date, insider_evidence = _insider_details(candidate_filings, form_col)
        insider_forms = list(
            _iter_filings_for_forms(candidate_filings, form_col, {"3", "4", "5"})
        )
        for entry in insider_forms:
            _emit_form_fetch(progress_fn, str(ticker), entry["form"], entry["filed_at"], entry["url"])
            if entry["url"]:
                strength = "Strong" if entry["form"].startswith("4") else "Weak"
                _emit_form_status(
                    progress_fn,
                    str(ticker),
                    entry["form"],
                    f"OK insider evidence captured ({strength})",
                )
            else:
                _emit_form_incomplete(progress_fn, str(ticker), entry["form"], "missing filing URL")
        biotech_peer_field, biotech_peer_evidence = _biotech_peer_read_from_events(
            str(ticker), str(sector), str(industry), events, universe
        )
        biotech_flag = "Y" if is_biotech(sector, industry) else "N"

        dilution_label = dilution if dilution in {"High", "Low"} else "TBD"
        catalyst_label = catalyst
        if catalyst_type:
            catalyst_label = f"{catalyst}: {catalyst_type}"

        governance_label = governance or "TBD"
        insider_label = insider or "TBD"

        evidence_map = {
            "runway": runway_evidence,
            "dilution": dilution_evidence.split(";") if dilution_evidence else [],
            "catalyst": [catalyst_url] if catalyst_url else [],
            "governance": governance_evidence.split(";") if governance_evidence else [],
            "insider": insider_evidence.split(";") if insider_evidence else [],
        }

        primary_links: list[str] = []
        secondary_links: list[str] = []
        subscore_flags = {}
        for key, links in evidence_map.items():
            prim, sec = _classify_evidence(links)
            primary_links.extend(prim)
            secondary_links.extend(sec)
            value_for_key = {
                "runway": runway_quarters,
                "dilution": dilution_label,
                "catalyst": catalyst_label,
                "governance": governance_label,
                "insider": insider_label,
            }.get(key)
            valid_value = value_for_key not in {"", None, "TBD", "Unknown"}
            subscore_flags[key] = valid_value and bool(prim)

        mandatory_ok = all(subscore_flags.get(key, False) for key in ["dilution", "runway", "catalyst"])
        subscore_count = sum(1 for v in subscore_flags.values() if v)
        materiality_raw = _materiality(subscore_count, catalyst_label, mandatory_ok)
        materiality_label = materiality_raw

        conviction = _conviction_from_subscores(subscore_count, catalyst_label, materiality_label)
        biotech_field = biotech_peer_field if biotech_flag == "Y" else "N:NonBiotech"
        status = _status_from_row(
            mandatory_ok,
            subscore_count,
            materiality_label,
            biotech_field,
            biotech_flag == "Y",
            BIOTECH_PEER_REQUIRED_FOR_VALIDATION,
        )
        if (
            BIOTECH_PEER_REQUIRED_FOR_VALIDATION
            and biotech_flag == "Y"
            and not biotech_field.startswith("Y_")
            and status != "Validated"
        ):
            logger.info(
                "[biotech_peer] excluded_from_validation: ticker=%s, reason=NoPeerEvidence, BiotechPeerRead=%s",
                ticker,
                biotech_field,
            )

        evidence_primary = _aggregate_evidence(primary_links)
        # Secondary evidence placeholder; currently unused but kept for schema stability.
        evidence_secondary = _aggregate_evidence(secondary_links) or ""

        price = getattr(row, "Price", None)
        if price is None or (isinstance(price, (float, int)) and pd.isna(price)):
            price = getattr(row, "Close", None)

        runway_display = str(runway_quarters) if runway_quarters is not None else "TBD"

        output_rows.append(
            {
                "Ticker": ticker,
                "Company": getattr(row, "Company"),
                "CIK": cik,
                "Sector": sector,
                "Industry": industry,
                "Price": price,
                "MarketCap": getattr(row, "MarketCap", None),
                "ADV20": getattr(row, "ADV20", None),
                "RunwayQuarters": runway_quarters,
                "Runway (qtrs)": runway_display,
                "DilutionScore": dilution,
                "Dilution": dilution_label,
                "DilutionKeyFilingURL": dilution_key_url,
                "CatalystScore": catalyst,
                "Catalyst": catalyst_label,
                "GovernanceScore": governance,
                "Governance": governance_label,
                "InsiderScore": insider,
                "Insider": insider_label,
                "RunwayEvidencePrimary": _aggregate_evidence(runway_evidence),
                "DilutionEvidencePrimary": dilution_evidence,
                "CatalystEvidencePrimary": _aggregate_evidence([catalyst_url] if catalyst_url else []),
                "GovernanceEvidencePrimary": governance_evidence,
                "InsiderEvidencePrimary": insider_evidence,
                "Evidence (Primary links)": evidence_primary,
                "Evidence (Secondary links)": evidence_secondary,
                "EvidencePrimary": evidence_primary,
                # Reserved for future secondary evidence inputs; normalized to empty string.
                "EvidenceSecondary": evidence_secondary,
                "PrimaryCatalystDate": catalyst_date,
                "PrimaryCatalystType": catalyst_type,
                "PrimaryCatalystURL": catalyst_url,
                "LastDilutionEventDate": last_dilution_date,
                "LastInsiderBuyDate": last_insider_date,
                "GoingConcernFlag": going_concern,
                "BiotechPeerRead": biotech_field,
                "Biotech Peer Read-Through (Y/N + link)": biotech_field,
                "BiotechPeerEvidence": biotech_peer_evidence,
                "SubscoresEvidencedCount": subscore_count,
                "Subscores Evidenced (x/5)": subscore_count,
                "Materiality": materiality_raw,
                "Materiality (pass/fail + note)": materiality_label,
                "ConvictionScore": conviction,
                "Status": status,
            }
        )

    runway_missing_with_evidence = len(runway_evidence_only)
    total_candidates = len(output_rows)
    with_quarters = sum(1 for row in output_rows if row.get("RunwayQuarters") is not None and pd.notna(row.get("RunwayQuarters")))
    missing_quarters = total_candidates - with_quarters

    logger.info(
        "weekly_w3: runway summary numeric=%s evidence_only=%s",
        runway_numeric_count,
        runway_missing_with_evidence,
    )
    logger.info(
        "WEEKLY_RUNWAY: total=%s with_quarters=%s missing=%s",
        total_candidates,
        with_quarters,
        missing_quarters,
    )
    if runway_evidence_only:
        logger.debug(
            "weekly_w3: runway evidence without numeric for %s",
            ", ".join(runway_evidence_only[:5]),
        )

    output_path = os.path.join(data_dir, "30_deep_research.csv")
    default_fields = [
        "Ticker",
        "Company",
        "CIK",
        "Sector",
        "Industry",
        "Price",
        "MarketCap",
        "ADV20",
        "RunwayQuarters",
        "Runway (qtrs)",
        "DilutionScore",
        "Dilution",
        "DilutionKeyFilingURL",
        "CatalystScore",
        "Catalyst",
        "GovernanceScore",
        "Governance",
        "InsiderScore",
        "Insider",
        "RunwayEvidencePrimary",
        "DilutionEvidencePrimary",
        "CatalystEvidencePrimary",
        "GovernanceEvidencePrimary",
        "InsiderEvidencePrimary",
        "Evidence (Primary links)",
        "Evidence (Secondary links)",
        "EvidencePrimary",
        "EvidenceSecondary",
        "PrimaryCatalystDate",
        "PrimaryCatalystType",
        "PrimaryCatalystURL",
        "LastDilutionEventDate",
        "LastInsiderBuyDate",
        "GoingConcernFlag",
        "BiotechPeerRead",
        "Biotech Peer Read-Through (Y/N + link)",
        "BiotechPeerEvidence",
        "SubscoresEvidencedCount",
        "Subscores Evidenced (x/5)",
        "Materiality",
        "Materiality (pass/fail + note)",
        "ConvictionScore",
        "Status",
    ]
    fieldnames = list(output_rows[0].keys()) if output_rows else default_fields
    ensure_csv(output_path, fieldnames)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    return pd.DataFrame(output_rows)


__all__ = [
    "compute_runway_from_html",
    "compute_runway_quarters",
    "run_weekly_deep_research",
    "_materiality",
    "_materiality_passed",
    "_conviction_from_subscores",
]
