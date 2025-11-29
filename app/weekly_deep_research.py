from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd

from app.config import load_config
from app.edgar_adapter import get_adapter
from app.utils import ensure_csv
from edgar_core.runway import extract_runway_from_filing

DILUTION_FORMS = {"S-3", "S-8", "424B", "424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "424B8"}
RUNWAY_FORMS = ("10-Q", "10-K", "20-F", "6-K", "40-F")

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


def _extract_numeric(text: str) -> float | None:
    cleaned = text.replace(",", "")
    cleaned = cleaned.replace("(", "-").replace(")", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def compute_runway_from_html(html_text: str) -> float | None:
    if not html_text:
        return None
    cash_match = re.search(r"cash and cash equivalents[:\s]*\$?([\d,\.\(\)-]+)", html_text, re.IGNORECASE)
    burn_match = re.search(r"operating activities[:\s]*\$?([\d,\.\(\)-]+)", html_text, re.IGNORECASE)
    if not cash_match or not burn_match:
        return None
    cash_val = _extract_numeric(cash_match.group(1))
    burn_val = _extract_numeric(burn_match.group(1))
    if cash_val is None or burn_val is None or burn_val == 0:
        return None
    quarterly_burn = abs(burn_val)
    return round(cash_val / quarterly_burn, 2)


def compute_runway_quarters(url: str, adapter=None) -> tuple[float | None, bool]:
    """Return (runway_quarters, used_primary_parser)."""

    if not url:
        return None, False

    adapter = adapter or get_adapter()

    try:
        filing = adapter._resolve_filing(url)  # type: ignore[attr-defined]
    except Exception:
        filing = None
        logger.debug("weekly_w3: _resolve_filing failed", exc_info=True)

    if filing is not None:
        try:
            result = extract_runway_from_filing(filing)
            quarters = result.get("runway_quarters")
            if quarters is not None and quarters > 0:
                return round(float(quarters), 2), True
        except Exception:
            logger.debug("weekly_w3: extract_runway_from_filing failed", exc_info=True)

    # fallback to HTML regex parsing
    fallback = _runway_from_filing(str(url), adapter=adapter)
    return fallback, False


def _runway_from_filing(url: str, adapter=None) -> float | None:
    if not url:
        return None
    path = url
    if url.startswith("file://"):
        path = url.replace("file://", "")
    candidate = Path(path)
    if candidate.exists():
        html_text = candidate.read_text(encoding="utf-8", errors="ignore")
        return compute_runway_from_html(html_text)
    if str(url).startswith("http"):
        try:
            edgar_adapter = adapter or get_adapter()
            html_text = edgar_adapter.download_filing_text(str(url))
            if html_text:
                return compute_runway_from_html(html_text)
        except Exception:
            return None
    return None


def _dilution_details(filings: pd.DataFrame, form_col: str) -> tuple[str, str, str | None]:
    forms = filings.get(form_col, pd.Series(dtype=str)).astype(str).tolist()
    normalized = {_normalize_form(f) for f in forms if f}
    evidence = []
    last_date = None
    for _, record in filings.iterrows():
        form = _normalize_form(record.get(form_col))
        if not form:
            continue
        if form in DILUTION_FORMS or any(form.startswith(prefix) for prefix in DILUTION_FORMS):
            url = record.get("FilingURL") or record.get("URL") or ""
            if url:
                evidence.append(str(url))
            date_val = record.get("FilingDate") or record.get("Date")
            if pd.notna(date_val):
                last_date = date_val
    if any(form in DILUTION_FORMS or form.startswith(tuple(DILUTION_FORMS)) for form in normalized):
        score = "High"
    elif normalized:
        score = "Low"
    else:
        score = "TBD"
    return score, _aggregate_evidence(evidence), last_date


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


def _biotech_peer_flag(sector: str, industry: str) -> str:
    combined = f"{sector} {industry}".lower()
    return "Y" if "biotech" in combined or "biotechnology" in combined else "N"


def _materiality(subscore_count: int, catalyst: str) -> str:
    if subscore_count >= 4 and catalyst != "None":
        return "High"
    if subscore_count >= 3:
        return "Medium"
    if subscore_count == 0:
        return "N/A"
    return "Low"


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


def _conviction_from_subscores(subscore_count: int, mandatory_ok: bool) -> str:
    if not mandatory_ok or subscore_count < 4:
        return ""
    if subscore_count >= 5:
        return "High"
    return "Medium"


def _materiality_label(raw: str) -> str:
    if not raw or raw == "N/A":
        return "N/A"
    lowered = raw.lower()
    if lowered in {"high", "medium"}:
        return f"pass ({raw})"
    if lowered in {"low", "fail"}:
        return f"fail ({raw})"
    return raw


def _status_from_row(
    mandatory_ok: bool,
    subscore_count: int,
    materiality: str,
    biotech_peer: str,
) -> str:
    materiality_lower = materiality.lower()
    materiality_ok = materiality_lower.startswith("pass") or materiality_lower.startswith("n/a")
    biotech_ok = not biotech_peer.startswith("TBD")
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
    adapter = get_adapter(cfg)

    shortlist_path = os.path.join(data_dir, "20_candidate_shortlist.csv")
    filings_path = os.path.join(data_dir, "02_filings.csv")
    events_path = os.path.join(data_dir, "09_events.csv")

    shortlist = _load_csv(shortlist_path)
    filings = _load_csv(filings_path)
    events = _load_csv(events_path)

    required_shortlist = ["Ticker", "Company", "CIK"]
    for col in required_shortlist:
        if col not in shortlist.columns:
            raise RuntimeError(f"{shortlist_path} missing required column {col}")

    output_rows: List[dict] = []

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
            relevant_mask = candidate_filings.get(form_col, pd.Series(dtype=str)).astype(str).str.upper().str.startswith(RUNWAY_FORMS)
            relevant = candidate_filings[relevant_mask]
            if not relevant.empty:
                if "FilingDate" in relevant.columns:
                    relevant = relevant.sort_values(by="FilingDate", ascending=False, na_position="last")
                runway_link = relevant.iloc[0].get("FilingURL") or relevant.iloc[0].get("URL", "")
                runway_quarters, used_primary = compute_runway_quarters(str(runway_link), adapter=adapter)
                if runway_quarters is None:
                    logger.info(
                        "weekly_w3: runway parse failed for %s (%s), leaving Runway (qtrs)=TBD",
                        ticker,
                        cik,
                    )
                elif not used_primary:
                    logger.debug("weekly_w3: runway regex fallback used for %s", ticker)
                runway_evidence.append(str(runway_link))

        dilution, dilution_evidence, last_dilution_date = _dilution_details(candidate_filings, form_col)
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
        biotech_flag = _biotech_peer_flag(str(sector), str(industry))
        biotech_peer_field = "N"
        biotech_peer_evidence = ""
        if biotech_flag == "Y":
            biotech_peer_evidence = "Peer: stub"
            biotech_peer_field = f"Y:{biotech_peer_evidence}" if biotech_peer_evidence else "TBD"

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

        subscore_count = sum(1 for v in subscore_flags.values() if v)
        materiality_raw = _materiality(subscore_count, catalyst_label)
        materiality_label = _materiality_label(materiality_raw)

        mandatory_ok = all(subscore_flags.get(key, False) for key in ["dilution", "runway", "catalyst"])
        conviction = _conviction_from_subscores(subscore_count, mandatory_ok)
        biotech_field = biotech_peer_field if biotech_flag == "Y" else "N"
        status = _status_from_row(mandatory_ok, subscore_count, materiality_label, biotech_field)

        evidence_primary = _aggregate_evidence(primary_links)
        evidence_secondary = _aggregate_evidence(secondary_links)

        price = getattr(row, "Price", None)
        if price is None or (isinstance(price, (float, int)) and pd.isna(price)):
            price = getattr(row, "Close", None)

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
                "Runway (qtrs)": runway_quarters if runway_quarters is not None else "TBD",
                "DilutionScore": dilution,
                "Dilution": dilution_label,
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
                "EvidenceSecondary": evidence_secondary,
                "PrimaryCatalystDate": catalyst_date,
                "PrimaryCatalystType": catalyst_type,
                "PrimaryCatalystURL": catalyst_url,
                "LastDilutionEventDate": last_dilution_date,
                "LastInsiderBuyDate": last_insider_date,
                "GoingConcernFlag": going_concern,
                "BiotechPeerRead": biotech_flag,
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


__all__ = ["compute_runway_from_html", "compute_runway_quarters", "run_weekly_deep_research"]
