from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from app.config import load_config
from app.edgar_adapter import get_adapter
from app.utils import ensure_csv

DILUTION_FORMS = {"S-3", "S-8", "424B", "424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "424B8"}
RUNWAY_FORMS = ("10-Q", "10-K", "20-F", "6-K", "40-F")


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
        score = "Unknown"
    return score, _aggregate_evidence(evidence), last_date


def _catalyst_details(events: pd.DataFrame) -> tuple[str, str | None, str | None, str | None]:
    if events is None or events.empty:
        return "None", None, None, None
    events = events.copy()
    events["Tier"] = events.get("Tier", pd.Series(dtype=str)).astype(str)
    tier1 = events[events["Tier"].str.contains("1", case=False, na=False)]
    target = tier1 if not tier1.empty else events
    sort_cols = [col for col in ["EventDate", "FilingDate"] if col in target.columns]
    if sort_cols:
        target = target.sort_values(by=sort_cols, ascending=True, na_position="last")
    row = target.iloc[0]
    score = "Tier-1" if not tier1.empty else "Tier-2"
    event_date = row.get("EventDate") or row.get("FilingDate")
    event_type = row.get("EventType") or row.get("ItemsNormalized") or row.get("ItemsPresent")
    url = row.get("FilingURL") or row.get("URL")
    return score, event_date, event_type, url


def _governance_details(filings: pd.DataFrame, form_col: str) -> tuple[str, str, str]:
    normalized = {_normalize_form(f) for f in filings.get(form_col, pd.Series(dtype=str)).astype(str) if f}
    evidence = []
    going_concern = "N"
    if not filings.empty:
        gov_mask = filings[form_col].astype(str).str.upper().str.contains("DEF 14A|10-K|20-F|40-F", regex=True)
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
        score = ""
    return score, going_concern, _aggregate_evidence(evidence)


def _insider_details(filings: pd.DataFrame, form_col: str) -> tuple[str, str | None, str]:
    forms = filings.get(form_col, pd.Series(dtype=str)).astype(str)
    evidence = []
    dates = []
    score = ""
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
    return "Low"


def _aggregate_evidence(primary_links: list[str]) -> str:
    clean = [link for link in primary_links if link]
    deduped = list(dict.fromkeys(clean))
    return ";".join(deduped)


def run_weekly_deep_research(data_dir: str | None = None) -> pd.DataFrame:
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
        filing_forms = candidate_filings.get(form_col, pd.Series(dtype=str))
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
                runway_quarters = _runway_from_filing(str(runway_link), adapter=adapter)
                runway_evidence.append(str(runway_link))
        dilution, dilution_evidence, last_dilution_date = _dilution_details(candidate_filings, form_col)
        ticker_series = events.get("Ticker", pd.Series(dtype=str))
        cik_series = events.get("CIK", pd.Series(dtype=str))
        candidate_events = events[(ticker_series.astype(str) == str(ticker)) | (cik_series.astype(str) == str(cik))]
        catalyst, catalyst_date, catalyst_type, catalyst_url = _catalyst_details(candidate_events)
        governance, going_concern, governance_evidence = _governance_details(candidate_filings, form_col)
        insider, last_insider_date, insider_evidence = _insider_details(candidate_filings, form_col)
        biotech_flag = _biotech_peer_flag(str(sector), str(industry))
        evidence_links: list[str] = []
        evidence_links.extend(runway_evidence)
        evidence_primary = _aggregate_evidence(evidence_links)

        subscores = {
            "runway": runway_quarters is not None,
            "dilution": dilution not in {"", "Unknown"} and bool(dilution_evidence),
            "catalyst": catalyst != "None" and bool(catalyst_url),
            "governance": bool(governance) and bool(governance_evidence),
            "insider": bool(insider) and bool(insider_evidence),
        }
        subscore_count = sum(1 for v in subscores.values())
        materiality = _materiality(subscore_count, catalyst)

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
                "DilutionScore": dilution,
                "CatalystScore": catalyst,
                "GovernanceScore": governance,
                "InsiderScore": insider,
                "RunwayEvidencePrimary": _aggregate_evidence(runway_evidence),
                "DilutionEvidencePrimary": dilution_evidence,
                "CatalystEvidencePrimary": _aggregate_evidence([catalyst_url] if catalyst_url else []),
                "GovernanceEvidencePrimary": governance_evidence,
                "InsiderEvidencePrimary": insider_evidence,
                "PrimaryCatalystDate": catalyst_date,
                "PrimaryCatalystType": catalyst_type,
                "PrimaryCatalystURL": catalyst_url,
                "LastDilutionEventDate": last_dilution_date,
                "LastInsiderBuyDate": last_insider_date,
                "GoingConcernFlag": going_concern,
                "BiotechPeerRead": biotech_flag,
                "BiotechPeerEvidence": "Peer: stub" if biotech_flag == "Y" else "",
                "SubscoresEvidencedCount": subscore_count,
                "Materiality": materiality,
                "ConvictionScore": None,
                "EvidencePrimary": evidence_primary,
                "EvidenceSecondary": "",
                "Status": "",
            }
        )

    output_path = os.path.join(data_dir, "30_deep_research.csv")
    if output_rows:
        fieldnames = list(output_rows[0].keys())
    else:
        fieldnames = [
            "Ticker",
            "Company",
            "CIK",
            "Sector",
            "Industry",
            "Price",
            "MarketCap",
            "ADV20",
            "RunwayQuarters",
            "DilutionScore",
            "CatalystScore",
            "GovernanceScore",
            "InsiderScore",
            "RunwayEvidencePrimary",
            "DilutionEvidencePrimary",
            "CatalystEvidencePrimary",
            "GovernanceEvidencePrimary",
            "InsiderEvidencePrimary",
            "PrimaryCatalystDate",
            "PrimaryCatalystType",
            "PrimaryCatalystURL",
            "LastDilutionEventDate",
            "LastInsiderBuyDate",
            "GoingConcernFlag",
            "BiotechPeerRead",
            "BiotechPeerEvidence",
            "SubscoresEvidencedCount",
            "Materiality",
            "ConvictionScore",
            "EvidencePrimary",
            "EvidenceSecondary",
            "Status",
        ]
    ensure_csv(output_path, fieldnames)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    return pd.DataFrame(output_rows)


__all__ = ["compute_runway_from_html", "run_weekly_deep_research"]
