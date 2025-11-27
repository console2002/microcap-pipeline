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


def _dilution_score(forms: Iterable[str]) -> str:
    normalized = {_normalize_form(f) for f in forms if f}
    if any(form in DILUTION_FORMS or form.startswith(tuple(DILUTION_FORMS)) for form in normalized):
        return "High"
    if normalized:
        return "Low"
    return "Unknown"


def _catalyst_score(events: pd.DataFrame) -> str:
    if events is None or events.empty:
        return "None"
    tier_series = events.get("Tier")
    if tier_series is not None and tier_series.astype(str).str.contains("1", case=False, na=False).any():
        return "Tier-1"
    return "Tier-2"


def _governance_score(forms: Iterable[str]) -> str:
    normalized = {_normalize_form(f) for f in forms if f}
    if any("DEF 14A" in form for form in normalized) or any(
        form.startswith(prefix) for form in normalized for prefix in RUNWAY_FORMS
    ):
        return "OK"
    return ""


def _insider_score(forms: Iterable[str]) -> str:
    normalized = {_normalize_form(f) for f in forms if f}
    if any(form.startswith("4") or form in {"3", "5"} for form in normalized):
        return "Weak"
    return ""


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
        if not candidate_filings.empty:
            relevant_mask = candidate_filings.get(form_col, pd.Series(dtype=str)).astype(str).str.upper().str.startswith(RUNWAY_FORMS)
            relevant = candidate_filings[relevant_mask]
            if not relevant.empty:
                runway_link = relevant.iloc[0].get("FilingURL") or relevant.iloc[0].get("URL", "")
                runway_quarters = _runway_from_filing(str(runway_link), adapter=adapter)
        dilution = _dilution_score(filing_forms)
        ticker_series = events.get("Ticker", pd.Series(dtype=str))
        cik_series = events.get("CIK", pd.Series(dtype=str))
        candidate_events = events[(ticker_series.astype(str) == str(ticker)) | (cik_series.astype(str) == str(cik))]
        catalyst = _catalyst_score(candidate_events)
        governance = _governance_score(filing_forms)
        insider = _insider_score(filing_forms)
        biotech_flag = _biotech_peer_flag(str(sector), str(industry))
        evidence_links: list[str] = []
        if runway_link:
            evidence_links.append(str(runway_link))
        evidence_links.extend(candidate_filings.get("FilingURL", pd.Series(dtype=str)).dropna().astype(str).tolist())
        evidence_primary = _aggregate_evidence(evidence_links)

        subscores = {
            "runway": runway_quarters is not None,
            "dilution": dilution not in {"", "Unknown"},
            "catalyst": catalyst != "None",
            "governance": bool(governance),
            "insider": bool(insider),
        }
        subscore_count = sum(1 for v in subscores.values() if v and bool(evidence_primary))
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
                "BiotechPeerRead": biotech_flag,
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
            "BiotechPeerRead",
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
