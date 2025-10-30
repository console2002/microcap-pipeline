"""Populate deep-research results with catalyst/dilution/governance insights."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional

import pandas as pd

from app.config import load_config
from app.utils import ensure_csv, log_line, utc_now_iso


ProgressCallback = Optional[Callable[[str, str], None]]


@dataclass(frozen=True)
class FilingEntry:
    """Normalized representation of a filing row."""

    filed_at: datetime
    form: str
    url: str

    def describe(self) -> str:
        parts: list[str] = [self.filed_at.strftime("%Y-%m-%d")]
        form = self.form.strip()
        if form:
            parts.append(form)
        url = self.url.strip()
        if url:
            parts.append(url)
        return " ".join(parts)


DILUTION_PREFIXES = (
    "S-1",
    "S-3",
    "S-4",
    "S-8",
    "424B",
)

CATALYST_PREFIXES = (
    "8-K",
    "6-K",
    "10-Q",
    "10-K",
    "S-4",
    "S-3",
    "S-1",
)

GOVERNANCE_PREFIXES = (
    "DEF 14",
    "8-A12",
)

INSIDER_PREFIXES = (
    "13D",
    "13G",
    "13F",
)


SCORE_COLUMNS = ["Materiality", "SubscoresEvidenced", "Status"]


def _progress_log_path() -> str:
    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "progress.csv")
    ensure_csv(path, ["timestamp", "status", "message"])
    return path


def _emit(status: str, message: str, progress_callback: ProgressCallback) -> None:
    path = _progress_log_path()
    log_line(path, [utc_now_iso(), status, message])
    if progress_callback is not None:
        try:
            progress_callback(status, message)
        except Exception:
            # Never allow GUI callbacks to break disk logging.
            pass


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


def _normalize_form(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().upper()


def _matches_prefix(value: str, prefixes: Iterable[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _split_semicolon(text: str) -> list[str]:
    if not text:
        return []
    return [segment.strip() for segment in text.split(";") if segment.strip()]


def _format_fda_row(row: pd.Series) -> str:
    parts: list[str] = []
    date_val = row.get("DecisionDate") or row.get("decision_date")
    if pd.notna(date_val):
        try:
            parts.append(pd.Timestamp(date_val).strftime("%Y-%m-%d"))
        except Exception:
            parts.append(str(date_val))
    event = str(row.get("EventType") or row.get("event_type") or "").strip()
    if event:
        parts.append(event)
    product = str(row.get("Product") or row.get("product") or "").strip()
    if product:
        parts.append(product)
    status = str(row.get("Status") or row.get("status") or "").strip()
    if status:
        parts.append(status)
    url = str(row.get("URL") or row.get("url") or "").strip()
    if url:
        parts.append(url)
    return " | ".join(parts)


def _build_filing_map(df: pd.DataFrame, cutoff: datetime) -> dict[str, list[FilingEntry]]:
    if df.empty:
        return {}

    df = df.copy()
    df["CIK_norm"] = df["CIK"].apply(_normalize_cik)
    df["FiledAt_dt"] = pd.to_datetime(df["FiledAt"], errors="coerce")
    df = df.dropna(subset=["CIK_norm", "FiledAt_dt"])
    df = df[df["FiledAt_dt"] >= cutoff]
    if df.empty:
        return {}

    df = df.sort_values(["CIK_norm", "FiledAt_dt"], ascending=[True, False])

    entries: dict[str, list[FilingEntry]] = {}
    for _, row in df.iterrows():
        cik = row["CIK_norm"]
        form = _normalize_form(row.get("Form"))
        url = str(row.get("URL") or "").strip()
        filed_at = pd.Timestamp(row["FiledAt_dt"]).to_pydatetime()
        entry = FilingEntry(filed_at=filed_at, form=form, url=url)
        entries.setdefault(cik, []).append(entry)
    return entries


def _derive_category(entries: list[FilingEntry], prefixes: Iterable[str]) -> list[str]:
    values: list[str] = []
    for entry in entries:
        if entry.form and _matches_prefix(entry.form, prefixes):
            values.append(entry.describe())
    return values


def _choose_catalyst(
    cik: str,
    filings: list[FilingEntry],
    prefixes: Iterable[str],
    fda_map: dict[str, str],
) -> tuple[str, bool]:
    fda_text = fda_map.get(cik)
    if fda_text:
        return fda_text, True

    filing_hits = _derive_category(filings, prefixes)
    if filing_hits:
        return "; ".join(filing_hits), True
    return "", False


def _derive_materiality(has_catalyst: bool, has_dilution: bool, has_governance: bool) -> str:
    if has_catalyst and has_dilution:
        return "High"
    if has_catalyst:
        return "Medium"
    if has_dilution or has_governance:
        return "Low"
    return ""


def _derive_status(has_catalyst: bool, has_dilution: bool, has_governance: bool) -> str:
    if has_catalyst:
        return "Active"
    if has_dilution:
        return "Dilution Watch"
    if has_governance:
        return "Governance Watch"
    return "Stale"


def _load_optional_scores(data_dir: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return lookup dictionaries for manual score columns keyed by CIK/Ticker."""

    by_cik: dict[str, dict] = {}
    by_ticker: dict[str, dict] = {}
    for filename in ("shortlist.csv", "candidates.csv"):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            continue
        if df.empty:
            continue

        columns_present = [col for col in SCORE_COLUMNS if col in df.columns]
        if not columns_present:
            continue

        temp = df.copy()
        temp["CIK_norm"] = temp["CIK"].apply(_normalize_cik) if "CIK" in temp.columns else ""
        temp["Ticker_norm"] = (
            temp["Ticker"].fillna("").astype(str).str.upper() if "Ticker" in temp.columns else ""
        )
        for _, row in temp.iterrows():
            payload = {col: row.get(col) for col in columns_present if pd.notna(row.get(col)) and str(row.get(col)).strip()}
            if not payload:
                continue
            cik = row.get("CIK_norm", "")
            ticker = row.get("Ticker_norm", "")
            if cik:
                by_cik.setdefault(cik, {}).update(payload)
            elif ticker:
                by_ticker.setdefault(ticker, {}).update(payload)
    return by_cik, by_ticker


def run(data_dir: str | None = None, progress_callback: ProgressCallback = None) -> str:
    """Populate ``research_results_full.csv`` and return its path."""

    cfg = load_config()
    base_dir = data_dir or cfg.get("Paths", {}).get("data", "./data")
    os.makedirs(base_dir, exist_ok=True)

    runway_path = os.path.join(base_dir, "research_results_runway.csv")
    filings_path = os.path.join(base_dir, "filings.csv")
    fda_path = os.path.join(base_dir, "fda.csv")

    if not os.path.exists(runway_path):
        raise FileNotFoundError("research_results_runway.csv not found; run parse_q10 stage first")
    if not os.path.exists(filings_path):
        raise FileNotFoundError("filings.csv not found; run filings stage first")

    df_runway = pd.read_csv(runway_path, encoding="utf-8")
    df_filings = pd.read_csv(filings_path, encoding="utf-8")

    df_fda = pd.DataFrame()
    if os.path.exists(fda_path):
        try:
            df_fda = pd.read_csv(fda_path, encoding="utf-8")
        except Exception:
            df_fda = pd.DataFrame()

    total_rows = len(df_runway)
    _emit("INFO", f"dr_populate start ({total_rows} rows)", progress_callback)

    output_path = os.path.join(base_dir, "research_results_full.csv")
    if df_runway.empty:
        df_runway.to_csv(output_path, index=False, encoding="utf-8")
        _emit("OK", f"dr_populate complete -> {output_path}", progress_callback)
        return output_path

    for col in SCORE_COLUMNS:
        if col not in df_runway.columns:
            df_runway[col] = pd.NA

    now = datetime.utcnow()
    days_back = int(cfg.get("Windows", {}).get("DaysBack_Filings", 60) or 60)
    cutoff = now - timedelta(days=days_back)

    filing_map = _build_filing_map(df_filings, cutoff)

    fda_map: dict[str, str] = {}
    if not df_fda.empty and "CIK" in df_fda.columns:
        temp = df_fda.copy()
        temp["CIK_norm"] = temp["CIK"].apply(_normalize_cik)
        temp = temp[temp["CIK_norm"].astype(bool)]
        if "DecisionDate" in temp.columns:
            temp["DecisionDate"] = pd.to_datetime(temp["DecisionDate"], errors="coerce")
            sort_key = "DecisionDate"
        elif "decision_date" in temp.columns:
            temp["decision_date"] = pd.to_datetime(temp["decision_date"], errors="coerce")
            sort_key = "decision_date"
        else:
            sort_key = None
        if sort_key is not None:
            temp = temp.sort_values(["CIK_norm", sort_key], ascending=[True, False])
        else:
            temp = temp.sort_values(["CIK_norm"], ascending=[True])
        for cik, group in temp.groupby("CIK_norm", sort=False):
            latest = group.head(1).iloc[0]
            fda_map[cik] = _format_fda_row(latest)

    scores_by_cik, scores_by_ticker = _load_optional_scores(base_dir)

    if "CIK" in df_runway.columns:
        df_runway["CIK_norm"] = df_runway["CIK"].apply(_normalize_cik)
    else:
        df_runway["CIK_norm"] = ""
    if "Ticker" in df_runway.columns:
        df_runway["Ticker_norm"] = df_runway["Ticker"].fillna("").astype(str).str.upper()
    else:
        df_runway["Ticker_norm"] = ""

    dilution_results: list[str] = []
    catalyst_results: list[str] = []
    governance_results: list[str] = []
    insider_results: list[str] = []
    materiality_results: list[str] = []
    subscores_results: list[str] = []
    status_results: list[str] = []

    last_pct = -1
    for idx, row in df_runway.iterrows():
        cik = row.get("CIK_norm", "") or _normalize_cik(row.get("CIK"))
        ticker = row.get("Ticker_norm", "") or str(row.get("Ticker") or "").upper()
        filings = filing_map.get(cik, [])

        dilution_hits = _derive_category(filings, DILUTION_PREFIXES)
        governance_hits = _derive_category(filings, GOVERNANCE_PREFIXES)
        insider_hits = _derive_category(filings, INSIDER_PREFIXES)

        catalyst_text, has_catalyst = _choose_catalyst(cik, filings, CATALYST_PREFIXES, fda_map)
        has_dilution = bool(dilution_hits)
        has_governance = bool(governance_hits)

        dilution_results.append("; ".join(_dedupe(dilution_hits)))
        governance_results.append("; ".join(_dedupe(governance_hits)))
        insider_results.append("; ".join(_dedupe(insider_hits)))
        catalyst_results.append(catalyst_text)

        score_seed: dict[str, str] = {}
        if cik and cik in scores_by_cik:
            score_seed.update(scores_by_cik[cik])
        elif ticker and ticker in scores_by_ticker:
            score_seed.update(scores_by_ticker[ticker])

        materiality = score_seed.get("Materiality")
        if not materiality:
            materiality = row.get("Materiality")
        if pd.isna(materiality) or str(materiality).strip() == "":
            materiality = _derive_materiality(has_catalyst, has_dilution, has_governance)
        materiality_results.append(str(materiality) if materiality is not None else "")

        subscores: list[str] = []
        if score_seed.get("SubscoresEvidenced"):
            subscores.extend(_split_semicolon(str(score_seed.get("SubscoresEvidenced"))))
        existing_sub = row.get("SubscoresEvidenced")
        if pd.notna(existing_sub):
            subscores.extend(_split_semicolon(str(existing_sub)))
        if has_catalyst:
            subscores.append("Catalyst")
        if has_dilution:
            subscores.append("Dilution")
        if has_governance:
            subscores.append("Governance")
        if insider_hits:
            subscores.append("Insider")
        subscores_results.append("; ".join(_dedupe(subscores)))

        status = score_seed.get("Status") or row.get("Status")
        if pd.isna(status) or str(status).strip() == "":
            status = _derive_status(has_catalyst, has_dilution, has_governance)
        status_results.append(str(status) if status is not None else "")

        pct = int(round(((idx + 1) / total_rows) * 100))
        if pct != last_pct:
            display = str(row.get("Ticker") or row.get("CIK") or "?")
            _emit("PROGRESS", f"({pct}%) dr_populate processed {display}", progress_callback)
            last_pct = pct

    df_runway["Dilution"] = dilution_results
    df_runway["Catalyst"] = catalyst_results
    df_runway["Governance"] = governance_results
    df_runway["Insider"] = insider_results
    df_runway["Materiality"] = materiality_results
    df_runway["SubscoresEvidenced"] = subscores_results
    df_runway["Status"] = status_results

    df_runway.drop(columns=["CIK_norm", "Ticker_norm"], inplace=True)
    df_runway.to_csv(output_path, index=False, encoding="utf-8")

    _emit("OK", f"dr_populate complete -> {output_path}", progress_callback)
    return output_path


__all__ = ["run"]
