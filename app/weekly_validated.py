from __future__ import annotations

import csv
import csv
import os
from typing import List

import pandas as pd

from app.config import load_config
from app.utils import ensure_csv

MANDATORY_FIELDS = ["RunwayQuarters", "DilutionScore", "CatalystScore"]


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def _is_biotech(sector: str, industry: str) -> bool:
    text = f"{sector} {industry}".lower()
    return "biotech" in text or "biotechnology" in text


def _passes_rules(row: pd.Series) -> bool:
    for field in MANDATORY_FIELDS:
        if pd.isna(row.get(field)) or row.get(field) in {"", "Unknown"}:
            return False
    if row.get("SubscoresEvidencedCount", 0) < 4:
        return False
    materiality = str(row.get("Materiality", "")).lower()
    if materiality in {"low", "fail", ""}:
        return False
    governance = str(row.get("GovernanceScore", "")).lower()
    if governance in {"highrisk", "concern"} and row.get("GoingConcernFlag") == "Y":
        return False
    insider = str(row.get("InsiderScore", "")).lower()
    if insider == "" or insider == "none":
        pass
    if _is_biotech(str(row.get("Sector", "")), str(row.get("Industry", ""))):
        if str(row.get("BiotechPeerRead", "")) != "Y":
            return False
    return True


def build_validated_selections(data_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    data_dir = data_dir or cfg.get("Paths", {}).get("data", "data")

    dr_path = os.path.join(data_dir, "30_deep_research.csv")
    universe_path = os.path.join(data_dir, "01_universe_gated.csv")
    shortlist_path = os.path.join(data_dir, "20_candidate_shortlist.csv")

    deep_research = _load_csv(dr_path)
    universe = _load_csv(universe_path)
    shortlist = _load_csv(shortlist_path)

    if deep_research.empty:
        raise RuntimeError("30_deep_research.csv missing or empty")

    merged = deep_research.copy()
    for source in (universe, shortlist):
        if source.empty:
            continue
        merged = merged.merge(
            source,
            how="left",
            on=[col for col in ["Ticker", "CIK"] if col in source.columns and col in merged.columns],
            suffixes=("", "_dup"),
        )
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dup_col in dup_cols:
            base_col = dup_col[:-4]
            if base_col in merged.columns:
                merged[base_col] = merged[base_col].combine_first(merged[dup_col])
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    statuses: List[str] = []
    for _, row in merged.iterrows():
        statuses.append("Validated" if _passes_rules(row) else "TBD - exclude")
    merged["Status"] = statuses

    validated = merged[merged["Status"] == "Validated"].copy()
    exclusions = merged[merged["Status"] != "Validated"].copy()

    val_path = os.path.join(data_dir, "40_validated_selections.csv")
    tbd_path = os.path.join(data_dir, "40_tbd_exclusions.csv")

    fieldnames = list(merged.columns)
    ensure_csv(val_path, fieldnames)
    with open(val_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, r in validated.iterrows():
            writer.writerow(r.to_dict())

    ensure_csv(tbd_path, fieldnames)
    with open(tbd_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, r in exclusions.iterrows():
            writer.writerow(r.to_dict())

    return validated, exclusions


__all__ = ["build_validated_selections"]
