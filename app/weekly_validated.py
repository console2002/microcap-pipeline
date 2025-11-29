from __future__ import annotations

import csv
import os
from typing import List, Tuple

import pandas as pd

from app.config import load_config
from app.utils import ensure_csv

MANDATORY_FIELDS = ["RunwayQuarters", "Dilution", "Catalyst"]


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def _is_biotech(sector: str, industry: str) -> bool:
    text = f"{sector} {industry}".lower()
    return "biotech" in text or "biotechnology" in text


def _subscore_evidenced(row: pd.Series, value_field: str, evidence_field: str) -> bool:
    value = row.get(value_field)
    if pd.isna(value) or str(value) in {"", "nan", "TBD", "Unknown"}:
        return False
    evidence = str(row.get(evidence_field, ""))
    return bool(evidence)


def _materiality_passed(materiality: str) -> bool:
    lowered = str(materiality).lower()
    if lowered.startswith("fail"):
        return False
    if lowered in {"", "nan"}:
        return True
    return True


def evaluate_validation(row: pd.Series) -> Tuple[str, str]:
    """Return (status, reason) using W3/W4 gating rules."""

    dilution_ok = _subscore_evidenced(row, "Dilution", "DilutionEvidencePrimary")
    runway_ok = _subscore_evidenced(row, "Runway (qtrs)", "RunwayEvidencePrimary")
    catalyst_ok = _subscore_evidenced(row, "Catalyst", "CatalystEvidencePrimary")
    mandatory_ok = dilution_ok and runway_ok and catalyst_ok

    subscore_count = int(row.get("Subscores Evidenced (x/5)", row.get("SubscoresEvidencedCount", 0)) or 0)
    materiality_field = row.get("Materiality (pass/fail + note)", row.get("Materiality", ""))
    materiality_ok = _materiality_passed(materiality_field)

    biotech_peer = str(row.get("Biotech Peer Read-Through (Y/N + link)", ""))
    biotech_ok = not biotech_peer.startswith("TBD")

    if mandatory_ok and subscore_count >= 4 and materiality_ok and biotech_ok:
        return "Validated", ""

    if not mandatory_ok:
        return "TBD — exclude", "Mandatory subscore missing"
    if subscore_count < 4:
        return "TBD — exclude", "Subscores <4/5"
    if not biotech_ok:
        return "TBD — exclude", "Biotech peer read-through missing"
    if not materiality_ok:
        return "TBD — exclude", "Materiality fail"
    return "TBD — exclude", "Did not meet validation rule"


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
    reasons: List[str] = []
    for _, row in merged.iterrows():
        status, reason = evaluate_validation(row)
        statuses.append(status)
        reasons.append(reason)
    merged["Status"] = statuses
    merged["Reason"] = reasons

    validated = merged[merged["Status"] == "Validated"].copy()
    exclusions = merged[merged["Status"] != "Validated"].copy()

    val_path = os.path.join(data_dir, "40_validated_selections.csv")
    tbd_path = os.path.join(data_dir, "40_tbd_exclusions.csv")

    # W4 schema fields
    validated_fields = [
        "Ticker",
        "Company",
        "CIK",
        "Sector",
        "Catalyst rationale",
        "Validation status",
        "ADV20",
        "Discovery price ref",
    ]
    exclusion_fields = ["Ticker", "Company", "CIK", "Reason"]

    validated_output: List[dict] = []
    for _, r in validated.iterrows():
        validated_output.append(
            {
                "Ticker": r.get("Ticker"),
                "Company": r.get("Company"),
                "CIK": r.get("CIK"),
                "Sector": r.get("Sector"),
                "Catalyst rationale": r.get("Catalyst", r.get("CatalystScore", "")),
                "Validation status": r.get("Status"),
                "ADV20": r.get("ADV20"),
                "Discovery price ref": r.get("Price", r.get("DiscoveryPrice", "")),
            }
        )

    exclusions_output: List[dict] = []
    for _, r in exclusions.iterrows():
        exclusions_output.append(
            {
                "Ticker": r.get("Ticker"),
                "Company": r.get("Company"),
                "CIK": r.get("CIK"),
                "Reason": r.get("Reason", "Did not meet validation rule"),
            }
        )

    ensure_csv(val_path, validated_fields)
    with open(val_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=validated_fields)
        writer.writeheader()
        for r in validated_output:
            writer.writerow(r)

    ensure_csv(tbd_path, exclusion_fields)
    with open(tbd_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=exclusion_fields)
        writer.writeheader()
        for r in exclusions_output:
            writer.writerow(r)

    print(
        f"WEEKLY_W3_W4 SUMMARY: 30_deep_research={len(merged)}, "
        f"40_validated={len(validated_output)}, 40_tbd={len(exclusions_output)}"
    )

    return pd.DataFrame(validated_output), pd.DataFrame(exclusions_output)


__all__ = ["build_validated_selections", "evaluate_validation"]
