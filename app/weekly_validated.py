from __future__ import annotations

import csv
import logging
import os
from typing import Callable, Dict, List, Tuple

import pandas as pd

from app.config import load_config
from app.utils import ensure_csv, utc_now_iso

MANDATORY_FIELDS = ["RunwayQuarters", "Dilution", "Catalyst"]

logger = logging.getLogger(__name__)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def _is_biotech(sector: str, industry: str) -> bool:
    text = f"{sector} {industry}".lower()
    return "biotech" in text or "biotechnology" in text


def _emit_progress(progress_fn: Callable[[str], None] | None, message: str) -> None:
    if progress_fn is None:
        return
    progress_fn(f"{utc_now_iso()} | {message}")


def _has_value(val) -> bool:
    return pd.notna(val) and str(val).strip() not in {"", "nan", "TBD", "Unknown"}


def _subscore_evidenced(row: pd.Series, value_fields, evidence_field: str) -> bool:
    if isinstance(value_fields, str):
        value_fields = [value_fields]

    value = None
    for field in value_fields:
        candidate = row.get(field)
        if pd.notna(candidate) and str(candidate) not in {"", "nan", "TBD", "Unknown"}:
            value = candidate
            break

    if value is None:
        return False

    evidence = str(row.get(evidence_field, ""))
    return bool(evidence)


def _materiality_passed(materiality: str) -> bool:
    lowered = str(materiality).strip().lower()
    return lowered.startswith("pass")


def _compute_validation_gates(row: pd.Series) -> tuple[Dict[str, bool], int]:
    dilution_ok = _subscore_evidenced(row, ["Dilution", "DilutionScore"], "DilutionEvidencePrimary")

    runway_numeric = _has_value(row.get("RunwayQuarters"))
    runway_display = str(row.get("Runway (qtrs)", "")).strip()
    runway_display_ok = runway_display not in {"", "nan", "TBD", "Unknown"}
    runway_evidence_ok = _has_value(row.get("RunwayEvidencePrimary"))
    runway_ok = (runway_numeric or runway_display_ok) and runway_evidence_ok

    catalyst_ok = _subscore_evidenced(row, ["Catalyst", "CatalystScore"], "CatalystEvidencePrimary")

    subscore_count = int(row.get("Subscores Evidenced (x/5)", row.get("SubscoresEvidencedCount", 0)) or 0)

    materiality_field = row.get("Materiality (pass/fail + note)", row.get("Materiality", ""))
    materiality_ok = _materiality_passed(materiality_field)

    biotech_peer = str(
        row.get("BiotechPeerRead", row.get("Biotech Peer Read-Through (Y/N + link)", ""))
    ).strip()
    biotech_needs_peer = biotech_peer.upper().startswith("Y") or biotech_peer.upper().startswith("TBD")
    biotech_ok = not (biotech_needs_peer and biotech_peer.upper().startswith("TBD"))

    gates = {
        "C1": runway_ok,
        "C2": dilution_ok,
        "C3": catalyst_ok,
        "C4": subscore_count >= 4,
        "C5": biotech_ok,
        "C6": materiality_ok,
    }
    return gates, subscore_count


def summarize_validation_gates(df_deep: pd.DataFrame) -> dict:
    """
    Compute aggregate validation statistics for the WEEKLY W4 step.

    Returns a dict with at least:
        {
            "N_rows": int,
            "C1_pass": int,
            "C1_fail": int,
            "C2_pass": int,
            "C2_fail": int,
            "C3_pass": int,
            "C3_fail": int,
            "C4_pass": int,
            "C4_fail": int,
            "C5_pass": int,
            "C5_fail": int,
            "C6_pass": int,
            "C6_fail": int,
            "N_validated_all_true": int,
            "N_any_gate_false": int,
        }
    """

    gate_labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    stats = {"N_rows": len(df_deep), "N_validated_all_true": 0, "N_any_gate_false": 0}

    for label in gate_labels:
        stats[f"{label}_pass"] = 0
        stats[f"{label}_fail"] = 0

    for _, row in df_deep.iterrows():
        gates, _ = _compute_validation_gates(row)
        all_true = all(gates.values())
        if all_true:
            stats["N_validated_all_true"] += 1
        else:
            stats["N_any_gate_false"] += 1

        for label in gate_labels:
            if gates.get(label):
                stats[f"{label}_pass"] += 1
            else:
                stats[f"{label}_fail"] += 1

    return stats


def evaluate_validation(row: pd.Series) -> Tuple[str, str]:
    """Return (status, reason) using W3/W4 gating rules."""

    gates, subscore_count = _compute_validation_gates(row)

    if all(gates.values()):
        return "Validated", ""

    missing_reasons = []
    if not gates["C1"]:
        missing_reasons.append("Mandatory subscore missing: Runway")
    if not gates["C2"]:
        missing_reasons.append("Mandatory subscore missing: Dilution")
    if not gates["C3"]:
        missing_reasons.append("Mandatory subscore missing: Catalyst")
    if not gates["C4"]:
        missing_reasons.append("Subscores <4/5")
    if not gates["C5"]:
        missing_reasons.append("Biotech peer missing")
    if not gates["C6"]:
        missing_reasons.append("Materiality fail")

    reason = "; ".join(missing_reasons) if missing_reasons else "Did not meet validation rule"
    return "TBD — exclude", reason


def build_validated_selections(
    data_dir: str | None = None, progress_fn: Callable[[str], None] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "Industry",
        "Venue",
        "Price",
        "MarketCap",
        "ADV20",
        "Runway (qtrs)",
        "RunwayQuarters",
        "RunwayEvidencePrimary",
        "DilutionScore",
        "DilutionEvidencePrimary",
        "CatalystScore",
        "CatalystEvidencePrimary",
        "GovernanceScore",
        "InsiderScore",
        "BiotechPeerRead",
        "SubscoresEvidencedCount",
        "Materiality",
        "ConvictionScore",
        "PrimaryCatalystDate",
        "PrimaryCatalystType",
        "PrimarySource",
        "SecondarySource",
        "Status",
        "ValidationStatus",
    ]
    exclusion_fields = [
        "Ticker",
        "Company",
        "CIK",
        "Sector",
        "Industry",
        "Reason",
        "Materiality",
        "Status",
    ]

    def _first_available(row: pd.Series, keys: list[str]):
        for key in keys:
            if key in row and pd.notna(row.get(key)):
                return row.get(key)
        return None

    validated_output: List[dict] = []
    for _, r in validated.iterrows():
        primary_links = str(r.get("Evidence (Primary links)", "")).split(";") if r.get("Evidence (Primary links)") else []
        secondary_links = (
            str(r.get("Evidence (Secondary links)", "")).split(";")
            if r.get("Evidence (Secondary links)")
            else []
        )
        validated_output.append(
            {
                "Ticker": r.get("Ticker"),
                "Company": r.get("Company"),
                "CIK": r.get("CIK"),
                "Sector": r.get("Sector"),
                "Industry": r.get("Industry"),
                "Venue": r.get("Venue"),
                "Price": _first_available(r, ["Price", "DiscoveryPrice", "Close"]),
                "MarketCap": _first_available(r, ["MarketCap", "Cap_Musd", "Cap($M)"]),
                "ADV20": _first_available(r, ["ADV20", "ADV20_k"]),
                "Runway (qtrs)": _first_available(r, ["Runway (qtrs)", "RunwayQuarters"]),
                "RunwayQuarters": r.get("RunwayQuarters"),
                "RunwayEvidencePrimary": r.get("RunwayEvidencePrimary"),
                "DilutionScore": r.get("Dilution", r.get("DilutionScore")),
                "DilutionEvidencePrimary": r.get("DilutionEvidencePrimary"),
                "CatalystScore": r.get("Catalyst", r.get("CatalystScore")),
                "CatalystEvidencePrimary": r.get("CatalystEvidencePrimary"),
                "GovernanceScore": r.get("Governance", r.get("GovernanceScore")),
                "InsiderScore": r.get("Insider", r.get("InsiderScore")),
                "BiotechPeerRead": r.get("BiotechPeerRead", r.get("Biotech Peer Read-Through (Y/N + link)")),
                "SubscoresEvidencedCount": _first_available(
                    r, ["SubscoresEvidencedCount", "Subscores Evidenced (x/5)"]
                ),
                "Materiality": r.get("Materiality", r.get("Materiality (pass/fail + note)")),
                "ConvictionScore": r.get("ConvictionScore"),
                "PrimaryCatalystDate": r.get("PrimaryCatalystDate"),
                "PrimaryCatalystType": r.get("PrimaryCatalystType"),
                "PrimarySource": primary_links[0] if primary_links else r.get("RunwayEvidencePrimary"),
                "SecondarySource": secondary_links[0] if secondary_links else r.get("EvidenceSecondary"),
                "Status": r.get("Status", "Validated"),
                "ValidationStatus": r.get("Status", "Validated"),
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
                "Materiality": r.get("Materiality", r.get("Materiality (pass/fail + note)")),
                "Status": r.get("Status"),
            }
        )

    stats = summarize_validation_gates(merged)
    msg = (
        "WEEKLY_VALIDATION "
        f"N_rows={stats['N_rows']} "
        f"N_validated_all_true={stats['N_validated_all_true']} "
        f"N_any_gate_false={stats['N_any_gate_false']} "
        f"C1={stats['C1_pass']}/{stats['C1_fail']} "
        f"C2={stats['C2_pass']}/{stats['C2_fail']} "
        f"C3={stats['C3_pass']}/{stats['C3_fail']} "
        f"C4={stats['C4_pass']}/{stats['C4_fail']} "
        f"C5={stats['C5_pass']}/{stats['C5_fail']} "
        f"C6={stats['C6_pass']}/{stats['C6_fail']}"
    )
    logger.info(msg)
    _emit_progress(progress_fn, msg)

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


__all__ = [
    "build_validated_selections",
    "evaluate_validation",
    "summarize_validation_gates",
]
