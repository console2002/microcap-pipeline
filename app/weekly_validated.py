from __future__ import annotations

import csv
import logging
import os
from typing import Callable, Dict, List, Tuple

import pandas as pd

from app.biotech_utils import is_biotech
from app.config import load_config
from app.settings import BIOTECH_PEER_REQUIRED_FOR_VALIDATION
from app.utils import ensure_csv, log_line, utc_now_iso

# Validation gates are orchestrated in W4. The gate booleans below feed into
# both Status and ValidationStatus for 40_validated_selections.csv. The current
# W4 implementation uses the following decision flags:
# - GATE_UNIVERSE: price/ADV20/cap hard gates are still respected at W4.
# - GATE_MANDATORY_SUBSCORES: Dilution, Runway, Catalyst all scored with primary
#   evidence; Secondary evidence is optional per weekly.txt.
# - GATE_RUNWAY_NUMERIC: RunwayQuarters must be numeric and positive.
# - GATE_SUBSCORE_COUNT: SubscoresEvidencedCount >= 4 as in W3.
# - GATE_BIOTECH_PEER: biotech names require a positive peer read-through.
# - GATE_MATERIALITY: materiality must not be a failing state.
#
# Status/ValidationStatus are set to "Validated" only when every gate above is
# true; otherwise they are marked "TBD - exclude" with an aggregated Reason.
# Mandatory score fields originate from 30_deep_research.csv (W3) and are
# optionally enriched with overlapping fields from 01_universe_gated.csv and
# 20_candidate_shortlist.csv during the merge below.
MANDATORY_FIELDS = ["RunwayQuarters", "Dilution", "Catalyst"]
VALIDATION_GATE_ORDER = [
    "GATE_UNIVERSE",
    "GATE_MANDATORY_SUBSCORES",
    "GATE_RUNWAY_NUMERIC",
    "GATE_SUBSCORE_COUNT",
    "GATE_BIOTECH_PEER",
    "GATE_MATERIALITY",
]

logger = logging.getLogger(__name__)

_PROGRESS_LOG_PATH: str | None = None


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
def _emit_progress(progress_fn: Callable[[str], None] | None, message: str) -> None:
    if progress_fn is None:
        return
    progress_fn(f"{utc_now_iso()} | {message}")


def _progress_log_path() -> str:
    global _PROGRESS_LOG_PATH
    if _PROGRESS_LOG_PATH:
        return _PROGRESS_LOG_PATH

    cfg = load_config()
    logs_dir = cfg.get("Paths", {}).get("logs", "./logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "progress.csv")
    ensure_csv(path, ["timestamp", "status", "message"])
    _PROGRESS_LOG_PATH = path
    return path


def _log_progress_line(message: str, progress_fn: Callable[[str], None] | None, status: str = "INFO") -> None:
    path = _progress_log_path()
    log_line(path, [utc_now_iso(), status, message])
    _emit_progress(progress_fn, message)


def _has_value(val) -> bool:
    return pd.notna(val) and str(val).strip() not in {"", "nan", "TBD", "Unknown"}


def _first_available_value(row: pd.Series, keys: List[str]):
    for key in keys:
        if key in row:
            val = row.get(key)
            if pd.notna(val):
                return val
    return None


def _coerce_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(numeric) else numeric


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


def _compute_validation_gates(row: pd.Series) -> tuple[Dict[str, bool], Dict[str, str], int, float | None]:
    dilution_value = _first_available_value(row, ["Dilution", "DilutionScore"])
    dilution_scored = _has_value(dilution_value)
    dilution_evidence = _has_value(row.get("DilutionEvidencePrimary"))

    runway_value = None
    for field in ["RunwayQuarters", "Runway (qtrs)"]:
        runway_value = _coerce_float(row.get(field))
        if runway_value is not None:
            break
    runway_numeric_ok = runway_value is not None and runway_value > 0
    runway_evidence_ok = _has_value(row.get("RunwayEvidencePrimary"))

    catalyst_value = _first_available_value(
        row,
        ["Catalyst", "CatalystScore", "PrimaryCatalystType"],
    )
    catalyst_scored = _has_value(catalyst_value)
    catalyst_evidence = any(
        _has_value(row.get(field))
        for field in ["CatalystEvidencePrimary", "PrimaryCatalystURL", "PrimarySource"]
    )

    mandatory_subscores_ok = dilution_scored and dilution_evidence and runway_evidence_ok and catalyst_scored and catalyst_evidence

    subscore_count = int(row.get("Subscores Evidenced (x/5)", row.get("SubscoresEvidencedCount", 0)) or 0)

    materiality_field = row.get("Materiality (pass/fail + note)", row.get("Materiality", ""))
    materiality_ok = _materiality_passed(materiality_field)

    biotech_peer = str(
        row.get("BiotechPeerRead", row.get("Biotech Peer Read-Through (Y/N + link)", ""))
    ).strip()
    biotech_candidate = is_biotech(str(row.get("Sector", "")), str(row.get("Industry", "")))

    def _biotech_peer_pass(peer_value: str) -> bool:
        normalized = str(peer_value or "").strip().upper()
        return normalized.startswith("Y") or normalized.startswith("PASS") or normalized.startswith("OK")

    if biotech_candidate:
        if BIOTECH_PEER_REQUIRED_FOR_VALIDATION:
            biotech_ok = _biotech_peer_pass(biotech_peer)
        else:
            biotech_ok = True
    else:
        biotech_ok = True

    price_value = _coerce_float(_first_available_value(row, ["Price", "DiscoveryPrice", "Close"]))
    adv_value = _coerce_float(_first_available_value(row, ["ADV20", "ADV20_k"]))
    cap_value = _coerce_float(_first_available_value(row, ["MarketCap", "Cap_Musd", "Cap($M)"]))

    universe_failures: list[str] = []
    if price_value is None or price_value < 1.0:
        universe_failures.append("Price<1")
    if adv_value is None or adv_value < 40_000:
        universe_failures.append("ADV<40k")
    if cap_value is None or cap_value >= 400_000_000:
        universe_failures.append("Cap≥400M")
    universe_ok = len(universe_failures) == 0

    gates = {
        "GATE_UNIVERSE": universe_ok,
        "GATE_MANDATORY_SUBSCORES": mandatory_subscores_ok,
        "GATE_RUNWAY_NUMERIC": runway_numeric_ok,
        "GATE_SUBSCORE_COUNT": subscore_count >= 4,
        "GATE_BIOTECH_PEER": biotech_ok,
        "GATE_MATERIALITY": materiality_ok,
    }

    reasons = {
        "GATE_UNIVERSE": ", ".join(universe_failures) if universe_failures else "",
        "GATE_MANDATORY_SUBSCORES": "; ".join(
            [
                msg
                for msg in [
                    None if dilution_scored else "Dilution missing/unknown",
                    None if dilution_evidence else "Dilution evidence missing",
                    None if runway_evidence_ok else "Runway evidence missing",
                    None if catalyst_scored else "Catalyst missing/unknown",
                    None if catalyst_evidence else "Catalyst evidence missing",
                ]
                if msg
            ]
        ),
        "GATE_RUNWAY_NUMERIC": "Runway missing/invalid" if not runway_numeric_ok else "",
        "GATE_SUBSCORE_COUNT": "Subscores <4/5" if subscore_count < 4 else "",
        "GATE_BIOTECH_PEER": "Biotech peer read missing/failed" if biotech_candidate and not biotech_ok else "",
        "GATE_MATERIALITY": "Materiality fail" if not materiality_ok else "",
    }

    return gates, reasons, subscore_count, runway_value


def summarize_validation_gates(df_deep: pd.DataFrame) -> dict:
    """
    Compute aggregate validation statistics for the WEEKLY W4 step.

    Returns a dict with at least:
        {
            "N_rows": int,
            "{gate}_pass": int,
            "{gate}_fail": int,
            "N_validated_all_true": int,
            "N_any_gate_false": int,
        }
    """

    gate_labels = VALIDATION_GATE_ORDER
    stats = {"N_rows": len(df_deep), "N_validated_all_true": 0, "N_any_gate_false": 0}

    for label in gate_labels:
        stats[f"{label}_pass"] = 0
        stats[f"{label}_fail"] = 0

    for _, row in df_deep.iterrows():
        gates, _, _, _ = _compute_validation_gates(row)
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

    gates, reasons, _, _ = _compute_validation_gates(row)

    if all(gates.values()):
        return "Validated", ""

    missing_reasons = [reasons.get(label, "") for label in VALIDATION_GATE_ORDER if not gates.get(label)]
    reason = "; ".join([r for r in missing_reasons if r]) or "Did not meet validation rule"
    return "TBD - exclude", reason


def _log_weekly_summary(data_dir: str, progress_fn: Callable[[str], None] | None = None) -> None:
    paths = {
        "universe": os.path.join(data_dir, "01_universe_gated.csv"),
        "filings": os.path.join(data_dir, "02_filings.csv"),
        "events": os.path.join(data_dir, "09_events.csv"),
        "shortlist": os.path.join(data_dir, "20_candidate_shortlist.csv"),
        "deep": os.path.join(data_dir, "30_deep_research.csv"),
        "validated": os.path.join(data_dir, "40_validated_selections.csv"),
        "tbd": os.path.join(data_dir, "40_tbd_exclusions.csv"),
    }

    counts = {name: len(_load_csv(path)) for name, path in paths.items()}
    msg = (
        "WEEKLY_SUMMARY: "
        f"universe={counts['universe']} "
        f"filings={counts['filings']} "
        f"events={counts['events']} "
        f"shortlist={counts['shortlist']} "
        f"deep={counts['deep']} "
        f"validated={counts['validated']} "
        f"tbd={counts['tbd']}"
    )
    logger.info(msg)
    _log_progress_line(msg, progress_fn)


def _log_weekly_validation_breakdown(
    stats: Dict[str, int], exclusions: pd.DataFrame, progress_fn: Callable[[str], None] | None = None
) -> None:
    reason_counts: Dict[str, int] = {}
    if not exclusions.empty and "Reason" in exclusions.columns:
        series = exclusions["Reason"].fillna("").astype(str)
        counts = series.value_counts()
        reason_counts = counts.to_dict()

    reason_parts = [f"{reason}={count}" for reason, count in list(reason_counts.items())[:4]]
    reason_text = ", ".join(reason_parts) if reason_parts else "none"
    msg = (
        "WEEKLY_VALIDATION: "
        f"validated={stats.get('N_validated_all_true', 0)} "
        f"tbd={stats.get('N_any_gate_false', 0)} "
        f"reasons=[{reason_text}]"
    )
    logger.info(msg)
    _log_progress_line(msg, progress_fn)


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
    gate_flags: Dict[str, List[bool]] = {label: [] for label in VALIDATION_GATE_ORDER}
    for _, row in merged.iterrows():
        gates, gate_reasons, _, _ = _compute_validation_gates(row)
        if all(gates.values()):
            status, reason = "Validated", ""
        else:
            reason_parts = [gate_reasons.get(label, "") for label in VALIDATION_GATE_ORDER if not gates.get(label)]
            reason = "; ".join([part for part in reason_parts if part]) or "Did not meet validation rule"
            status = "TBD - exclude"
        statuses.append(status)
        reasons.append(reason)
        for label in VALIDATION_GATE_ORDER:
            gate_flags[label].append(bool(gates.get(label)))

    merged["Status"] = statuses
    merged["Reason"] = reasons
    for label in VALIDATION_GATE_ORDER:
        merged[label] = gate_flags[label]

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
        "RunwayQuarters",
        "Runway (qtrs)",
        "RunwaySourceURL",
        "RunwaySourceFiledAt",
        "DilutionScore",
        "CatalystScore",
        "GovernanceScore",
        "InsiderScore",
        "BiotechPeerRead",
        "SubscoresEvidencedCount",
        "Materiality",
        "ConvictionScore",
        "PrimaryCatalystDate",
        "PrimaryCatalystType",
        "PrimaryCatalystTier",
        "PrimaryCatalystURL",
        "RunwayEvidencePrimary",
        "DilutionEvidencePrimary",
        "CatalystEvidencePrimary",
        "PrimarySource",
        "SecondarySource",
        "Status",
        "ValidationStatus",
        # Optional gate debug fields to align 40_validated with weekly.txt.
        *VALIDATION_GATE_ORDER,
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
        *VALIDATION_GATE_ORDER,
    ]

    validated_output: List[dict] = []
    for _, r in validated.iterrows():
        primary_raw = r.get("Evidence (Primary links)")
        primary_links = (
            str(primary_raw).split(";") if pd.notna(primary_raw) and str(primary_raw).strip() else []
        )
        secondary_raw = r.get("Evidence (Secondary links)")
        secondary_links = (
            str(secondary_raw).split(";") if pd.notna(secondary_raw) and str(secondary_raw).strip() else []
        )
        primary_catalyst_url = _first_available_value(
            r, ["PrimaryCatalystURL", "PrimarySource", "PrimaryFilingURL"]
        )
        validated_output.append(
            {
                "Ticker": r.get("Ticker"),
                "Company": r.get("Company"),
                "CIK": r.get("CIK"),
                "Sector": r.get("Sector"),
                "Industry": r.get("Industry"),
                "Venue": r.get("Venue"),
                "Price": _first_available_value(r, ["Price", "DiscoveryPrice", "Close"]),
                "MarketCap": _first_available_value(r, ["MarketCap", "Cap_Musd", "Cap($M)"]),
                "ADV20": _first_available_value(r, ["ADV20", "ADV20_k"]),
                "Runway (qtrs)": _first_available_value(r, ["Runway (qtrs)", "RunwayQuarters"]),
                "RunwayQuarters": r.get("RunwayQuarters"),
                "RunwaySourceURL": r.get("RunwaySourceURL"),
                "RunwaySourceFiledAt": r.get("RunwaySourceFiledAt"),
                "DilutionScore": r.get("Dilution", r.get("DilutionScore")),
                "CatalystScore": r.get("Catalyst", r.get("CatalystScore")),
                "GovernanceScore": r.get("Governance", r.get("GovernanceScore")),
                "InsiderScore": r.get("Insider", r.get("InsiderScore")),
                "BiotechPeerRead": r.get(
                    "BiotechPeerRead", r.get("Biotech Peer Read-Through (Y/N + link)")
                ),
                "SubscoresEvidencedCount": _first_available_value(
                    r, ["SubscoresEvidencedCount", "Subscores Evidenced (x/5)"]
                ),
                "Materiality": r.get("Materiality", r.get("Materiality (pass/fail + note)")),
                "ConvictionScore": r.get("ConvictionScore"),
                "PrimaryCatalystDate": _first_available_value(r, ["PrimaryCatalystDate", "EventDate"]),
                "PrimaryCatalystType": _first_available_value(r, ["PrimaryCatalystType", "CatalystType"]),
                "PrimaryCatalystTier": _first_available_value(r, ["PrimaryCatalystTier", "EventTier"]),
                "PrimaryCatalystURL": primary_catalyst_url,
                "RunwayEvidencePrimary": r.get("RunwayEvidencePrimary"),
                "DilutionEvidencePrimary": r.get("DilutionEvidencePrimary"),
                "CatalystEvidencePrimary": r.get("CatalystEvidencePrimary"),
                "PrimarySource": primary_catalyst_url
                or (primary_links[0] if primary_links else r.get("RunwayEvidencePrimary")),
                # Reserved for future secondary evidence; normalize to empty string when absent.
                "SecondarySource": (
                    secondary_links[0]
                    if secondary_links
                    else ("" if pd.isna(r.get("EvidenceSecondary")) else r.get("EvidenceSecondary", ""))
                ),
                "Status": r.get("Status", "Validated"),
                # Legacy mirror of Status kept for compatibility with older consumers.
                "ValidationStatus": r.get("Status", "Validated"),
                **{label: bool(r.get(label)) for label in VALIDATION_GATE_ORDER},
            }
        )

    exclusions_output: List[dict] = []
    for _, r in exclusions.iterrows():
        exclusions_output.append(
            {
                "Ticker": r.get("Ticker"),
                "Company": r.get("Company"),
                "CIK": r.get("CIK"),
                "Sector": r.get("Sector"),
                "Industry": r.get("Industry"),
                "Reason": r.get("Reason", "Did not meet validation rule"),
                "Materiality": r.get("Materiality", r.get("Materiality (pass/fail + note)")),
                "Status": r.get("Status"),
                **{label: bool(r.get(label)) for label in VALIDATION_GATE_ORDER},
            }
        )

    stats = summarize_validation_gates(merged)
    gate_parts = [f"{label}={stats[f'{label}_pass']}/{stats[f'{label}_fail']}" for label in VALIDATION_GATE_ORDER]
    gate_text = " ".join(gate_parts)
    msg = (
        "WEEKLY_VALIDATION "
        f"N_rows={stats['N_rows']} "
        f"N_validated_all_true={stats['N_validated_all_true']} "
        f"N_any_gate_false={stats['N_any_gate_false']} "
        f"{gate_text}"
    )
    logger.info(msg)
    _log_progress_line(msg, progress_fn)

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

    _log_weekly_validation_breakdown(stats, exclusions, progress_fn)
    _log_weekly_summary(data_dir, progress_fn)

    return pd.DataFrame(validated_output), pd.DataFrame(exclusions_output)


__all__ = [
    "build_validated_selections",
    "evaluate_validation",
    "summarize_validation_gates",
]
