import logging
from typing import Dict, Iterable, Tuple

import pandas as pd

from app.weekly_deep_research import DILUTION_FORMS

logger = logging.getLogger(__name__)


US_RUNWAY_FORM_PREFIXES: Tuple[str, ...] = (
    "10-Q",
    "10-Q/A",
    "10-QT",
    "10-QT/A",
    "10-K",
    "10-K/A",
    "10-KT",
    "10-KT/A",
)

FPI_ANNUAL_FORM_PREFIXES: Tuple[str, ...] = (
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
)

FPI_INTERIM_FORM_PREFIXES: Tuple[str, ...] = (
    "6-K",
    "6-K/A",
)


def _normalize_form(value: object) -> str:
    return str(value).strip().upper() if value is not None else ""


def _role_limits(cfg: Dict) -> Dict[str, int | None]:
    raw_limits = cfg.get("FilingsPerTickerLimits") or {}
    limits: Dict[str, int | None] = {}
    for role, raw in raw_limits.items():
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = None
        if val is None:
            limits[role] = None
        else:
            limits[role] = val if val >= 0 else 0
    return limits


def forms_by_role(cfg: Dict) -> Dict[str, set[str]]:
    whitelist = cfg.get("FilingsWhitelistByRole") or {}
    groups = cfg.get("FilingsGroups", {}) or {}

    role_forms: Dict[str, set[str]] = {
        "Runway": set(),
        "Governance": set(),
        "Dilution": set(),
        "Catalyst": set(),
        "Insider": set(),
    }

    runway_and_gov = {_normalize_form(f) for f in whitelist.get("RunwayAndGovernance", [])}
    governance_group = {
        _normalize_form(form)
        for form in (groups.get("Governance", {}) or {}).get("forms", [])
        if form
    }

    runway_prefixes = (
        set(US_RUNWAY_FORM_PREFIXES)
        | set(FPI_ANNUAL_FORM_PREFIXES)
        | set(FPI_INTERIM_FORM_PREFIXES)
    )

    role_forms["Runway"].update(
        form for form in runway_and_gov if any(form.startswith(prefix) for prefix in runway_prefixes)
    )
    role_forms["Runway"].update(runway_prefixes)

    role_forms["Governance"].update(runway_and_gov)
    role_forms["Governance"].update(governance_group)

    role_forms["Dilution"].update({_normalize_form(f) for f in whitelist.get("Dilution", []) if f})
    role_forms["Dilution"].update({_normalize_form(f) for f in DILUTION_FORMS})

    role_forms["Catalyst"].update({_normalize_form(f) for f in whitelist.get("Catalyst", []) if f})
    role_forms["Insider"].update({_normalize_form(f) for f in whitelist.get("Insider", []) if f})

    return role_forms


def _roles_for_form(form_value: str, mapping: Dict[str, set[str]]) -> set[str]:
    roles: set[str] = set()
    for role, forms in mapping.items():
        if any(form_value.startswith(prefix) for prefix in forms if prefix):
            roles.add(role)
    return roles


def count_filings_by_role(df: pd.DataFrame, cfg: Dict) -> tuple[dict[str, int], dict[str, set[str]]]:
    if df is None or df.empty:
        return {}, {}

    form_col = "Form" if "Form" in df.columns else "FormType" if "FormType" in df.columns else None
    ticker_col = "Ticker" if "Ticker" in df.columns else "CIK" if "CIK" in df.columns else None
    if form_col is None or ticker_col is None:
        return {}, {}

    mapping = forms_by_role(cfg)
    counts: dict[str, int] = {role: 0 for role in mapping}
    tickers: dict[str, set[str]] = {role: set() for role in mapping}

    for _, row in df.iterrows():
        form_val = _normalize_form(row.get(form_col))
        roles = _roles_for_form(form_val, mapping)
        if not roles:
            continue
        ticker_val = str(row.get(ticker_col, "")).strip().upper()
        for role in roles:
            counts[role] += 1
            if ticker_val:
                tickers[role].add(ticker_val)

    counts = {role: count for role, count in counts.items() if count > 0}
    tickers = {role: vals for role, vals in tickers.items() if vals}
    return counts, tickers


def prune_filings_by_role(
    df: pd.DataFrame, cfg: Dict
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    if df is None or df.empty:
        return df, {}

    form_col = "Form" if "Form" in df.columns else "FormType" if "FormType" in df.columns else None
    filed_col = "FiledAt" if "FiledAt" in df.columns else None
    group_key = "Ticker" if "Ticker" in df.columns else "CIK" if "CIK" in df.columns else None

    if form_col is None or filed_col is None or group_key is None:
        return df, {}

    work = df.copy()
    work["_FormNorm"] = work[form_col].fillna("").astype(str).str.upper().str.strip()
    work["_FiledDt"] = pd.to_datetime(work[filed_col], errors="coerce", utc=True)

    mapping = forms_by_role(cfg)
    limits = _role_limits(cfg)
    unique_forms = work["_FormNorm"].unique().tolist()
    roles_cache = {form: _roles_for_form(form, mapping) for form in unique_forms}

    kept_indices: set[int] = set()
    applied: dict[str, dict[str, dict[str, int]]] = {}

    for ticker, group in work.groupby(group_key, dropna=False):
        ticker_str = str(ticker).strip().upper()
        ticker_roles: dict[str, dict[str, int]] = {}

        for role, patterns in mapping.items():
            cap = limits.get(role)
            if cap is None and role not in limits:
                cap = None

            role_mask = group["_FormNorm"].apply(
                lambda val: any(val.startswith(prefix) for prefix in patterns if prefix)
            )
            role_group = group[role_mask]
            if role_group.empty:
                continue

            if role == "Runway":
                role_group = role_group.copy()
                core_prefixes = US_RUNWAY_FORM_PREFIXES + FPI_ANNUAL_FORM_PREFIXES
                role_group["_RunwayCore"] = role_group["_FormNorm"].apply(
                    lambda val: any(val.startswith(prefix) for prefix in core_prefixes)
                )
                sort_cols = ["_RunwayCore", "_FiledDt", "URL", "_FormNorm"]
                sort_order = [False, False, True, True]
            else:
                sort_cols = ["_FiledDt", "URL", "_FormNorm"]
                sort_order = [False, True, True]

            sorted_group = role_group.sort_values(
                by=sort_cols,
                ascending=sort_order,
                na_position="last",
            )

            take = len(sorted_group) if cap is None else cap
            subset = sorted_group.head(take)

            if role == "Runway" and subset.empty:
                subset = sorted_group.head(1)

            kept_indices.update(subset.index.tolist())

            if cap is not None:
                kept_count = len(subset)
                if role == "Runway" and kept_count == 0 and not role_group.empty:
                    kept_count = 1
                ticker_roles[role] = {
                    "cap": cap,
                    "available": len(role_group),
                    "kept": kept_count,
                }

        # Always retain filings that do not map to any capped role
        role_free_mask = ~group["_FormNorm"].apply(lambda val: bool(roles_cache.get(val)))
        kept_indices.update(group[role_free_mask].index.tolist())

        if ticker_roles:
            applied[ticker_str or str(ticker)] = ticker_roles

    if not kept_indices:
        return work.drop(columns=["_FormNorm", "_FiledDt"], errors="ignore"), applied

    result = work.loc[work.index.isin(kept_indices)].copy()
    return result.drop(columns=["_FormNorm", "_FiledDt", "_RunwayCore"], errors="ignore"), applied
