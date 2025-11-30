import json, os
from typing import Any, Dict


def weekly_allowed_forms(cfg: Dict[str, Any] | None = None) -> set[str]:
    """
    Return the set of forms allowed for WEEKLY filings, derived from config.

    Priority:
      1) Use FilingsWhitelistByRole (union of all groups) if present.
      2) Fallback to flat FilingsWhitelist list if groups missing.
    """

    if cfg is None:
        cfg = load_config()

    by_role = cfg.get("FilingsWhitelistByRole") or {}
    if by_role:
        forms: set[str] = set()
        for group in by_role.values():
            if group:
                forms.update(group)
        return {str(form).strip().upper() for form in forms if str(form).strip()}

    return {str(form).strip().upper() for form in cfg.get("FilingsWhitelist", []) if str(form).strip()}

def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)

    return cfg

def save_config(cfg: Dict[str, Any], path: str = "config.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def filings_form_lookbacks(cfg: Dict[str, Any]) -> Dict[str, int]:
    groups = cfg.get("FilingsGroups", {}) or {}
    out: Dict[str, int] = {}
    for group in groups.values():
        try:
            lookback = int(group.get("lookback_days", 0) or 0)
        except Exception:
            continue
        forms = group.get("forms") or []
        for form in forms:
            if not form:
                continue
            key = str(form).strip().upper()
            if not key:
                continue
            current = out.get(key, 0)
            if lookback > current:
                out[key] = lookback
    return out


def filings_max_lookback(cfg: Dict[str, Any], default: int = 60) -> int:
    mapping = filings_form_lookbacks(cfg)
    if not mapping:
        return default
    return max(mapping.values())
