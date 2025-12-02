import json, os
from typing import Any, Dict


def weekly_allowed_forms(cfg: Dict[str, Any] | None = None) -> set[str]:
    """
    Return the set of forms allowed for WEEKLY filings, derived from FilingsWhitelistByRole.
    """

    if cfg is None:
        cfg = load_config()

    by_role = cfg.get("FilingsWhitelistByRole") or {}
    forms: set[str] = set()
    for group in by_role.values():
        if group:
            forms.update(group)
    return {str(form).strip().upper() for form in forms if str(form).strip()}

def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)

    diag_cfg = cfg.get("Diagnostics") or {}
    if not isinstance(diag_cfg, dict):
        diag_cfg = {}
    diag_cfg.setdefault(
        "Path", os.path.join(cfg.get("Paths", {}).get("data", "data"), "run_diagnostics.jsonl")
    )
    diag_cfg.setdefault("Enabled", False)
    cfg["Diagnostics"] = diag_cfg

    return cfg

def save_config(cfg: Dict[str, Any], path: str = "config.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def filings_form_lookbacks(cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Return per-form lookback days from ``cfg['FilingsLookbacks']``.

    Expected structure in config.json::

        "FilingsLookbacks": {
          "10-Q": {"lookback_days": 180},
          "8-K": {"lookback_days": 400}
        }
    """
    fl = cfg.get("FilingsLookbacks", {}) or {}
    out: Dict[str, int] = {}

    for form, entry in fl.items():
        key = str(form).strip().upper()
        if not key:
            continue
        if isinstance(entry, dict):
            days = entry.get("lookback_days")
        else:
            days = entry
        if days is None:
            continue
        try:
            out[key] = int(days)
        except Exception:
            continue
    return out


def filings_max_lookback(cfg: Dict[str, Any], default: int = 60) -> int:
    mapping = filings_form_lookbacks(cfg)
    if not mapping:
        return default
    return max(mapping.values())
