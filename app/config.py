import json, os
from typing import Any, Dict

def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    os.makedirs(cfg["Paths"]["logs"], exist_ok=True)

    return cfg

def save_config(cfg: Dict[str, Any], path: str = "config.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
