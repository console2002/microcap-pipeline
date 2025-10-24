import os, json, time

def lock_path(cfg: dict) -> str:
    return os.path.join(cfg["Paths"]["data"], "run.lock")

def is_locked(cfg: dict) -> bool:
    return os.path.exists(lock_path(cfg))

def create_lock(cfg: dict, run_type: str):
    os.makedirs(cfg["Paths"]["data"], exist_ok=True)
    with open(lock_path(cfg), "w", encoding="utf-8") as f:
        json.dump({"run_type": run_type, "started": time.time()}, f)

def clear_lock(cfg: dict):
    try:
        os.remove(lock_path(cfg))
    except FileNotFoundError:
        pass
