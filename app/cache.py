import os
import pandas as pd
from datetime import datetime, timedelta

def _csv_path(cfg: dict, name: str) -> str:
    return os.path.join(cfg["Paths"]["data"], f"{name}.csv")

def _load_csv(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        # return empty frame with expected columns
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path, encoding="utf-8")

def append_antijoin_purge(
    cfg: dict,
    name: str,
    df_new: pd.DataFrame,
    key_cols: list[str],
    keep_days: int | None = None,
    date_col: str = "Date"
) -> int:
    """
    Guarantees CSV exists with headers even if df_new is empty.

    Steps:
    - Load old cache (or create empty with same columns)
    - Coerce key columns in BOTH frames to text to avoid int/str mismatch
    - Anti-join on key_cols
    - Append only new rows
    - Purge based on rolling window if keep_days is set
    - Save back to disk with headers
    - Return how many new rows we added
    """
    path = _csv_path(cfg, name)

    # Make sure df_new has stable columns (even if df_new is empty)
    new_cols = df_new.columns.tolist()
    df_old = _load_csv(path, columns=new_cols)

    # --- normalize key cols to text in BOTH frames so merge doesn't blow up ---
    for k in key_cols:
        if k in df_new.columns:
            df_new[k] = df_new[k].astype(str).fillna("")
        if k in df_old.columns:
            df_old[k] = df_old[k].astype(str).fillna("")

    # Anti-join to find truly new rows
    if df_old.empty:
        df_unique = df_new.copy()
    else:
        marker = df_new.merge(
            df_old[key_cols].drop_duplicates(),
            on=key_cols,
            how="left",
            indicator=True
        )
        df_unique = marker[marker["_merge"] == "left_only"].drop(columns=["_merge"])

    # Build combined without concat([empty, empty])
    if df_old.empty and df_unique.empty:
        df_all = df_old.copy()
    elif df_old.empty:
        df_all = df_unique.copy()
    elif df_unique.empty:
        df_all = df_old.copy()
    else:
        df_all = pd.concat([df_old, df_unique], ignore_index=True)

    # Rolling purge if requested
    if keep_days is not None and date_col in df_all.columns:
        cutoff_date = (datetime.utcnow() - timedelta(days=keep_days)).date().isoformat()
        if not df_all[date_col].isna().all():
            df_all = df_all[df_all[date_col].fillna(cutoff_date) >= cutoff_date]

    # Always write back with headers
    df_all.to_csv(path, index=False, encoding="utf-8")

    # Rows added this run
    return 0 if df_unique.empty else len(df_unique)

