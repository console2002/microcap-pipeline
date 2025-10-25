import pandas as pd

def build_shortlist(cfg: dict, candidates: pd.DataFrame) -> pd.DataFrame:
    hg = cfg["HardGates"]
    rule = cfg["ShortlistRule"]

    df = candidates.copy()

    # catalyst flags
    has_sec = df["LatestForm"].notna() & df["LatestForm"].astype(str).ne("")
    has_fda = df["FDA_EventType"].notna() & df["FDA_EventType"].astype(str).ne("")

    # core filters
    mask = (
        (df["Close"] >= hg["MinPrice"]) &
        (df["MarketCap"] >= hg["CapMin"]) &
        (df["MarketCap"] <= hg["CapMax"]) &
        (df["ADV20"] >= hg["ADV20_Min"]) &
        (
            (rule["UseFilingsCatalyst"] & has_sec) |
            (rule["UseFDACatalyst"] & has_fda)
        )
    )

    out = df.loc[mask, [
        "Ticker","Company","Sector","CIK",
        "Close","ADV20","MarketCap",
        "LatestForm","FiledAt","FilingURL","FilingsSummary","FilingURLsAll",
        "FDA_EventType","FDA_Date","FDA_URL","FDA_Summary","FDA_URLsAll"
    ]].sort_values(["Ticker"])

    return out
