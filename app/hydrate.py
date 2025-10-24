import os
import pandas as pd

def hydrate_candidates(cfg: dict) -> pd.DataFrame:
    base = cfg["Paths"]["data"]

    # load caches
    profiles_path = os.path.join(base, "profiles.csv")
    prices_path   = os.path.join(base, "prices.csv")
    filings_path  = os.path.join(base, "filings.csv")
    fda_path      = os.path.join(base, "fda.csv")

    profiles = pd.read_csv(profiles_path, encoding="utf-8") if os.path.exists(profiles_path) else pd.DataFrame()
    prices   = pd.read_csv(prices_path,   encoding="utf-8") if os.path.exists(prices_path)   else pd.DataFrame()
    filings  = pd.read_csv(filings_path,  encoding="utf-8") if os.path.exists(filings_path)  else pd.DataFrame()
    fda      = pd.read_csv(fda_path,      encoding="utf-8") if os.path.exists(fda_path)      else pd.DataFrame()

    # latest price + ADV20
    if not prices.empty:
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        prices = prices.sort_values(["Ticker","Date"])
        prices["ADV20"] = prices.groupby("Ticker")["Volume"].transform(lambda s: s.rolling(20).mean())
        latest_px = prices.groupby("Ticker").tail(1)[["Ticker","Date","Close","ADV20"]]
    else:
        latest_px = pd.DataFrame(columns=["Ticker","Date","Close","ADV20"])

    # latest filing per CIK
    if not filings.empty:
        filings["FiledAt"] = pd.to_datetime(filings["FiledAt"], errors="coerce")
        latest_fil = (
            filings.sort_values(["CIK","FiledAt"])
                   .groupby("CIK")
                   .tail(1)[["CIK","Form","FiledAt","URL"]]
        )
    else:
        latest_fil = pd.DataFrame(columns=["CIK","Form","FiledAt","URL"])

    # latest FDA per CIK/Company (stub mapping: match on CIK first)
    if not fda.empty:
        fda["DecisionDate"] = pd.to_datetime(fda["DecisionDate"], errors="coerce")
        # prefer CIK match if present
        if "CIK" in fda.columns:
            latest_fda = (
                fda.sort_values(["CIK","DecisionDate"])
                   .groupby("CIK")
                   .tail(1)[["CIK","EventType","DecisionDate","URL"]]
            )
        else:
            latest_fda = pd.DataFrame(columns=["CIK","EventType","DecisionDate","URL"])
    else:
        latest_fda = pd.DataFrame(columns=["CIK","EventType","DecisionDate","URL"])

    # join profiles with latest px
    cand = profiles.merge(
        latest_px.rename(columns={
            "Date":"PriceDate"
        }),
        on="Ticker",
        how="left"
    )

    # add latest SEC filing
    cand = cand.merge(
        latest_fil.rename(columns={
            "Form":"LatestForm",
            "URL":"FilingURL"
        }),
        on="CIK",
        how="left"
    )

    # add latest FDA event
    cand = cand.merge(
        latest_fda.rename(columns={
            "EventType":"FDA_EventType",
            "DecisionDate":"FDA_Date",
            "URL":"FDA_URL"
        }),
        on="CIK",
        how="left"
    )

    # final shape
    wanted_cols = [
        "Ticker","Company","CIK","MarketCap",
        "PriceDate","Close","ADV20",
        "LatestForm","FiledAt","FilingURL",
        "FDA_EventType","FDA_Date","FDA_URL"
    ]
    for col in wanted_cols:
        if col not in cand.columns:
            cand[col] = None

    return cand[wanted_cols]
