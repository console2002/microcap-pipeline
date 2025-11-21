import os
import pandas as pd

from app.csv_names import csv_path


def _normalize_cik(series: pd.Series) -> pd.Series:
    """Normalize CIK values while preserving missing entries."""

    if series.empty:
        return series

    normalized = series.astype("string").str.strip()
    normalized = normalized.replace("", pd.NA)
    normalized = normalized.str.replace(r"\.0+$", "", regex=True)
    return normalized.str.zfill(10)


def hydrate_candidates(cfg: dict) -> pd.DataFrame:
    base = cfg["Paths"]["data"]

    # load caches
    profiles_path = csv_path(base, "profiles")
    prices_path = csv_path(base, "prices")
    filings_path = csv_path(base, "filings")
    fda_path = csv_path(base, "fda")

    profiles = pd.read_csv(profiles_path, encoding="utf-8") if os.path.exists(profiles_path) else pd.DataFrame()
    prices   = pd.read_csv(prices_path,   encoding="utf-8") if os.path.exists(prices_path)   else pd.DataFrame()
    filings  = pd.read_csv(filings_path,  encoding="utf-8") if os.path.exists(filings_path)  else pd.DataFrame()
    fda      = pd.read_csv(fda_path,      encoding="utf-8") if os.path.exists(fda_path)      else pd.DataFrame()

    if not profiles.empty and "CIK" in profiles.columns:
        profiles["CIK"] = profiles["CIK"].fillna("").astype(str).str.zfill(10).str.strip()

    # latest price + ADV20
    if not prices.empty:
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        prices = prices.sort_values(["Ticker","Date"])
        prices["ADV20"] = prices.groupby("Ticker")["Volume"].transform(lambda s: s.rolling(20).mean())
        latest_px = prices.groupby("Ticker").tail(1)[["Ticker","Date","Close","ADV20"]]
    else:
        latest_px = pd.DataFrame(columns=["Ticker","Date","Close","ADV20"])

    def _combine_entries(series: pd.Series) -> str:
        entries: list[str] = []
        for val in series:
            if pd.isna(val):
                continue
            text = str(val).strip()
            if text:
                entries.append(text)
        return "\n".join(entries)

    def _combine_unique(series: pd.Series) -> str:
        seen: list[str] = []
        for val in series:
            if pd.isna(val):
                continue
            text = str(val).strip()
            if not text:
                continue
            if text not in seen:
                seen.append(text)
        return "\n".join(seen)

    # latest filing per CIK, plus aggregate evidence
    if not filings.empty and "CIK" in filings.columns:
        filings["CIK"] = _normalize_cik(filings["CIK"])
        filings["FiledAt"] = pd.to_datetime(filings["FiledAt"], errors="coerce")
        latest_fil = (
            filings.sort_values(["CIK","FiledAt"])
                   .groupby("CIK")
                   .tail(1)[["CIK","Form","FiledAt","URL"]]
        )

        def _format_filing(row: pd.Series) -> str:
            parts: list[str] = []
            filed_val = row.get("FiledAt")
            if pd.notna(filed_val):
                if isinstance(filed_val, pd.Timestamp):
                    parts.append(filed_val.strftime("%Y-%m-%d"))
                else:
                    parts.append(str(filed_val))
            form_txt = str(row.get("Form", "") or "").strip()
            if form_txt:
                parts.append(form_txt)
            url_txt = str(row.get("URL", "") or "").strip()
            if url_txt:
                parts.append(url_txt)
            base_text = " | ".join(parts)
            if not base_text:
                return base_text
            desc_txt = str(row.get("Desc", "") or "").strip()
            if desc_txt:
                return f"{base_text} | {desc_txt}"
            return base_text

        filings_sorted = filings.sort_values(["CIK", "FiledAt"], ascending=[True, False])
        filings_sorted["FilingEntry"] = filings_sorted.apply(_format_filing, axis=1)
        filings_evidence = (
            filings_sorted.groupby("CIK")
                           .agg(
                               FilingsSummary=("FilingEntry", _combine_entries),
                               FilingURLsAll=("URL", _combine_unique)
                           )
                           .reset_index()
        )
    else:
        latest_fil = pd.DataFrame(columns=["CIK","Form","FiledAt","URL"])
        filings_evidence = pd.DataFrame(columns=["CIK","FilingsSummary","FilingURLsAll"])

    # latest FDA per CIK/Company (stub mapping: match on CIK first)
    if not fda.empty and "CIK" in fda.columns:
        fda["CIK"] = _normalize_cik(fda["CIK"])
        fda["DecisionDate"] = pd.to_datetime(fda["DecisionDate"], errors="coerce")
        latest_fda = (
            fda.sort_values(["CIK","DecisionDate"])
               .groupby("CIK")
               .tail(1)[["CIK","EventType","DecisionDate","URL"]]
        )

        def _format_fda(row: pd.Series) -> str:
            parts: list[str] = []
            decision_val = row.get("DecisionDate")
            if pd.notna(decision_val):
                if isinstance(decision_val, pd.Timestamp):
                    parts.append(decision_val.strftime("%Y-%m-%d"))
                else:
                    parts.append(str(decision_val))
            event_txt = str(row.get("EventType", "") or "").strip()
            if event_txt:
                parts.append(event_txt)
            product_txt = str(row.get("Product", "") or "").strip()
            if product_txt:
                parts.append(product_txt)
            url_txt = str(row.get("URL", "") or "").strip()
            if url_txt:
                parts.append(url_txt)
            return " | ".join(parts)

        fda_sorted = fda.sort_values(["CIK", "DecisionDate"], ascending=[True, False])
        fda_sorted["FDAEntry"] = fda_sorted.apply(_format_fda, axis=1)
        fda_evidence = (
            fda_sorted.groupby("CIK")
                      .agg(
                          FDA_Summary=("FDAEntry", _combine_entries),
                          FDA_URLsAll=("URL", _combine_unique)
                      )
                      .reset_index()
        )
    else:
        latest_fda = pd.DataFrame(columns=["CIK","EventType","DecisionDate","URL"])
        fda_evidence = pd.DataFrame(columns=["CIK","FDA_Summary","FDA_URLsAll"])

    # join profiles with latest px
    cand = profiles.merge(
        latest_px.rename(columns={
            "Date":"PriceDate"
        }),
        on="Ticker",
        how="left",
    )

    # add latest SEC filing
    cand = cand.merge(
        latest_fil.rename(columns={
            "Form":"LatestForm",
            "URL":"FilingURL"
        }),
        on="CIK",
        how="left",
    )

    # add latest FDA event
    cand = cand.merge(
        latest_fda.rename(columns={
            "EventType":"FDA_EventType",
            "DecisionDate":"FDA_Date",
            "URL":"FDA_URL"
        }),
        on="CIK",
        how="left",
    )

    # attach evidence detail columns
    cand = cand.merge(
        filings_evidence,
        on="CIK",
        how="left",
    )

    cand = cand.merge(
        fda_evidence,
        on="CIK",
        how="left",
    )

    # final shape
    wanted_cols = [
        "Ticker","Company","Sector","CIK","MarketCap",
        "PriceDate","Close","ADV20",
        "LatestForm","FiledAt","FilingURL","FilingsSummary","FilingURLsAll",
        "FDA_EventType","FDA_Date","FDA_URL","FDA_Summary","FDA_URLsAll"
    ]
    for col in wanted_cols:
        if col not in cand.columns:
            cand[col] = None

    result = cand[wanted_cols].copy()

    if "Ticker" in result.columns:
        ticker_series = result["Ticker"].astype("string")
        valid_mask = ticker_series.notna() & ticker_series.str.strip().ne("")
        result = result.loc[valid_mask].copy()
        result["Ticker"] = ticker_series.loc[valid_mask].str.strip()

    return result
