import json
from app.http import HttpClient
from app.universe_filters import load_drop_filters, should_drop_record

SEC_URL = "https://www.sec.gov/files/company_tickers.json"

def load_sec_universe(client: HttpClient, cfg: dict) -> list[dict]:
    r = client.get(
        SEC_URL,
        params=None,
        per_minute=cfg["RateLimitsPerMin"]["SEC"]
    )
    data = json.loads(r.text)

    # SEC structure: { "0": {"ticker":"A","cik_str":"0000320193","title":"Apple Inc."}, ... }
    records = []
    for _, row in data.items():
        ticker = (row.get("ticker") or "").upper()
        cik_raw = row.get("cik_str") or ""
        cik = str(cik_raw).zfill(10) if cik_raw else ""
        company = row.get("title") or ""

        if not ticker:
            continue

        records.append({
            "Ticker": ticker,
            "CIK": cik,
            "Company": company
        })

    # basic ticker cleanup (".US" etc.) if enabled
    if cfg["Universe"]["NormalizeTicker"]:
        for rec in records:
            if rec["Ticker"].endswith(".US"):
                rec["Ticker"] = rec["Ticker"][:-3]

    # drop obvious junk patterns
    substring_patterns, word_patterns = load_drop_filters(cfg)
    clean = []
    for rec in records:
        if should_drop_record(rec["Company"], rec["Ticker"], substring_patterns, word_patterns):
            continue
        clean.append(rec)

    return clean
