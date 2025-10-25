from typing import Optional, Callable, List
from app.http import HttpClient
from app.cancel import CancelledRun
from app.fda_company_map import get_fda_entities_for_tickers


def _check_cancel(stop_flag: Optional[dict]):
    if stop_flag and stop_flag.get("stop"):
        raise CancelledRun("cancel requested during FDA stage")


def _safe_openfda_get(client: HttpClient, url: str, params: dict, per_minute: int):
    """
    openFDA returns 404 when there are no results. Our HttpClient.get()
    would raise on >=400. We treat 404 or 400 as 'no data' instead of hard error.
    Anything else (5xx etc.) will still retry inside HttpClient.get().
    """
    import requests

    host = requests.utils.urlparse(url).netloc
    limiter = client._limiter_for(host, per_minute)  # reuse limiter/bookkeeping
    limiter.acquire()
    client.call_counts[host] = client.call_counts.get(host, 0) + 1
    client.host_limits[host] = per_minute

    try:
        r = client.session.get(url, params=params, timeout=client.timeout)
    except (requests.Timeout, requests.ConnectionError) as e:
        # treat network fail as no results for now
        return None

    if r.status_code in (400, 404):
        return None
    if r.status_code >= 500:
        # server side issue -> just treat as no data for now instead of killing run
        return None
    if r.status_code >= 400:
        # any other 4xx we consider 'no data'
        return None

    return r


def _normalize_device_results(results: List[dict], company: str, cik: str, ticker: str) -> List[dict]:
    """
    Results from /device/510k.json
    We emit minimal schema we care about.
    """
    out = []
    for rec in results:
        k_no = rec.get("k_number") or ""
        decision_date = rec.get("decision_date") or ""
        device_name = rec.get("device_name") or ""
        link = rec.get("link") or rec.get("decision_description") or ""

        # Build a deterministic EventID
        event_id = f"510k:{k_no}"

        out.append({
            "EventID": event_id,
            "ApplicantName": company,
            "EventType": "510k",
            "DecisionDate": decision_date,
            "Product": device_name,
            "URL": link,
            "CIK": cik,
            "Ticker": ticker
        })
    return out


def _normalize_drug_results(results: List[dict], company: str, cik: str, ticker: str) -> List[dict]:
    """
    Results from /drug/drugsfda.json
    Structure is nested: results -> applications -> products
    We'll flatten applications/products into individual rows.
    """
    out = []
    for top in results:
        apps = top.get("applications") or []
        for app in apps:
            app_no = app.get("application_number") or ""
            sponsor = app.get("sponsor_name") or company
            approval_date = app.get("approval_date") or app.get("original_approval_date") or ""
            prods = app.get("products") or []

            for p in prods:
                brand = p.get("brand_name") or ""
                # Deterministic EventID: application_number + brand
                event_id = f"drug:{app_no}:{brand}"

                out.append({
                    "EventID": event_id,
                    "ApplicantName": sponsor,
                    "EventType": "drug_approval",
                    "DecisionDate": approval_date,
                    "Product": brand,
                    "URL": "",  # drugsfda doesn't really give a direct link per product
                    "CIK": cik,
                    "Ticker": ticker
                })
    return out


def fetch_fda_events(
    client: HttpClient,
    cfg: dict,
    df_filings,  # DataFrame of recent filings (ticker/company/CIK)
    progress_fn: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[dict] = None
) -> List[dict]:
    """
    Core FDA fetch logic.

    - Take unique (Company, CIK, Ticker) from recent filings only.
    - For each company:
        - If FDA.EnableDevice -> query 510k
        - If FDA.EnableDrug   -> query drugsfda
    - Limit=1000, use applicant:"Company Name"
    - Respect rate limit cfg["RateLimitsPerMin"]["OpenFDA"]
    - Cancel-aware and progress-aware.
    - Return list of normalized rows.
    """

    out_rows: List[dict] = []

    if df_filings is None or df_filings.empty:
        if progress_fn:
            progress_fn("[FDA] aborting fetch – no filings rows provided")
        return out_rows

    if not (cfg["FDA"]["EnableDevice"] or cfg["FDA"]["EnableDrug"]):
        if progress_fn:
            progress_fn("[FDA] both device/drug lookups disabled; skipping")
        return out_rows  # nothing requested

    # Build mapping of tickers to FDA entity names for high/med confidence targets.
    # This ensures we query openFDA using the entity strings it expects.
    filing_tickers = (
        df_filings[["Ticker"]]
        .dropna()
        .assign(Ticker=lambda df: df["Ticker"].astype(str).str.strip())
        .loc[lambda df: df["Ticker"] != ""]
        .drop_duplicates()["Ticker"]
        .tolist()
    )

    if progress_fn:
        progress_fn(f"[FDA] preparing targets for {len(filing_tickers)} filing tickers")

    mapped_targets = []
    seen_entities = set()
    for entry in get_fda_entities_for_tickers(filing_tickers):
        entity_name = str(entry.get("FDA_EntityName", "")).strip()
        if not entity_name or entity_name in seen_entities:
            continue

        mapped_targets.append(
            {
                "Company": entity_name,
                "CIK": str(entry.get("CIK", "")).strip(),
                "Ticker": str(entry.get("Ticker", "")).strip(),
            }
        )
        seen_entities.add(entity_name)

    # Include fallback to the company names from SEC filings for any tickers
    # without an explicit FDA mapping so we keep legacy coverage.
    sec_targets = (
        df_filings[["Company", "CIK", "Ticker"]]
        .dropna(subset=["Company"])
        .assign(Company=lambda df: df["Company"].astype(str).str.strip())
        .assign(CIK=lambda df: df["CIK"].astype(str).str.strip())
        .assign(Ticker=lambda df: df["Ticker"].astype(str).str.strip())
        .loc[lambda df: df["Company"] != ""]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    targets = mapped_targets[:]
    mapped_tickers = {t["Ticker"] for t in mapped_targets if t.get("Ticker")}

    for row in sec_targets:
        ticker = row.get("Ticker", "")
        if ticker in mapped_tickers:
            continue
        company = row.get("Company", "")
        if company in seen_entities:
            continue
        targets.append(row)
        seen_entities.add(company)

    total = len(targets)

    if progress_fn:
        progress_fn(
            f"[FDA] mapped {len(mapped_targets)} tickers; {len(sec_targets)} fallback companies; total targets {total}"
        )

    if total == 0:
        if progress_fn:
            progress_fn("[FDA] no companies to query after mapping; exiting")
        return out_rows
    per_minute = cfg["RateLimitsPerMin"]["OpenFDA"]
    api_key = cfg.get("OpenFDAKey", "")

    fetched_device = 0
    fetched_drug = 0

    for i, row in enumerate(targets, start=1):
        _check_cancel(stop_flag)

        company = str(row.get("Company", "")).strip()
        cik     = str(row.get("CIK", "")).strip()
        ticker  = str(row.get("Ticker", "")).strip()

        if progress_fn and (i % 10 == 0 or i == total):
            pct = int((i / max(total, 1)) * 100)
            progress_fn(
                f"[FDA] {i}/{total} companies ({pct}%) {company} {client.stats_string()}"
            )

        # Query device 510(k) if enabled
        device_rows = []
        if cfg["FDA"]["EnableDevice"]:
            params_device = {
                "search": f'applicant:"{company}"',
                "limit": "1000"
            }
            if api_key:
                params_device["api_key"] = api_key

            r_dev = _safe_openfda_get(
                client,
                "https://api.fda.gov/device/510k.json",
                params_device,
                per_minute
            )
            if r_dev:
                body = r_dev.json()
                dev_results = body.get("results") or []
                device_rows = _normalize_device_results(dev_results, company, cik, ticker)
                out_rows.extend(device_rows)

        _check_cancel(stop_flag)

        # Query drug approval data if enabled
        drug_rows = []
        if cfg["FDA"]["EnableDrug"]:
            params_drug = {
                "search": f'applicant:"{company}"',
                "limit": "1000"
            }
            if api_key:
                params_drug["api_key"] = api_key

            r_drug = _safe_openfda_get(
                client,
                "https://api.fda.gov/drug/drugsfda.json",
                params_drug,
                per_minute
            )
            if r_drug:
                body = r_drug.json()
                drug_results = body.get("results") or []
                drug_rows = _normalize_drug_results(drug_results, company, cik, ticker)
                out_rows.extend(drug_rows)

        _check_cancel(stop_flag)

        fetched_device += len(device_rows)
        fetched_drug += len(drug_rows)

        if progress_fn:
            progress_fn(
                (
                    f"[FDA] company {company or ticker}: "
                    f"device_rows={len(device_rows)} "
                    f"drug_rows={len(drug_rows)} {client.stats_string()}"
                )
            )

    if progress_fn:
        progress_fn(
            (
                "[FDA] total events collected "
                f"device={fetched_device} "
                f"drug={fetched_drug} "
                f"overall={len(out_rows)} "
                f"{client.stats_string()}"
            )
        )

    return out_rows
