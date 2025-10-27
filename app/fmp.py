from datetime import date, timedelta, datetime
from typing import Callable, Optional
from app.http import HttpClient
from app.cancel import CancelledRun

FMP_HOST = "https://financialmodelingprep.com"


def _check_cancel(stop_flag: Optional[dict]):
    if stop_flag and stop_flag.get("stop"):
        raise CancelledRun("cancel requested")


def _chunk(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def fetch_profiles(
    client: HttpClient,
    cfg: dict,
    tickers: list[str],
    progress_fn: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[dict] = None
) -> list[dict]:
    """
    Pull company profile data from FMP in batches of BatchSizes.Profiles.

    Before:
      we fanned out concurrent futures for all batches.
      That could spike call volume and made cancel sluggish.

    Now:
      we loop sequentially, honoring rate limit + stop_flag,
      which keeps calls sane and lets cancel stop immediately.
    """

    out = []

    key       = cfg["FMPKey"]
    bs        = cfg["BatchSizes"]["Profiles"]
    ratelimit = cfg["RateLimitsPerMin"]["FMP"]

    batches = list(_chunk(tickers, bs))
    total_batches = len(batches)

    for idx, batch in enumerate(batches, start=1):
        _check_cancel(stop_flag)

        symbols_csv = ",".join(batch)
        url = f"{FMP_HOST}/api/v3/profile/{symbols_csv}"
        resp = client.get(url, {"apikey": key}, ratelimit)

        # report progress after each batch
        if progress_fn:
            pct = int((idx / max(total_batches, 1)) * 100)
            progress_fn(f"[profiles] {idx}/{total_batches} batches ({pct}%) {client.stats_string()}")

        data = resp.json() or []
        for rec in data:
            ticker = (rec.get("symbol") or "").upper()
            if ticker.endswith(".US"):
                ticker = ticker[:-3]

            cik_val = rec.get("cik") or ""
            cik_txt = str(cik_val).zfill(10) if cik_val else ""

            exchange = rec.get("exchangeShortName") or rec.get("exchange") or ""
            company  = rec.get("companyName") or ""
            country  = rec.get("country") or ""
            price    = rec.get("price")
            mcap     = rec.get("mktCap") or rec.get("marketCap")

            # hard gate filters
            if cfg["Universe"]["Exchanges"] and exchange not in cfg["Universe"]["Exchanges"]:
                continue
            if price is not None and price < cfg["HardGates"]["MinPrice"]:
                continue
            if mcap is not None and mcap < cfg["HardGates"]["CapMin"]:
                continue
            if mcap is not None and mcap > cfg["HardGates"]["CapMax"]:
                continue
            if country:
                cu = country.upper()
                if cu not in ("US", "USA", "UNITED STATES"):
                    if exchange not in cfg["Universe"]["Exchanges"]:
                        continue

            out.append({
                "Ticker": ticker,
                "CIK": cik_txt,
                "Company": company,
                "Exchange": exchange,
                "Country": country,
                "Currency": rec.get("currency") or "",
                "Sector": rec.get("sector") or "",
                "Industry": rec.get("industry") or "",
                "MarketCap": mcap,
                "Price": price,
                "UpdatedAt": datetime.utcnow().isoformat()
            })

    return out


def fetch_filings(
    client: HttpClient,
    cfg: dict,
    tickers: list[str],
    progress_fn: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[dict] = None
) -> list[dict]:
    """
    Pull SEC filings per ticker.
    Uses 'type' field (not 'form') and performs a prefix match
    so '8-K/A' counts as '8-K', etc.
    """

    out = []

    key       = cfg["FMPKey"]
    wl_forms  = set(cfg["FilingsWhitelist"])
    days_back = cfg["Windows"]["DaysBack_Filings"]
    cutoff    = date.today() - timedelta(days=days_back)
    ratelimit = cfg["RateLimitsPerMin"]["FMP"]

    total = len(tickers)

    for idx, tck in enumerate(tickers, start=1):
        # cancel check
        if stop_flag and stop_flag.get("stop"):
            raise CancelledRun("cancel requested during filings")

        url = f"{FMP_HOST}/api/v3/sec_filings/{tck}"
        resp = client.get(url, {"apikey": key}, ratelimit)
        data = resp.json() or []

        # progress update every 25 tickers
        if progress_fn and (idx % 25 == 0 or idx == total):
            pct = int((idx / max(total, 1)) * 100)
            progress_fn(f"[filings] {idx}/{total} tickers ({pct}%) {client.stats_string()}")

        for rec in data:
            # --- use 'type' field instead of 'form'
            form_raw = rec.get("type", "") or rec.get("form", "")
            # loose match: allow prefixes (e.g., "8-K/A" matches "8-K")
            if not any(form_raw.startswith(x) for x in wl_forms):
                continue

            filed_raw = rec.get("filingDate") or rec.get("fillingDate") or ""
            try:
                filed_date = datetime.strptime(filed_raw[:10], "%Y-%m-%d").date()
            except Exception:
                filed_date = None
            if not filed_date or filed_date < cutoff:
                continue

            cik_val = rec.get("cik") or ""
            cik_txt = str(cik_val).zfill(10) if cik_val else ""

            out.append({
                "CIK": cik_txt,
                "Ticker": (rec.get("symbol") or "").upper(),
                "Company": rec.get("companyName") or "",
                "Form": form_raw,
                "FiledAt": filed_raw,
                "URL": rec.get("link") or ""
            })

    return out



def fetch_prices(
    client: HttpClient,
    cfg: dict,
    tickers: list[str],
    progress_fn: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[dict] = None
) -> list[dict]:
    """
    Call FMP /historical-price-full/{ticker}?timeseries=...

    Like filings, FMP only exposes one-symbol historical pull,
    so it's 1 request per ticker. We now run sequentially.

    We emit progress every 25 tickers (and final).
    """

    out = []

    key       = cfg["FMPKey"]
    ratelimit = cfg["RateLimitsPerMin"]["FMP"]
    days      = cfg["Windows"]["DaysBack_Prices"]

    total = len(tickers)

    for idx, tck in enumerate(tickers, start=1):
        _check_cancel(stop_flag)

        url    = f"{FMP_HOST}/api/v3/historical-price-full/{tck}"
        # NOTE:
        #   FMP's `serietype=line` parameter trims the payload down to
        #   date/close-only entries.  That was unintentionally stripping the
        #   open/high/low/volume fields we persist to prices.csv.  By omitting
        #   the parameter we receive the full OHLCV payload again.
        params = {"apikey": key, "timeseries": days}
        resp   = client.get(url, params, ratelimit)

        if progress_fn and (idx % 25 == 0 or idx == total):
            pct = int((idx / max(total, 1)) * 100)
            progress_fn(f"[prices] {idx}/{total} tickers ({pct}%) {client.stats_string()}")

        rj = resp.json() or {}
        t = (rj.get("symbol") or rj.get("ticker") or "").upper()
        if t.endswith(".US"):
            t = t[:-3]

        hist = rj.get("historical") or []
        for h in hist:
            out.append({
                "Date":   h.get("date"),
                "Ticker": t,
                "Open":   h.get("open"),
                "High":   h.get("high"),
                "Low":    h.get("low"),
                "Close":  h.get("close"),
                "Volume": h.get("volume")
            })

    return out
