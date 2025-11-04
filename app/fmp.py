from datetime import date, timedelta, datetime
from typing import Callable, Optional
import re
from app.http import HttpClient
from app.cancel import CancelledRun
from app.config import filings_form_lookbacks, filings_max_lookback
from app.universe_filters import load_drop_filters, should_drop_record

FMP_HOST = "https://financialmodelingprep.com"


def normalize_exchange(raw: str) -> str:
    if not raw:
        return ""

    cleaned = raw.strip()
    if not cleaned:
        return ""

    upper = cleaned.upper()

    mapping = {
        "NASDAQ": "NASDAQ",
        "NASDAQGS": "NASDAQ",
        "NASDAQGM": "NASDAQ",
        "NASDAQCM": "NASDAQ",
        "NYSE": "NYSE",
        "NYSE AMERICAN": "NYSEAM",
        "AMEX": "NYSEAM",
    }

    return mapping.get(upper, upper)


def _check_cancel(stop_flag: Optional[dict]):
    if stop_flag and stop_flag.get("stop"):
        raise CancelledRun("cancel requested")


def _chunk(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def _normalize_sec_url(url: str) -> str:
    if not url:
        return ""

    cleaned = url.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("http://"):
        cleaned = "https://" + cleaned[len("http://"):]

    lower = cleaned.lower()

    if lower.startswith("https://sec.gov") and not lower.startswith("https://www.sec.gov"):
        cleaned = "https://www.sec.gov" + cleaned[len("https://sec.gov"):]
        lower = cleaned.lower()

    if lower.startswith("https://www.sec.gov/ix?doc="):
        return cleaned

    if lower.startswith("/archives/") or lower.startswith("archives/"):
        path = cleaned.lstrip("/")
        return f"https://www.sec.gov/ix?doc=/{path}"

    if (
        lower.startswith("https://www.sec.gov/archives/")
        and (
            lower.endswith(".htm")
            or lower.endswith(".html")
            or lower.endswith(".xhtml")
        )
        and not lower.endswith("-index.htm")
        and not lower.endswith("-index.html")
        and not lower.endswith("-index.xhtml")
    ):
        path = cleaned[len("https://www.sec.gov"):]
        if not path.startswith("/"):
            path = "/" + path
        return f"https://www.sec.gov/ix?doc={path}"

    return cleaned


def _extract_filing_description(rec: dict) -> str:
    if not isinstance(rec, dict):
        return ""

    candidates = (
        rec.get("items"),
        rec.get("documentTitle"),
        rec.get("primaryDocDescription"),
        rec.get("primaryDocumentDescription"),
        rec.get("description"),
    )

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            parts = [str(part).strip() for part in candidate if part]
            candidate_text = " ".join(part for part in parts if part)
        else:
            candidate_text = str(candidate)

        candidate_text = re.sub(r"\s+", " ", candidate_text or "").strip()
        if candidate_text:
            return candidate_text[:120]

    return ""


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
    substring_patterns, word_patterns = load_drop_filters(cfg)

    batches = list(_chunk(tickers, bs))
    total_batches = len(batches)

    allowed_exchange_norms: set[str] = set()
    for exch in cfg["Universe"]["Exchanges"] or []:
        norm = normalize_exchange(exch)
        if norm:
            allowed_exchange_norms.add(norm)

    default_exchange_norms = {"NASDAQ", "NYSE", "NYSEAM"}

    otc_pattern = re.compile(r"(?i)\b(?:OTC|OTCQX|OTCQB|PINK|GREY)\b")

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
            exch_norm = normalize_exchange(exchange)
            company  = rec.get("companyName") or ""
            country  = rec.get("country") or ""
            price    = rec.get("price")
            mcap     = rec.get("mktCap") or rec.get("marketCap")

            if should_drop_record(company, ticker, substring_patterns, word_patterns):
                continue

            # hard gate filters
            if otc_pattern.search(exchange):
                continue

            if allowed_exchange_norms:
                if exch_norm not in allowed_exchange_norms:
                    continue
            else:
                if exch_norm not in default_exchange_norms:
                    continue
            if price is not None and price < cfg["HardGates"]["MinPrice"]:
                continue
            if mcap is not None and mcap < cfg["HardGates"]["CapMin"]:
                continue
            if mcap is not None and mcap > cfg["HardGates"]["CapMax"]:
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

    key = cfg["FMPKey"]
    whitelist = [str(x).strip().upper() for x in cfg.get("FilingsWhitelist", [])]
    wl_forms = {form for form in whitelist if form}
    whitelist_ordered = sorted(wl_forms, key=len, reverse=True)
    form_lookbacks = filings_form_lookbacks(cfg)
    max_lookback = filings_max_lookback(cfg)
    ratelimit = cfg["RateLimitsPerMin"]["FMP"]
    today = date.today()
    cutoff_cache: dict[int, date] = {}

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
            form_upper = (form_raw or "").strip().upper()
            if not form_upper:
                continue

            # loose match: allow prefixes (e.g., "8-K/A" matches "8-K")
            matched_prefix = next(
                (prefix for prefix in whitelist_ordered if form_upper.startswith(prefix)),
                None,
            )
            if not matched_prefix:
                continue

            filed_raw = rec.get("filingDate") or rec.get("fillingDate") or ""
            try:
                filed_date = datetime.strptime(filed_raw[:10], "%Y-%m-%d").date()
            except Exception:
                filed_date = None
            if not filed_date:
                continue

            lookback_days = form_lookbacks.get(matched_prefix, max_lookback)
            if lookback_days and lookback_days > 0:
                cutoff = cutoff_cache.setdefault(
                    lookback_days, today - timedelta(days=lookback_days)
                )
                if filed_date < cutoff:
                    continue

            cik_val = rec.get("cik") or ""
            cik_txt = str(cik_val).zfill(10) if cik_val else ""

            source_url = (
                rec.get("finalLink")
                or rec.get("linkToFilingDetails")
                or rec.get("reportUrl")
                or rec.get("link")
                or ""
            )

            desc_text = _extract_filing_description(rec)

            out.append({
                "CIK": cik_txt,
                "Ticker": (rec.get("symbol") or "").upper(),
                "Company": rec.get("companyName") or "",
                "Form": form_raw,
                "FiledAt": filed_raw,
                "URL": _normalize_sec_url(source_url),
                "Desc": desc_text,
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
