from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Optional

from edgar import Company, set_identity
from edgar.httprequests import download_text
from edgar.reference.tickers import get_company_tickers

from app.cancel import CancelledRun
from app.config import filings_form_lookbacks, filings_max_lookback, load_config
from app.rate_limit import RateLimiter
from app.universe_filters import load_drop_filters, should_drop_record


logger = logging.getLogger(__name__)

_ADAPTER: "EdgarAdapter" | None = None


def _normalize_cik(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(10)


def _normalize_ticker(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


class EdgarAdapter:
    """Central adapter for all EDGAR interactions."""

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        edgar_cfg = cfg.get("Edgar", {})
        self.forms_whitelist = [
            str(value).strip().upper()
            for value in cfg.get("FilingsWhitelist", [])
            if str(value).strip()
        ]
        self.form_lookbacks = filings_form_lookbacks(cfg)
        self.max_lookback = filings_max_lookback(cfg)

        throttle_per_min = edgar_cfg.get("ThrottlePerMin") or cfg.get(
            "RateLimitsPerMin", {}
        ).get("SEC")
        self.rate_limiter = RateLimiter(throttle_per_min) if throttle_per_min else None

        self._configure_identity(edgar_cfg.get("UserAgent") or cfg.get("UserAgent"))

    def _configure_identity(self, user_agent: Optional[str]) -> None:
        if not user_agent:
            logger.warning("EDGAR identity missing; requests may be rejected")
            return

        try:
            set_identity(user_agent)
            logger.info("Configured EDGAR identity for requests")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to configure EDGAR identity: %s", exc)

    def _rate_limit(self) -> None:
        if self.rate_limiter:
            self.rate_limiter.acquire()

    def load_company_universe(self) -> list[dict]:
        """Return SEC company universe via edgartools ticker dataset."""

        try:
            df = get_company_tickers()
        except Exception as exc:
            logger.error("Failed to load EDGAR ticker universe: %s", exc)
            return []

        substring_patterns, word_patterns = load_drop_filters(self.cfg)

        records: list[dict] = []
        for _, row in df.iterrows():
            ticker = _normalize_ticker(row.get("ticker") or row.get("symbol"))
            if not ticker:
                continue

            cik = _normalize_cik(row.get("cik_str") or row.get("cik") or row.get("CIK"))
            company = (row.get("title") or row.get("name") or "").strip()

            if self.cfg.get("Universe", {}).get("NormalizeTicker") and ticker.endswith(
                ".US"
            ):
                ticker = ticker[:-3]

            if should_drop_record(company, ticker, substring_patterns, word_patterns):
                continue

            records.append({"Ticker": ticker, "CIK": cik, "Company": company})

        return records

    def _is_within_lookback(self, form: str, filed_at: str) -> bool:
        form_upper = (form or "").strip().upper()
        if not form_upper or not filed_at:
            return False

        try:
            filed_date = datetime.strptime(str(filed_at)[:10], "%Y-%m-%d").date()
        except Exception:
            return False

        lookback_days = self.form_lookbacks.get(form_upper, self.max_lookback)
        if not lookback_days:
            return True

        cutoff = date.today() - timedelta(days=lookback_days)
        return filed_date >= cutoff

    def fetch_recent_filings(
        self,
        tickers: Iterable[str],
        progress_fn: Optional[Callable[[str], None]] = None,
        stop_flag: Optional[dict] = None,
        *,
        skip_tickers: Optional[set[str]] = None,
        on_batch: Optional[Callable[[list[dict], str], None]] = None,
    ) -> list[dict]:
        """Fetch filings for tickers filtered by whitelist/lookback rules.

        Parameters
        ----------
        tickers:
            Iterable of ticker strings to query.
        progress_fn:
            Optional callback for progress messages.
        stop_flag:
            Shared flag to allow cancellation mid-run.
        skip_tickers:
            If provided, any ticker already present in this set will be skipped.
        on_batch:
            Optional callback invoked with each ticker's filings as soon as they
            are fetched. Enables streaming writes instead of buffering the
            entire response.
        """

        start_date = date.today() - timedelta(days=self.max_lookback)
        start_expr = start_date.isoformat() + ":"

        results: list[dict] = []
        whitelist = [form for form in self.forms_whitelist if form]
        ticker_list = list(tickers)
        total = len(ticker_list)

        for idx, ticker in enumerate(ticker_list, start=1):
            if stop_flag and stop_flag.get("stop"):
                raise CancelledRun("cancel requested during EDGAR filings")

            ticker_norm = _normalize_ticker(ticker)
            if not ticker_norm:
                continue

            if skip_tickers and ticker_norm in skip_tickers:
                if progress_fn:
                    progress_fn(
                        f"[edgar filings] skipping {ticker_norm} (already cached)"
                    )
                continue

            if progress_fn:
                progress_fn(f"[edgar filings] starting {ticker_norm} ({idx}/{total})")

            try:
                self._rate_limit()
                company = Company(ticker_norm)
            except Exception as exc:
                logger.warning("EDGAR company lookup failed for %s: %s", ticker_norm, exc)
                continue

            try:
                self._rate_limit()
                filings = company.get_filings(
                    form=whitelist or None,
                    filing_date=start_expr,
                )
            except Exception as exc:
                logger.warning("EDGAR filings fetch failed for %s: %s", ticker_norm, exc)
                continue

            if filings is None:
                continue

            batch: list[dict] = []
            try:
                for filing in filings:
                    form_value = getattr(filing, "form", "")
                    filed_at = getattr(filing, "filing_date", "")
                    if not self._is_within_lookback(form_value, filed_at):
                        continue

                    self._rate_limit()

                    filing_url = (
                        getattr(filing, "filing_url", None)
                        or getattr(filing, "homepage_url", None)
                        or getattr(filing, "url", None)
                        or ""
                    )

                    batch.append(
                        {
                            "CIK": _normalize_cik(getattr(filing, "cik", "")),
                            "Ticker": ticker_norm,
                            "Company": getattr(filing, "company", ""),
                            "Form": form_value,
                            "FiledAt": filed_at,
                            "URL": filing_url,
                            "Desc": "",
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "EDGAR filings iteration failed for %s: %s", ticker_norm, exc
                )
                batch = []

            if batch:
                results.extend(batch)
                if on_batch:
                    on_batch(batch, ticker_norm)

            if progress_fn and (idx % 25 == 0 or idx == total):
                pct = int((idx / max(total, 1)) * 100)
                progress_fn(
                    f"[edgar filings] {idx}/{total} tickers ({pct}%)"
                )

        return results

    def download_filing_text(self, url: str) -> str:
        """Download filing HTML/text using edgartools HTTP client."""

        if not url:
            raise ValueError("filing URL is required")

        self._rate_limit()
        return download_text(url)

    def stats_string(self) -> str:
        if not self.rate_limiter:
            return "edgar n/a"
        return (
            f"edgar {self.rate_limiter.current_window_count()}/"
            f"{self.rate_limiter.per_minute} this_min"
        )


def get_adapter(cfg: Optional[dict] = None) -> EdgarAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = EdgarAdapter(cfg)
    return _ADAPTER


def set_adapter(adapter: EdgarAdapter) -> EdgarAdapter:
    """Seed the module-level adapter for shared throttling and identity."""

    global _ADAPTER
    _ADAPTER = adapter
    return _ADAPTER

