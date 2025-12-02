from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Callable, Iterable, Optional

from edgar import Company, Financials, Filing, get_by_accession_number_enriched, set_identity
from edgar.httprequests import download_text
from edgar.reference.tickers import get_company_tickers

from app.cancel import CancelledRun
from app.config import (
    filings_form_lookbacks,
    filings_max_lookback,
    load_config,
    weekly_allowed_forms,
)
from app.rate_limit import RateLimiter
from app.runway_financials import (
    RUNWAY_REASON_NO_BALANCE,
    RUNWAY_REASON_NO_CASHFLOW,
    RUNWAY_REASON_NO_PERIODS,
    RUNWAY_REASON_NO_XBRL,
    RUNWAY_REASON_OK,
    RUNWAY_REASON_PARSER_ERROR,
    RUNWAY_REASON_UNSUPPORTED_FORM,
    compute_runway_from_financials,
)
from app.universe_filters import load_drop_filters, should_drop_record
from parse.postproc import finalize_runway_result
from parse.units import round_half_up


logger = logging.getLogger(__name__)


def partition_completed_and_pending(
    all_tickers: Iterable[str], completed_tickers: Iterable[str]
) -> tuple[set[str], set[str]]:
    """Return completed and pending ticker sets for a filings fetch.

    The helper is intentionally simple and deterministic so it can be tested
    without requiring any network calls or executor machinery.
    """

    completed_set = {ticker for ticker in completed_tickers if ticker}
    all_set = {ticker for ticker in all_tickers if ticker}
    pending_set = all_set - completed_set
    return completed_set, pending_set

_ADAPTER: "EdgarAdapter" | None = None
_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)
_ACCESSION_FALLBACK_RE = re.compile(r"(\d{10})[-_]?(\d{2})[-_]?(\d{6})")


def _format_accession(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if not digits:
        return ""
    digits = digits.zfill(18)
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def _parse_accession_from_url(url: str) -> tuple[str, str]:
    try:
        match = _ACCESSION_RE.search(url)
    except Exception:
        return "", ""
    if match:
        cik_digits = re.sub(r"\D", "", match.group(1) or "")
        accession_digits = _format_accession(match.group(2) or "")
        return cik_digits.zfill(10), accession_digits

    try:
        fallback = _ACCESSION_FALLBACK_RE.search(url)
    except Exception:
        return "", ""

    if not fallback:
        return "", ""

    cik_digits = fallback.group(1) or ""
    accession_digits = fallback.group(1) + fallback.group(2) + fallback.group(3)
    return _normalize_cik(cik_digits), _format_accession(accession_digits)


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

    _RUNWAY_LOG_PREFIX = "edgar_runway"

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        edgar_cfg = cfg.get("Edgar", {})
        self.forms_whitelist = sorted(weekly_allowed_forms(cfg))
        self.form_lookbacks = filings_form_lookbacks(cfg)
        self.max_lookback = filings_max_lookback(cfg)

        self.rate_limit_wait_secs = 0.0
        self._rate_limit_lock = Lock()

        throttle_per_min = edgar_cfg.get("ThrottlePerMin") or cfg.get(
            "RateLimitsPerMin", {}
        ).get("SEC")
        self.rate_limiter = RateLimiter(throttle_per_min) if throttle_per_min else None

        self._configure_identity(edgar_cfg.get("UserAgent") or cfg.get("UserAgent"))

        self.last_filings_stats: dict | None = None

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
            start_wait = time.monotonic()
            self.rate_limiter.acquire()
            waited = time.monotonic() - start_wait
            if waited > 0:
                with self._rate_limit_lock:
                    self.rate_limit_wait_secs += waited

    def _filing_fetch_workers(self) -> int:
        workers_cfg = self.cfg.get("Workers", {}) if isinstance(self.cfg, dict) else {}
        edgar_cfg = workers_cfg.get("EDGAR") or workers_cfg.get("Edgar") or {}
        try:
            workers = int(edgar_cfg.get("Workers"))
        except (TypeError, ValueError):
            workers = 1
        return workers if workers and workers > 0 else 1

    def _filing_fetch_timeout(self) -> int | None:
        workers_cfg = self.cfg.get("Workers", {}) if isinstance(self.cfg, dict) else {}
        edgar_cfg = workers_cfg.get("EDGAR") or workers_cfg.get("Edgar") or {}
        try:
            timeout = edgar_cfg.get("TimeoutSeconds")
            timeout = int(timeout) if timeout is not None else None
        except (TypeError, ValueError):
            timeout = None
        return timeout if timeout and timeout > 0 else None

    def _resolve_filing(self, filing_or_url) -> Optional[Filing]:
        if isinstance(filing_or_url, Filing):
            return filing_or_url

        if not filing_or_url:
            return None

        cik, accession = _parse_accession_from_url(str(filing_or_url))
        if not accession:
            return None

        try:
            self._rate_limit()
            return get_by_accession_number_enriched(accession)
        except Exception as exc:
            logger.warning("Failed to fetch filing %s: %s", accession, exc)
            return None

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

    def _fetch_filings_for_ticker(
        self,
        ticker: str,
        *,
        whitelist: list[str],
        start_expr: str,
        progress_fn: Optional[Callable[[str], None]] = None,
        stop_flag: Optional[dict] = None,
        idx: int = 0,
        total: int = 0,
    ) -> tuple[list[dict], dict]:
        """Resolve a single ticker's filings list via edgartools."""

        # Primary filings fetcher: all EDGAR list retrievals originate here.
        if stop_flag and stop_flag.get("stop"):
            raise CancelledRun("cancel requested during EDGAR filings")

        ticker_norm = _normalize_ticker(ticker)
        if not ticker_norm:
            return [], {"ticker": ticker, "raw_count": 0, "kept_count": 0, "duration_ms": 0}

        if progress_fn:
            progress_fn(f"[edgar filings] starting {ticker_norm} ({idx}/{total})")

        start_time = time.monotonic()
        fetch_duration_ms = 0
        raw_count = 0
        batch: list[dict] = []
        cik_value = ""

        try:
            company = Company(ticker_norm)
            cik_value = _normalize_cik(getattr(company, "cik", ""))
        except Exception as exc:
            logger.warning("EDGAR company lookup failed for %s: %s", ticker_norm, exc)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return [], {
                "ticker": ticker_norm,
                "raw_count": 0,
                "kept_count": 0,
                "duration_ms": duration_ms,
                "cik": cik_value,
                "fetch_ms": fetch_duration_ms,
            }

        try:
            # Only rate-limit the outbound filings fetch, not local object creation.
            call_start = time.monotonic()
            self._rate_limit()
            filings = company.get_filings(
                form=whitelist or None,
                filing_date=start_expr,
            )
            filings_list = list(filings) if filings is not None else []
            raw_count = len(filings_list)
            fetch_duration_ms = int((time.monotonic() - call_start) * 1000)
        except Exception as exc:
            logger.warning("EDGAR filings fetch failed for %s: %s", ticker_norm, exc)
            filings_list = []

        def _first_attr(obj: object, attrs: list[str]) -> str:
            for attr in attrs:
                value = getattr(obj, attr, None)
                if value:
                    return str(value)
            return ""

        try:
            for filing in filings_list:
                form_value = getattr(filing, "form", "")
                filed_at = getattr(filing, "filing_date", "")
                if not self._is_within_lookback(form_value, filed_at):
                    continue

                filing_url = (
                    getattr(filing, "filing_url", None)
                    or getattr(filing, "homepage_url", None)
                    or getattr(filing, "url", None)
                    or ""
                )

                accession_value = _first_attr(
                    filing,
                    [
                        "accession_no",
                        "accession",
                        "accession_number",
                        "accession_number_no_dashes",
                        "accessionNumber",
                    ],
                )

                master_txt_url = _first_attr(
                    filing,
                    [
                        "txt_url",
                        "text_url",
                        "master_text_url",
                        "full_text_url",
                        "primary_document_url",
                        "primary_doc_url",
                    ],
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
                        "Accession": accession_value,
                        "MasterTxtURL": master_txt_url,
                    }
                )
        except Exception as exc:
            logger.warning("EDGAR filings iteration failed for %s: %s", ticker_norm, exc)
            batch = []

        duration_ms = int((time.monotonic() - start_time) * 1000)
        log_message = (
            "WEEKLY_FILINGS_TICKER: "
            f"ticker={ticker_norm} "
            f"cik={cik_value} "
            f"raw_filings={raw_count} "
            f"kept_after_form_filter={len(batch)} "
            f"fetch_ms={fetch_duration_ms} "
            f"duration_ms={duration_ms}"
        )
        logger.info(log_message)
        if progress_fn:
            progress_fn(log_message)

        return batch, {
            "ticker": ticker_norm,
            "raw_count": raw_count,
            "kept_count": len(batch),
            "duration_ms": duration_ms,
            "fetch_ms": fetch_duration_ms,
            "cik": cik_value,
        }

    def fetch_recent_filings(
        self,
        tickers: Iterable[str],
        progress_fn: Optional[Callable[[str], None]] = None,
        stop_flag: Optional[dict] = None,
        *,
        skip_tickers: Optional[set[str]] = None,
        on_batch: Optional[Callable[[list[dict], str], None]] = None,
    ) -> list[dict]:
        """Fetch filings for tickers filtered by whitelist/lookback rules."""

        start_date = date.today() - timedelta(days=self.max_lookback)
        start_expr = start_date.isoformat() + ":"

        results: list[dict] = []
        whitelist = [form for form in self.forms_whitelist if form]
        ticker_list = list(tickers)
        total = len(ticker_list)

        stats = {"tickers": 0, "raw_filings": 0, "kept_filings": 0}
        per_ticker: list[dict] = []
        completed_tickers: set[str] = set()
        timed_out = False
        fetch_start = time.monotonic()

        with self._rate_limit_lock:
            self.rate_limit_wait_secs = 0.0

        def _handle_batch(batch: list[dict], ticker_norm: str, per_ticker_stats: dict) -> None:
            nonlocal results
            stats["tickers"] += 1
            stats["raw_filings"] += per_ticker_stats.get("raw_count", 0)
            stats["kept_filings"] += per_ticker_stats.get("kept_count", 0)
            per_ticker.append(per_ticker_stats)
            if batch:
                results.extend(batch)
                if on_batch:
                    on_batch(batch, ticker_norm)

        workers = self._filing_fetch_workers()
        timeout_seconds = self._filing_fetch_timeout()
        to_process: list[tuple[int, str]] = []
        for idx, ticker in enumerate(ticker_list, start=1):
            ticker_norm = _normalize_ticker(ticker)
            if not ticker_norm:
                continue
            if skip_tickers and ticker_norm in skip_tickers:
                if progress_fn:
                    progress_fn(
                        f"[edgar filings] skipping {ticker_norm} (already cached)"
                    )
                continue
            to_process.append((idx, ticker_norm))

        timeout_display = timeout_seconds if timeout_seconds is not None else "disabled"
        logger.info(
            "[edgar filings] start: total_tickers=%s, timeout_seconds=%s, max_workers=%s",
            len(to_process),
            timeout_display,
            workers,
        )

        if workers <= 1:
            for idx, ticker_norm in to_process:
                batch, per_ticker_stats = self._fetch_filings_for_ticker(
                    ticker_norm,
                    whitelist=whitelist,
                    start_expr=start_expr,
                    progress_fn=progress_fn,
                    stop_flag=stop_flag,
                    idx=idx,
                    total=total,
                )
                _handle_batch(batch, ticker_norm, per_ticker_stats)
                completed_tickers.add(ticker_norm)

                if progress_fn and (idx % 25 == 0 or idx == total):
                    pct = int((idx / max(total, 1)) * 100)
                    progress_fn(
                        f"[edgar filings] {idx}/{total} tickers ({pct}%)"
                    )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for idx, ticker_norm in to_process:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[edgar filings] worker_start: batch_size=1, tickers=[%s]",
                            ticker_norm,
                        )
                    future = executor.submit(
                        self._fetch_filings_for_ticker,
                        ticker_norm,
                        whitelist=whitelist,
                        start_expr=start_expr,
                        progress_fn=progress_fn,
                        stop_flag=stop_flag,
                        idx=idx,
                        total=total,
                    )
                    futures[future] = ticker_norm

                completed = 0
                pending = set(futures.keys())

                def _iter_completed(current_pending):
                    if timeout_seconds is None:
                        try:
                            return as_completed(current_pending)
                        except TypeError:  # compatibility with patched/mocked as_completed
                            return as_completed(current_pending)

                    try:
                        return as_completed(current_pending, timeout=timeout_seconds)
                    except TypeError:  # compatibility with patched/mocked as_completed
                        return as_completed(current_pending)

                while pending:
                    try:
                        for future in _iter_completed(pending):
                            pending.discard(future)
                            ticker_norm = futures[future]
                            try:
                                batch, per_ticker_stats = future.result()
                            except Exception as exc:  # pragma: no cover - defensive guard
                                logger.warning(
                                    "EDGAR filings fetch crashed for %s: %s", ticker_norm, exc
                                )
                                per_ticker_stats = {
                                    "ticker": ticker_norm,
                                    "raw_count": 0,
                                    "kept_count": 0,
                                    "duration_ms": 0,
                                    "fetch_ms": 0,
                                    "cik": "",
                                }
                                if progress_fn:
                                    progress_fn(
                                        f"[edgar filings] error {ticker_norm}: {exc}"
                                    )
                                batch = []

                            _handle_batch(batch, ticker_norm, per_ticker_stats)
                            completed_tickers.add(ticker_norm)
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "[edgar filings] worker_done: batch_size=%s, tickers=[%s]",
                                    len(batch),
                                    ticker_norm,
                                )

                            completed += 1
                            if progress_fn and (completed % 25 == 0 or completed == total):
                                pct = int((completed / max(total, 1)) * 100)
                                progress_fn(
                                    f"[edgar filings] {completed}/{total} tickers ({pct}%)"
                                )
                    except TimeoutError:
                        elapsed = round(time.monotonic() - fetch_start, 3)
                        stalled = [futures[future] for future in pending]
                        completed_now, pending_now = partition_completed_and_pending(
                            [ticker for _, ticker in to_process], completed_tickers
                        )
                        pending_now |= set(stalled)
                        completed_now -= set(stalled)
                        timed_out = True

                        logger.warning(
                            "[edgar filings] timeout: elapsed=%ss, timeout=%ss, "
                            "completed=%s, pending=%s, pending_tickers=%s",
                            elapsed,
                            timeout_seconds,
                            len(completed_now),
                            len(pending_now),
                            stalled,
                        )
                        if progress_fn:
                            progress_fn(
                                "[edgar filings] timeout waiting for tickers: {}".format(
                                    ", ".join(stalled)
                                )
                            )
                        for future in list(pending):
                            future.cancel()
                        pending.clear()
                        break

        rl_wait = 0.0
        with self._rate_limit_lock:
            rl_wait = round(self.rate_limit_wait_secs, 3)
            stats["rl_wait_sec"] = rl_wait

        if per_ticker:
            kept_counts = [entry.get("kept_count", 0) for entry in per_ticker]
            raw_counts = [entry.get("raw_count", 0) for entry in per_ticker]
            try:
                stats["kept_min"] = min(kept_counts)
                stats["kept_max"] = max(kept_counts)
                stats["kept_median"] = sorted(kept_counts)[len(kept_counts) // 2]
                stats["raw_max"] = max(raw_counts)
            except ValueError:
                pass

        completed_now, pending_now = partition_completed_and_pending(
            [ticker for _, ticker in to_process], completed_tickers
        )

        stats["timeout_pending"] = len(pending_now)
        stats["scheduled_tickers"] = len(to_process)
        stats["timed_out"] = timed_out

        summary_line = (
            "[edgar filings] summary: "
            f"total={len(to_process)} "
            f"completed={len(completed_now)} "
            f"timeout_pending={len(pending_now)}"
        )
        logger.info(summary_line)

        summary_msg = (
            "EDGAR_FILINGS_SUMMARY: "
            f"tickers={stats.get('tickers', 0)} "
            f"raw={stats.get('raw_filings', 0)} "
            f"kept={stats.get('kept_filings', 0)} "
            f"rl_wait_sec={rl_wait}"
        )
        logger.info(summary_msg)

        self.last_filings_stats = stats
        return results

    def download_filing_text(self, url: str) -> str:
        """Download filing HTML/text using edgartools HTTP client."""

        if not url:
            raise ValueError("filing URL is required")

        self._rate_limit()
        return download_text(url)

    def _log_runway_warning(self, code: str, filing: object, **fields) -> None:
        filing_fields = {
            "ticker": _normalize_ticker(getattr(filing, "ticker", "") or getattr(filing, "symbol", "")),
            "cik": _normalize_cik(getattr(filing, "cik", "") or getattr(filing, "cik_str", "")),
            "form": getattr(filing, "form", ""),
            "filed_at": getattr(filing, "filing_date", ""),
            "accession": getattr(filing, "accession_no", ""),
        }
        payload = {**filing_fields, **{k: v for k, v in fields.items() if v}}
        context = ", ".join(
            f"{k}={v}" for k, v in payload.items() if v is not None and str(v) != ""
        )
        logger.warning(f"{self._RUNWAY_LOG_PREFIX}: {code} [{context}]")

    def runway_from_financials(self, filing_or_url, form_hint: Optional[str]):
        """Compute runway using edgartools Financials as the canonical source."""

        filing = self._resolve_filing(filing_or_url)
        if filing is None:
            return {"reason_code": RUNWAY_REASON_PARSER_ERROR, "reason_detail": "unable to resolve filing"}

        financials = None
        try:
            financials = getattr(filing, "financials", None)
        except Exception:
            financials = None
        if financials is None:
            financials = Financials.extract(filing)

        if financials is None:
            self._log_runway_warning("missing_xbrl", filing)
            return {"reason_code": RUNWAY_REASON_NO_XBRL, "reason_detail": "missing XBRL/financials"}

        computation = compute_runway_from_financials(financials)

        if computation.runway_quarters is None and computation.reason_code == RUNWAY_REASON_OK:
            computation.reason_code = RUNWAY_REASON_NO_CASHFLOW
            computation.reason_detail = "operating cash flow missing"

        source_url = (
            getattr(filing, "filing_url", None)
            or getattr(filing, "homepage_url", None)
            or getattr(filing, "url", None)
            or ""
        )

        period_months = computation.period_months
        ocf_quarterly = None
        ocf_for_finalize = computation.ocf if period_months else None
        if ocf_for_finalize is not None:
            try:
                quarters = period_months / 3.0
                ocf_quarterly = ocf_for_finalize / quarters if quarters else None
            except Exception:
                ocf_quarterly = None

        result = finalize_runway_result(
            cash=computation.cash,
            ocf_raw=ocf_for_finalize,
            ocf_quarterly=ocf_quarterly,
            period_months=period_months,
            assumption="edgartools financials",
            note=computation.reason_detail or "",
            form_type=getattr(filing, "form", form_hint),
            units_scale=1,
            status="OK" if computation.reason_code == RUNWAY_REASON_OK else computation.reason_code,
            source_tags=["XBRL"],
        )

        if computation.runway_quarters is not None:
            result["runway_quarters_raw"] = computation.runway_quarters
            result["runway_quarters"] = round_half_up(computation.runway_quarters)
            result["runway_quarters_display"] = round_half_up(computation.runway_quarters, 2)
            result["runway_months_display"] = round_half_up(computation.runway_quarters * 3, 2)

        result.update(
            {
                "period_months": period_months,
                "cash": computation.cash,
                "ocf": computation.ocf,
                "reason_code": computation.reason_code,
                "reason_detail": computation.reason_detail,
                "filing_date": getattr(filing, "filing_date", ""),
                "filing_url": source_url,
                "form_type": getattr(filing, "form", form_hint),
                "source_tags": ["XBRL"],
            }
        )

        result.setdefault("runway_quarters", computation.runway_quarters)

        return result

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

