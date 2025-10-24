import requests, time
from requests import Response
from typing import Dict, Tuple
from app.rate_limit import RateLimiter


class HttpClient:
    """
    Rate-limited + retrying GET client.
    Auto-pause on 429/5xx.
    Tracks per-host:
      - rolling calls in last minute
      - total calls this run
      - per-minute limit
    So GUI can show e.g.:
      financialmodelingprep.com 45/240 this_min total=320
    """

    def __init__(self, user_agent: str, timeout: int, retries: int, backoff_secs: Tuple[int, int, int]):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.retries = retries
        self.backoff_secs = backoff_secs

        # host -> RateLimiter
        self.limiters: Dict[str, RateLimiter] = {}
        # host -> total calls made this run
        self.call_counts: Dict[str, int] = {}
        # host -> configured per-minute limit
        self.host_limits: Dict[str, int] = {}

    def _limiter_for(self, host: str, per_minute: int) -> RateLimiter:
        if host not in self.limiters:
            self.limiters[host] = RateLimiter(per_minute)
        # store limit so we can report it
        self.host_limits[host] = per_minute
        return self.limiters[host]

    def get(self, url: str, params: dict | None, per_minute: int) -> Response:
        """
        GET with rate limit + retry/backoff.
        Increments self.call_counts[host].
        """
        host = requests.utils.urlparse(url).netloc
        limiter = self._limiter_for(host, per_minute)

        last_exc = None
        for attempt in range(self.retries):
            # rate limit (blocks as needed)
            limiter.acquire()

            # book-keep call count
            self.call_counts[host] = self.call_counts.get(host, 0) + 1

            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                self._sleep_backoff(attempt)
                continue

            if r.status_code == 429:
                # rate limited => backoff, then retry
                last_exc = RuntimeError("429 Too Many Requests")
                self._sleep_backoff(attempt)
                continue

            if 500 <= r.status_code < 600:
                # server error => retry
                last_exc = RuntimeError(f"{r.status_code} Server Error")
                self._sleep_backoff(attempt)
                continue

            if r.status_code >= 400:
                # fatal client error
                r.raise_for_status()

            return r

        if last_exc:
            raise last_exc
        raise RuntimeError("request failed with no response")

    def _sleep_backoff(self, attempt: int) -> None:
        idx = min(attempt, len(self.backoff_secs) - 1)
        time.sleep(self.backoff_secs[idx])

    def stats_string(self) -> str:
        """
        Build a short status string for GUI like:
        financialmodelingprep.com 52/240 this_min total=612 | www.sec.gov 2/10 this_min total=2
        """
        parts = []
        for host, total_calls in self.call_counts.items():
            limiter = self.limiters.get(host)
            used_now = limiter.current_window_count() if limiter else 0
            limit_now = self.host_limits.get(host, 0)
            parts.append(f"{host} {used_now}/{limit_now} this_min total={total_calls}")
        return " | ".join(parts)
