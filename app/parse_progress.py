import re
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class FormCount:
    parsed: int = 0
    valid: int = 0
    missing: int = 0

    def ensure_consistency(self) -> None:
        total = self.valid + self.missing
        if self.parsed < total:
            self.parsed = total

    def totals_match(self) -> bool:
        return self.parsed == self.valid + self.missing


class ParseProgressTracker:
    def __init__(self, on_change: Callable[[dict[str, FormCount], FormCount, list[str]], None] | None):
        self.on_change = on_change
        self.form_stats: dict[str, FormCount] = {}
        self.exhibit_stats = FormCount()
        self.ticker_last_form: dict[str, str] = {}
        self.ticker_last_key: dict[str, tuple[str, str, str]] = {}
        self.ticker_outcomes: set[tuple[str, str, str]] = set()
        self.compute_events: set[tuple[str, str, str]] = set()
        self.exhibit_tail: deque[str] = deque(maxlen=20)

    def reset(self) -> None:
        self.form_stats = {}
        self.exhibit_stats = FormCount()
        self.ticker_last_form = {}
        self.ticker_last_key = {}
        self.ticker_outcomes = set()
        self.compute_events = set()
        self.exhibit_tail.clear()
        self._notify()

    def process_message(self, full_message: str) -> None:
        detail = self._extract_parse_detail(full_message)
        if not detail:
            return

        if detail.startswith("parse_q10:"):
            if detail.startswith("parse_q10: start"):
                self.reset()
            return

        match = re.match(r"parse_q10\s*\[(?P<status>[^\]]+)\]\s*(?P<body>.*)", detail)
        if not match:
            return

        body = match.group("body")
        if not body:
            return

        changed = False
        if " fetching " in body:
            changed = self._handle_fetch(body)
        elif body.startswith("compute_runway:"):
            changed = self._handle_compute(body)
        elif " runway status " in body:
            changed = self._handle_status(body)
        elif " incomplete:" in body:
            changed = self._handle_incomplete(body)
        elif re.search(r"\brunway\b.*\bqtrs\b", body):
            changed = self._handle_ok(body)

        if changed:
            self._notify()

    def _extract_parse_detail(self, message: str) -> str | None:
        parts = message.split("|", 1)
        detail = parts[1].strip() if len(parts) == 2 else message.strip()
        if not detail.startswith("parse_q10"):
            return None
        return detail

    def _handle_fetch(self, body: str) -> bool:
        match = re.match(r"(?P<ticker>\S+)\s+fetching\s+(?P<form>[^\s]+)", body)
        if not match:
            return False
        form = self._canonical_form(match.group("form"))
        was_new_form = form not in self.form_stats
        self.form_stats.setdefault(form, FormCount())
        ticker = match.group("ticker").strip()
        if ticker:
            self.ticker_last_form[ticker] = form
            date_match = re.search(r"filed\s+([0-9]{4}-[0-9]{2}-[0-9]{2})", body)
            date_key = date_match.group(1) if date_match else ""
            self.ticker_last_key[ticker] = (ticker, form, date_key)
        return was_new_form

    def _handle_compute(self, body: str) -> bool:
        remainder = body[len("compute_runway:"):].strip()
        if not remainder:
            return False
        ticker = remainder.split()[0]
        form_match = re.search(r"form=([^\s]+)", body)
        form = self._canonical_form(form_match.group(1)) if form_match else "Unknown"
        date_match = re.search(r"date=([0-9]{4}-[0-9]{2}-[0-9]{2})", body)
        date_key = date_match.group(1) if date_match else ""
        key = (ticker, form, date_key)

        stats = self.form_stats.setdefault(form, FormCount())
        self.ticker_last_form[ticker] = form
        self.ticker_last_key[ticker] = key

        if key not in self.compute_events:
            self.compute_events.add(key)
            stats.parsed += 1
            stats.ensure_consistency()
            return True
        return False

    def _handle_status(self, body: str) -> bool:
        ticker, status_text = body.split(" runway status ", 1)
        return self._record_outcome(ticker.strip(), status_text.strip(), status_text=status_text.strip())

    def _handle_ok(self, body: str) -> bool:
        match = re.match(r"(?P<ticker>\S+)\s+runway\s+", body)
        if not match:
            return False
        ticker = match.group("ticker").strip()
        return self._record_outcome(ticker, "OK", force_valid=True)

    def _handle_incomplete(self, body: str) -> bool:
        match = re.match(r"(?P<ticker>\S+)\s+(?P<form>[^\s]+)\s+incomplete:\s*(?P<reason>.*)", body)
        if not match:
            return False

        ticker = match.group("ticker").strip()
        form = self._canonical_form(match.group("form"))
        reason = match.group("reason").strip()

        if not ticker:
            return False

        self.form_stats.setdefault(form, FormCount())
        self.ticker_last_form[ticker] = form
        self.ticker_last_key.setdefault(ticker, (ticker, form, ""))

        status_text = f"{form} incomplete: {reason}" if reason else f"{form} incomplete"
        return self._record_outcome(ticker, "incomplete", status_text=status_text)

    def _record_outcome(
        self,
        ticker: str,
        status: str,
        *,
        status_text: str | None = None,
        force_valid: bool = False,
    ) -> bool:
        if not ticker:
            return False

        form = self.ticker_last_form.get(ticker, "Unknown")
        canonical_form = self._canonical_form(form)
        stats = self.form_stats.setdefault(canonical_form, FormCount())

        key = self.ticker_last_key.get(ticker, (ticker, canonical_form, ""))
        if key in self.ticker_outcomes:
            return False

        is_valid = force_valid or self._status_is_valid(status)
        if is_valid:
            stats.valid += 1
        else:
            stats.missing += 1
        stats.ensure_consistency()
        self.ticker_outcomes.add(key)

        changed = True
        if status_text and "exhibit" in status_text.lower():
            if is_valid:
                self.exhibit_stats.valid += 1
            else:
                self.exhibit_stats.missing += 1
            self.exhibit_stats.ensure_consistency()
            entry = f"{ticker} – {status_text}"
            self.exhibit_tail.append(entry)
            changed = True

        return changed

    def _status_is_valid(self, status: str) -> bool:
        return status.strip().lower().startswith("ok")

    def _canonical_form(self, form: str | None) -> str:
        if not form:
            return "Unknown"
        raw_text = str(form).strip()
        if not raw_text:
            return "Unknown"
        text = raw_text.upper()
        if text.endswith("/A"):
            text = text[:-2]
        for prefix, mapped in (
            ("10-QT", "10-Q"),
            ("10-Q", "10-Q"),
            ("10-KT", "10-K"),
            ("10-K", "10-K"),
            ("20-F", "20-F"),
            ("40-F", "40-F"),
            ("6-K", "6-K"),
        ):
            if text.startswith(prefix):
                return mapped
        if text == "UNKNOWN":
            return "Unknown"
        return text

    def _notify(self) -> None:
        if self.on_change:
            self.on_change(self.form_stats, self.exhibit_stats, list(self.exhibit_tail))
