"""Parser for extracting cash runway metrics from 10-Q/10-K filings."""
from __future__ import annotations

import html
import json
import logging
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlunparse


_USER_AGENT: str | None = None

_ACCESSION_RE = re.compile(r"/data/(\d{1,10})/([\w-]+)/", re.IGNORECASE)
_FORM_TYPE_PATTERN = re.compile(
    r"(10-QT?|10-KT?|20-F|40-F|6-K)(?:/A)?",
    re.IGNORECASE,
)

_LOGGER = logging.getLogger(__name__)


def _log_parse_event(level: int, message: str, **fields: object) -> None:
    details: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        details.append(f"{key}={value}")
    if details:
        message = f"{message} ({', '.join(details)})"
    _LOGGER.log(level, message)


def _preview_text(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    sanitized = text.replace("\r", " ").replace("\n", " ")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > limit:
        return sanitized[: limit - 3] + "..."
    return sanitized


def _unescape_html_entities(html_text: str, *, context: Optional[str] = None) -> str:
    if not html_text:
        return html_text
    before = _preview_text(html_text)
    unescaped = html.unescape(html_text)
    after = _preview_text(unescaped)
    _log_parse_event(
        logging.DEBUG,
        "html entity unescape",
        url=context,
        unescape_before=before,
        unescape_after=after,
    )
    return unescaped


def _format_score_tuple(score: Optional[tuple[object, ...]]) -> str:
    if not score:
        return "()"
    return "(" + ", ".join(str(part) for part in score) + ")"


def _canonicalize_sec_filing_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if not host.endswith("sec.gov"):
        return url

    if not any(path.startswith(prefix) for prefix in ("/ix", "/ixviewer", "/cgi-bin/viewer")):
        return url

    query = parse_qs(parsed.query)
    doc_values = query.get("doc") or []
    if not doc_values:
        return url

    doc_value = unquote(str(doc_values[0] or "")).strip()
    if not doc_value:
        return url

    doc_parsed = urlparse(doc_value)
    fragmentless = doc_parsed._replace(fragment="")
    if fragmentless.scheme and fragmentless.netloc:
        canonical = fragmentless.geturl()
    else:
        doc_path = fragmentless.path or doc_value
        if not doc_path.startswith("/"):
            doc_path = f"/{doc_path.lstrip('/')}"
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or "www.sec.gov"
        canonical = urlunparse((scheme, netloc, doc_path, "", fragmentless.query, ""))

    return canonical


def _user_agent() -> str:
    global _USER_AGENT
    if _USER_AGENT is None:
        try:
            from app.config import load_config
            cfg = load_config()
            user_agent = str(cfg.get("UserAgent", "")).strip()
        except Exception:
            user_agent = ""
        if not user_agent:
            user_agent = "microcap-pipeline/1.0"
        _USER_AGENT = user_agent
    return _USER_AGENT


def _fetch_url(url: str) -> bytes:
    req = request.Request(url, headers={"User-Agent": _user_agent()})
    with request.urlopen(req) as response:
        return response.read()


def _fetch_json(url: str) -> dict:
    try:
        raw = _fetch_url(url)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error fetching JSON ({exc.code}): {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error fetching JSON ({exc.reason}): {url}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error fetching JSON: {url} ({exc.__class__.__name__}: {exc})"
        ) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


_NUMERIC_TOKEN_PATTERN = re.compile(
    r"[\d\$\(\)\-−€£¥₩₽₹\.,'’\u00A0\u202F]"
)

_CURRENCY_MARKERS = [
    "US$",
    "U.S.$",
    "C$",
    "CA$",
    "CAD",
    "USD",
    "EUR",
    "GBP",
    "AUD",
    "JPY",
    "RMB",
    "CHF",
    "HK$",
    "SGD",
    "NZD",
    "€",
    "£",
    "¥",
]

_CURRENCY_MARKER_PATTERN = "|".join(
    sorted({re.escape(m.upper()) for m in _CURRENCY_MARKERS}, key=len, reverse=True)
)

_NUMBER_SEARCH_PATTERN = re.compile(
    rf"[\(\[]*(?:({_CURRENCY_MARKER_PATTERN})\s*)?[-+]?\d[\d,\.\u00A0\u202F'’]*(?:[\.,]\d+)?(?:\s*({_CURRENCY_MARKER_PATTERN}))?[\)\]]?",
    re.IGNORECASE,
)

_CURRENCY_CONTEXT_WORDS = {
    re.sub(r"[^A-Z$€£¥₹₩₽]", "", marker.upper())
    for marker in _CURRENCY_MARKERS
}

_GROUP_SEPARATOR_CHARS = " ,.\u00A0\u202F'’"
_GROUP_SEPARATOR_CLASS = re.escape(_GROUP_SEPARATOR_CHARS)
_GROUPED_THOUSANDS_PATTERN = re.compile(
    rf"^\d{{1,3}}([{_GROUP_SEPARATOR_CLASS}]\d{{3}})+([\.,]\d+)?$"
)

_TOKEN_LOOKAHEAD = 200
_SCALE_PATTERN = re.compile(
    r"\([^)]*?\bin\s+(thousand|thousands|million|millions)\b[^)]*\)",
    re.IGNORECASE,
)

_OCF_IFRS_PATTERN = re.compile(
    r"net\s+cash\s+(?:flows?\s+)?(?:provided|providing|from|used|generated)\s+(?:by|from|in)\s+operating\s+activities",
    re.IGNORECASE,
)

_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<t[hd][^>]*>.*?</t[hd]>", re.IGNORECASE | re.DOTALL)

_EXHIBIT_FILENAME_PRIORITY = [
    "ex99-2",
    "ex99d2",
    "ex_99_2",
    "exhibit99-2",
    "ex99-1",
    "ex99d1",
    "ex_99_1",
]

FORM_ADAPTERS = {
    "10-Q": "US_Quarterly",
    "10-K": "US_Annual",
    "20-F": "Foreign_Annual",
    "40-F": "Foreign_Annual",
    "6-K": "SixK",
}

_BALANCE_HEADERS = [
    "CONSOLIDATED BALANCE SHEETS",
    "CONDENSED CONSOLIDATED BALANCE SHEETS",
    "BALANCE SHEETS",
]

_CASHFLOW_HEADERS_BASE = [
    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONSOLIDATED STATEMENT OF CASH FLOWS",
    "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS",
    "STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "INTERIM CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS",
]

_CASHFLOW_HEADER_PATTERNS = [
    re.compile(
        r"CONSOLIDATED\s*(?:CONDENSED\s*)?(?:INTERIM\s*)?STATEMENTS?\s*OF\s*CASH\s*FLOWS",
        re.IGNORECASE,
    ),
]

_CASH_KEYWORDS_BALANCE = [
    "Cash and cash equivalents",
    "Cash and cash equivalents, current",
    "Cash",
    "Cash, cash equivalents and restricted cash",
    "Cash and cash equivalents including restricted cash",
    "Cash and restricted cash",
]

_CASH_KEYWORDS_FLOW = [
    "Cash and cash equivalents, end of period",
    "Cash and cash equivalents at end of period",
    "Cash, cash equivalents and restricted cash, end of period",
    "Cash and cash equivalents, including restricted cash, end of period",
    "Cash, end of period",
    "Cash - End of period",
    "Cash – End of period",
    "Cash end of period",
    "Cash at end of period",
    "Cash at period end",
    "Cash, end of year",
    "Cash end of year",
    "Cash, end of the year",
    "End of period",
    "Cash and cash equivalents and restricted cash",
    "Cash and restricted cash, end of period",
    "Cash and restricted cash at end of period",
    "Cash and cash equivalents and restricted cash, end of period",
]

_OCF_KEYWORDS_BURN_BASE = [
    "Net cash used in operating activities - continuing operations",
    "Net cash used in operating activities — continuing operations",
    "Net cash used in operating activities from continuing operations",
    "Net cash used for operating activities - continuing operations",
    "Net cash (used in) operating activities - continuing operations",
    "Net cash (used in) operating activities — continuing operations",
    "Net cash used in operating activities, continuing operations",
    "Net cash (used in) operating activities, continuing operations",
    "Net cash used for operating activities, continuing operations",
    "Net cash used in operating activities",
    "Net cash used for operating activities",
    "Net cash (used in) operating activities",
    "Net cash flows used in operating activities",
]

_OCF_KEYWORDS_PROVIDED_BASE = [
    "Net cash provided by operating activities - continuing operations",
    "Net cash provided by operating activities — continuing operations",
    "Net cash provided by operating activities from continuing operations",
    "Net cash provided by (used in) operating activities - continuing operations",
    "Net cash provided by (used in) operating activities — continuing operations",
    "Net cash from operating activities - continuing operations",
    "Net cash from operating activities — continuing operations",
    "Net cash provided by operating activities, continuing operations",
    "Net cash provided by (used in) operating activities, continuing operations",
    "Net cash flows from operating activities, continuing operations",
    "Net cash from operating activities, continuing operations",
    "Net cash provided by operating activities",
    "Net cash provided by (used in) operating activities",
    "Net cash flows from operating activities",
    "Net cash from operating activities",
]

_OCF_KEYWORDS_BURN_EXTRA = [
    "Net cash used in operating activities — continuing operations",
    "Net cash (used in) operating activities — continuing operations",
    "Net cash used for operating activities — continuing operations",
    "Net cash used in operating activities",
    "Net cash used for operating activities",
    "Net cash (used in) operating activities",
    "Net cash flows used in operating activities",
    "Net cash flows (used in) operating activities",
    "Net cash used in operations",
    "Cash used in operations",
    "Cash used in operating activities",
    "Cash flows used in operating activities",
    "Cash flows (used in) operating activities",
    "Cash flows provided by (used in) operating activities",
    "Net cash outflow from operating activities",
    "Net cash (outflow) from operating activities",
    "Net cash used by operating activities",
]

_OCF_KEYWORDS_PROVIDED_EXTRA = [
    "Net cash provided by operating activities — continuing operations",
    "Net cash provided by (used in) operating activities — continuing operations",
    "Net cash provided from operating activities — continuing operations",
    "Net cash from operating activities — continuing operations",
    "Net cash flows from operating activities — continuing operations",
    "Net cash provided by operating activities",
    "Net cash provided from operating activities",
    "Net cash provided by (used in) operating activities",
    "Net cash from operating activities",
    "Net cash flows from operating activities",
    "Net cash flow from operating activities",
    "Net cash generated from operating activities",
    "Net cash generated by operating activities",
    "Net cash flows generated by operating activities",
    "Net cash flows generated from operating activities",
    "Cash generated from operations",
    "Cash generated by operations",
    "Net cash inflow from operating activities",
    "Cash flows provided by operating activities",
    "Cash flows provided by (used in) operating activities",
    "Cash flows from operating activities",
]


_FACT_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
]
_PERIOD_TEXT_PATTERNS = [
    (re.compile(r"three\s+months\s+ended", re.IGNORECASE), 3),
    (re.compile(r"six\s+months\s+ended", re.IGNORECASE), 6),
    (re.compile(r"nine\s+months\s+ended", re.IGNORECASE), 9),
    (re.compile(r"twelve\s+months\s+ended", re.IGNORECASE), 12),
    (re.compile(r"twelve\s+month\s+period", re.IGNORECASE), 12),
    (re.compile(r"year\s+ended", re.IGNORECASE), 12),
]

_PERIOD_MONTH_MAP = {
    "Q1": 3,
    "Q2": 6,
    "Q3": 9,
    "Q4": 12,
    "FY": 12,
    "HY": 6,
    "H1": 6,
    "H2": 6,
    "YTD": 12,
}

_SCALE_MULTIPLIER = {
    "thousands": 1_000,
    "thousand": 1_000,
    "millions": 1_000_000,
    "million": 1_000_000,
}


def _round_half_up(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    try:
        quant = Decimal("1").scaleb(-digits)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return float(value)
    return float(rounded)


def _months_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
    if not start or not end:
        return None
    delta_days = (end - start).days
    if delta_days <= 0:
        return None
    approx_months = delta_days / 30.4375
    if approx_months <= 0:
        return None
    rounded = int(round(approx_months))
    for target in (3, 6, 9, 12):
        if abs(rounded - target) <= 1:
            return target
    return rounded


def _infer_period_months(fact: Optional[dict], form_type: Optional[str]) -> Optional[int]:
    if fact:
        fp_value = str(fact.get("fp") or "").upper()
        if fp_value in _PERIOD_MONTH_MAP:
            months = _PERIOD_MONTH_MAP[fp_value]
            if months in {3, 6, 9, 12}:
                return months
        start = _parse_fact_timestamp(fact.get("start"))
        end = _parse_fact_timestamp(fact.get("end") or fact.get("instant"))
        months = _months_between(start, end)
        if months:
            return months

    if not form_type:
        return None

    defaults = _form_defaults(form_type)
    return defaults.get("period_months_default")


def _normalize_ocf_value(
    value: Optional[float],
    period_months: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[str]]:
    if value is None:
        return None, period_months, None

    if period_months in {3, 6, 9, 12}:
        divisor_map = {3: 1, 6: 2, 9: 3, 12: 4}
        assumption_map = {3: "", 6: "6m/2", 9: "9m/3", 12: "annual/4"}
        divisor = divisor_map[period_months]
        normalized_value = float(value) / float(divisor)
        return normalized_value, period_months, assumption_map[period_months]

    return float(value) if value is not None else None, period_months, None


def _detect_scale_multiplier(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = _SCALE_PATTERN.search(text)
    if not match:
        return None
    keyword = match.group(1).lower()
    return _SCALE_MULTIPLIER.get(keyword)


def normalize_number(token: Optional[str]) -> Optional[float]:
    if token is None:
        return None
    candidate = str(token).strip()
    if not candidate:
        return None
    try:
        return float(_to_float(candidate))
    except ValueError:
        cleaned = re.sub(r"[^0-9\-\+\(\)\.\,\u00A0\u202F'’]", "", candidate)
        if not cleaned:
            return None
        try:
            return float(_to_float(cleaned))
        except ValueError:
            return None


def find_scale_near(fragment: Optional[str], full_html: Optional[str] = None) -> int:
    def _search_lines(lines: Iterable[str]) -> Optional[int]:
        for line in lines:
            match = _SCALE_PATTERN.search(line)
            if match:
                keyword = match.group(1).lower()
                value = _SCALE_MULTIPLIER.get(keyword)
                if value:
                    return value
        return None

    if fragment:
        upper_fragment = fragment
        head_lines = upper_fragment.splitlines()[:5]
        scale = _search_lines(head_lines)
        if scale:
            return scale

        table_idx = upper_fragment.lower().find("<table")
        if table_idx > 0:
            before_table = upper_fragment[:table_idx]
            scale = _search_lines(before_table.splitlines()[-5:])
            if scale:
                return scale

    if fragment and full_html:
        snippet = fragment[:400]
        search_target = snippet
        idx = full_html.find(search_target)
        if idx == -1:
            snippet = _strip_html(fragment)[:200]
            idx = full_html.find(snippet)
        if idx != -1:
            prefix = full_html[:idx]
            lines = prefix.splitlines()
            scale = _search_lines(lines[-5:])
            if scale:
                return scale
            window_start = max(0, idx - 5000)
            window_text = full_html[window_start:idx]
            match = _SCALE_PATTERN.search(window_text)
            if match:
                keyword = match.group(1).lower()
                value = _SCALE_MULTIPLIER.get(keyword)
                if value:
                    return value
            after_text = full_html[idx : idx + 2000]
            match_after = _SCALE_PATTERN.search(after_text)
            if match_after:
                keyword = match_after.group(1).lower()
                value = _SCALE_MULTIPLIER.get(keyword)
                if value:
                    return value

    if full_html:
        fallback_lines = full_html.splitlines()
        scale = _search_lines(fallback_lines[:5])
        if scale:
            return scale

    return 1


def _to_float(num_str: str) -> float:
    text = num_str.strip()
    match = _NUMBER_SEARCH_PATTERN.search(text)
    if not match:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    value = match.group(0)
    negative = "(" in value and ")" in value

    cleaned = value.replace("(", "").replace(")", "")
    cleaned = cleaned.replace("−", "-").replace("\u2212", "-")
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = (
        cleaned.replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
        .replace("\u205F", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    marker_regex = re.compile(
        rf"^(?:{_CURRENCY_MARKER_PATTERN})(?:\s*)",
        re.IGNORECASE,
    )
    suffix_regex = re.compile(
        rf"(?:\s*)(?:{_CURRENCY_MARKER_PATTERN})$",
        re.IGNORECASE,
    )

    cleaned = marker_regex.sub("", cleaned).strip()
    cleaned = suffix_regex.sub("", cleaned).strip()

    leading_negative = cleaned.startswith("-")
    if cleaned.startswith("-") or cleaned.startswith("+"):
        cleaned = cleaned[1:].strip()

    negative = negative or leading_negative

    # Strip residual non-numeric prefixes/suffixes while preserving separators
    while cleaned and cleaned[0] not in "0123456789":
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] not in "0123456789,.'’ ":
        cleaned = cleaned[:-1]

    if not cleaned:
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    unsigned = cleaned

    grouped_match = _GROUPED_THOUSANDS_PATTERN.match(unsigned)
    normalized_number: Optional[str] = None

    if grouped_match:
        decimal_group = grouped_match.group(2)
        decimal_char = decimal_group[0] if decimal_group else None
        decimal_added = False
        digits: list[str] = []
        for ch in unsigned:
            if ch.isdigit():
                digits.append(ch)
            elif decimal_char and ch == decimal_char and not decimal_added:
                digits.append(".")
                decimal_added = True
            else:
                continue
        normalized_number = "".join(digits)
    else:
        compact = unsigned.replace(" ", "").replace("\u00A0", "").replace("\u202F", "").replace("\u2009", "")
        compact = compact.replace("'", "").replace("’", "")

        if "," in compact and "." in compact:
            last_comma = compact.rfind(",")
            last_dot = compact.rfind(".")
            if last_dot > last_comma:
                compact = compact.replace(",", "")
            else:
                compact = compact.replace(".", "")
                compact = compact.replace(",", ".")
        elif "," in compact:
            if compact.count(",") == 1 and 1 <= len(compact.split(",")[1]) <= 3:
                compact = compact.replace(",", ".", 1)
            else:
                compact = compact.replace(",", "")
        elif "." in compact and compact.count(".") > 1:
            last_dot = compact.rfind(".")
            compact = compact.replace(".", "")
            if last_dot != -1:
                compact = compact[:last_dot] + "." + compact[last_dot:]

        normalized_number = compact

    if not normalized_number or not re.search(r"\d", normalized_number):
        raise ValueError(f"unable to parse numeric value from '{num_str}'")

    number = float(normalized_number)
    if negative:
        number = -number
    return number


def _normalize_for_match(text: str) -> str:
    if not text:
        return text
    # NBSP -> space
    text = text.replace("\u00A0", " ")
    # unify unicode dashes to ASCII hyphen
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    # collapse "- / -" -> "-"
    text = re.sub(r"-\s*/\s*-", "-", text)
    # squeeze whitespace
    return re.sub(r"\s+", " ", text)


def _extract_number_after_keyword(text: str, keywords: Iterable[str]) -> Optional[float]:
    if not text:
        return None
    norm_text = _normalize_for_match(text)
    for keyword in keywords:
        k = _normalize_for_match(keyword)
        pattern = re.compile(re.escape(k), re.IGNORECASE)
        m = pattern.search(norm_text)
        if not m:
            continue
        remainder = norm_text[m.end() : m.end() + 8000]
        tokens = remainder.split()

        def _token_has_alpha(token: str) -> bool:
            stripped = token.strip()
            if not stripped:
                return False
            if re.fullmatch(r"&[A-Za-z]+;", stripped):
                return False
            return bool(re.search(r"[A-Za-z]", stripped))

        numeric_tokens: list[dict[str, object]] = []
        for idx, token in enumerate(tokens[:_TOKEN_LOOKAHEAD]):
            if not _NUMERIC_TOKEN_PATTERN.search(token):
                continue
            prev_token = tokens[idx - 1] if idx > 0 else ""
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            next2_token = tokens[idx + 2] if idx + 2 < len(tokens) else ""
            try:
                value = _to_float(token)
            except ValueError:
                continue

            digit_count = len(re.findall(r"\d", token))
            prev_stripped = prev_token.strip()
            next_stripped = next_token.strip()
            has_open = (
                "(" in token
                or "[" in token
                or prev_stripped.startswith("(")
                or prev_stripped.startswith("[")
            )
            has_close = (
                ")" in token
                or "]" in token
                or next_stripped.endswith(")")
                or next_stripped.endswith("]")
            )
            is_parenthetical = has_open and has_close
            is_negative = bool(
                value < 0
                or "-" in token
                or "−" in token
                or "\u2212" in token
                or is_parenthetical
            )
            has_separator = bool(re.search(r"[\.,\u00A0\u202F'’]", token))
            has_currency = any(symbol in token for symbol in "$€£¥₹₩₽")
            normalized_group = re.sub(r"[^0-9,\.\u00A0\u202F'’]", "", token)
            prev_has_alpha = _token_has_alpha(prev_token)
            next_has_alpha = _token_has_alpha(next_token)
            next2_has_alpha = _token_has_alpha(next2_token)

            numeric_tokens.append(
                {
                    "idx": idx,
                    "token": token,
                    "value": float(value),
                    "digit_count": digit_count,
                    "is_negative": is_negative,
                    "has_separator": has_separator,
                    "has_currency": has_currency,
                    "normalized_group": normalized_group,
                    "prev_has_alpha": prev_has_alpha,
                    "next_has_alpha": next_has_alpha,
                    "next2_has_alpha": next2_has_alpha,
                    "prev_token": prev_token,
                    "next_token": next_token,
                }
            )

        previous_idx_val: Optional[int] = None
        current_block = 0
        for token_info in numeric_tokens:
            idx_val = int(token_info["idx"])
            if previous_idx_val is not None and idx_val - previous_idx_val > 6:
                current_block += 1
            token_info["block_id"] = current_block
            previous_idx_val = idx_val

        matched_fragment = _preview_text(norm_text[m.start() : m.end()])
        _log_parse_event(
            logging.DEBUG,
            "caption_match",
            keyword=keyword,
            caption_match=matched_fragment,
            candidates=len(numeric_tokens),
        )

        if not numeric_tokens:
            continue

        strong_flags = [
            bool(
                token_info["digit_count"] >= 4
                or token_info["has_separator"]
                or token_info["has_currency"]
                or (
                    token_info["is_negative"]
                    and token_info["digit_count"] >= 3
                )
            )
            for token_info in numeric_tokens
        ]
        strong_suffix = [0] * len(numeric_tokens)
        running = 0
        for i in range(len(numeric_tokens) - 1, -1, -1):
            strong_suffix[i] = running
            if strong_flags[i]:
                running += 1

        scored_tokens: list[dict[str, object]] = []
        for i, token_info in enumerate(numeric_tokens):
            idx = int(token_info["idx"])  # safe cast
            token = str(token_info["token"])
            value = float(token_info["value"])
            digit_count = int(token_info["digit_count"])
            is_negative = bool(token_info["is_negative"])
            has_separator = bool(token_info["has_separator"])
            has_currency = bool(token_info["has_currency"])
            normalized_group = str(token_info["normalized_group"])
            strong_remaining = strong_suffix[i]
            prev_has_alpha = bool(token_info.get("prev_has_alpha"))
            next_has_alpha = bool(token_info.get("next_has_alpha"))
            next2_has_alpha = bool(token_info.get("next2_has_alpha"))
            prev_token_raw = str(token_info.get("prev_token") or "")
            next_token_raw = str(token_info.get("next_token") or "")
            prev_clean = re.sub(r"[^A-Za-z$€£¥₹₩₽]", "", prev_token_raw).upper()
            next_clean = re.sub(r"[^A-Za-z$€£¥₹₩₽]", "", next_token_raw).upper()
            currency_neighbor = prev_clean in _CURRENCY_CONTEXT_WORDS or next_clean in _CURRENCY_CONTEXT_WORDS
            block_id = int(token_info.get("block_id", 0))

            thousands_grouped = bool(
                normalized_group
                and _GROUPED_THOUSANDS_PATTERN.match(normalized_group)
            )

            footnote_like = False
            if digit_count <= 2 and abs(value) < 100:
                footnote_like = True
            elif (
                digit_count == 3
                and not is_negative
                and not has_separator
                and not has_currency
                and abs(value) < 200
                and strong_remaining >= 2
            ):
                footnote_like = True
            elif (
                digit_count == 4
                and not is_negative
                and not has_separator
                and not has_currency
                and 1900 <= abs(value) <= 2100
                and strong_remaining >= 1
            ):
                footnote_like = True
            elif prev_has_alpha and (next_has_alpha or next2_has_alpha) and not currency_neighbor:
                footnote_like = True

            big_pref = 0 if (
                digit_count >= 4
                or has_separator
                or has_currency
                or thousands_grouped
            ) else 1

            footnote_flag = 1 if footnote_like else 0
            neg_pref = 0 if is_negative else 1
            magnitude_rank = -abs(value)
            pos_rank = idx
            score = (footnote_flag, neg_pref, big_pref, pos_rank, magnitude_rank)

            token_info.update(
                {
                    "score": score,
                    "footnote_flag": footnote_flag,
                    "neg_pref": neg_pref,
                    "big_pref": big_pref,
                    "magnitude_rank": magnitude_rank,
                    "pos_rank": pos_rank,
                    "block_id": block_id,
                }
            )
            scored_tokens.append(token_info)

        sorted_for_log = sorted(
            scored_tokens,
            key=lambda info: (-abs(float(info["value"])), int(info["idx"])),
        )
        for entry in sorted_for_log[:5]:
            score_repr = _format_score_tuple(entry.get("score"))
            _log_parse_event(
                logging.DEBUG,
                "caption_candidate",
                keyword=keyword,
                candidate_token=str(entry["token"]),
                value=float(entry["value"]),
                score=score_repr,
            )

        if not scored_tokens:
            continue

        blocks_order = sorted({int(info.get("block_id", 0)) for info in scored_tokens})
        allowed_block: Optional[int] = None
        for block in blocks_order:
            if any(
                int(info.get("block_id", 0)) == block
                and int(info.get("footnote_flag", 0)) == 0
                for info in scored_tokens
            ):
                allowed_block = block
                break
        if allowed_block is None:
            allowed_block = blocks_order[0] if blocks_order else 0

        best_candidate: Optional[dict[str, object]] = None
        for info in scored_tokens:
            if int(info.get("block_id", 0)) != allowed_block:
                continue
            if best_candidate is None or info["score"] < best_candidate["score"]:
                best_candidate = info

        if best_candidate is not None:
            score_repr = _format_score_tuple(best_candidate.get("score"))
            _log_parse_event(
                logging.DEBUG,
                "caption_choice",
                keyword=keyword,
                chosen_token=str(best_candidate["token"]),
                score=score_repr,
                value=float(best_candidate["value"]),
            )
            return float(best_candidate["value"])
    return None


def _strip_html(html_text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_text or "")
    return " ".join(without_tags.split())


def _extract_html_section(
    html_text: str, headers: Iterable[object], window: int = 20000
) -> Optional[str]:
    if not html_text:
        return None
    lower = html_text.lower()
    for header in headers:
        if header is None:
            continue
        if isinstance(header, re.Pattern):
            pattern = header
        else:
            pattern = re.compile(re.escape(str(header)), re.IGNORECASE)
        search_from = 0
        while True:
            match = pattern.search(html_text, search_from)
            if not match:
                break
            idx = match.start()
            end = min(len(html_text), idx + window)
            close_idx = lower.find("</table", idx)
            if close_idx != -1:
                close_end = lower.find(">", close_idx)
                if close_end != -1 and close_end + 1 > end:
                    end = min(len(html_text), close_end + 1)
            snippet = html_text[idx:end]
            snippet_lower = snippet.lower()
            if "<table" in snippet_lower or "<tr" in snippet_lower:
                return snippet
            search_from = match.end()
    return None


def _parse_html_cashflow_sections(html_text: str, *, context_url: Optional[str] = None) -> dict:
    _log_parse_event(logging.DEBUG, "runway: parse_html start", url=context_url)

    text = _strip_html(html_text)
    defaults = _form_defaults(None)

    cashflow_headers = list(defaults["cashflow_headers"])
    header_candidates = cashflow_headers + list(_CASHFLOW_HEADER_PATTERNS)

    balance_section_html = _extract_html_section(html_text, _BALANCE_HEADERS, window=60000)
    cashflow_section_html = _extract_html_section(html_text, header_candidates, window=120000)

    balance_section_text = _strip_html(balance_section_html) if balance_section_html else None
    cashflow_section_text = _strip_html(cashflow_section_html) if cashflow_section_html else None

    balance_scale = find_scale_near(balance_section_html, html_text) if balance_section_html else 1
    cashflow_scale = find_scale_near(cashflow_section_html, html_text) if cashflow_section_html else 1
    default_scale = find_scale_near(html_text, html_text)

    burn_keywords = defaults["ocf_keywords_burn"]
    provided_keywords = defaults["ocf_keywords_provided"]

    normalized_burn = [_normalize_for_match(k).lower() for k in burn_keywords]
    normalized_provided = [_normalize_for_match(k).lower() for k in provided_keywords]
    normalized_cash_flow_keywords = [_normalize_for_match(k).lower() for k in _CASH_KEYWORDS_FLOW]

    def _rows_from_table(table_html: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_match in _ROW_PATTERN.finditer(table_html):
            row_html = row_match.group(0)
            cells = [_strip_html(cell) for cell in _CELL_PATTERN.findall(row_html)]
            if cells:
                rows.append(cells)
        return rows

    tables: list[dict[str, object]] = []
    if cashflow_section_html:
        for idx, table_match in enumerate(_TABLE_PATTERN.finditer(cashflow_section_html), start=1):
            table_html = table_match.group(0)
            rows = _rows_from_table(table_html)
            if not rows:
                continue
            table_scale = find_scale_near(table_html, html_text)
            if not table_scale:
                table_scale = 1
            tables.append({"index": idx, "rows": rows, "scale": int(table_scale)})

    cash_value: Optional[float] = None
    cash_scale_used = 1

    if balance_section_text:
        balance_match = _extract_number_after_keyword(balance_section_text, _CASH_KEYWORDS_BALANCE)
        if balance_match is not None:
            cash_value = float(balance_match) * float(balance_scale)
            cash_scale_used = balance_scale

    def _match_cash_from_rows(rows: list[list[str]]) -> Optional[float]:
        prev_cash_label = False

        def _row_numbers(row: list[str]) -> list[float]:
            combined = " ".join(str(part or "") for part in row[1:])
            values: list[float] = []
            for match in _NUMBER_SEARCH_PATTERN.finditer(combined):
                token = match.group(0)
                try:
                    values.append(float(_to_float(token)))
                except ValueError:
                    continue
            return values

        for row in rows:
            normalized_row = _normalize_for_match(" ".join(row)).lower()
            if not normalized_row:
                prev_cash_label = False
                continue

            has_cash_phrase = "cash and cash equivalents" in normalized_row
            has_restricted_phrase = "cash & restricted cash" in normalized_row or "cash and restricted cash" in normalized_row
            if "net decrease" in normalized_row or "net increase" in normalized_row:
                prev_cash_label = False
                continue

            row_numbers = _row_numbers(row)

            end_tokens = {"end of period", "end of the period", "end of year", "end of the year"}
            has_end_marker = any(token in normalized_row for token in end_tokens)

            if has_end_marker and row_numbers:
                if has_cash_phrase or has_restricted_phrase or prev_cash_label:
                    return row_numbers[0]

            if (has_cash_phrase or has_restricted_phrase) and not has_end_marker:
                prev_cash_label = True
                continue

            prev_cash_label = False

        return None

    for table in tables:
        rows_obj = table.get("rows")
        if not isinstance(rows_obj, list):
            continue
        value = _match_cash_from_rows(rows_obj)
        if value is not None:
            scale_val = int(table.get("scale", 1))
            cash_value = float(value) * float(scale_val)
            cash_scale_used = scale_val
            break

    if cash_value is None and cashflow_section_text:
        flow_match = _extract_number_after_keyword(cashflow_section_text, _CASH_KEYWORDS_FLOW)
        if flow_match is not None:
            cash_value = float(flow_match) * float(cashflow_scale)
            cash_scale_used = cashflow_scale
    if cash_value is None:
        text_match = _extract_number_after_keyword(text, _CASH_KEYWORDS_FLOW)
        if text_match is not None:
            cash_value = float(text_match) * float(default_scale)
            cash_scale_used = default_scale

    ocf_value: Optional[float] = None
    ocf_scale_used = cashflow_scale or default_scale or 1

    if tables:
        for table in tables:
            rows_obj = table.get("rows")
            if not isinstance(rows_obj, list):
                continue
            matched_in_table = False
            normalized_index = table.get("index")
            for row in rows_obj:
                if not isinstance(row, list):
                    continue
                normalized_row = _normalize_for_match(" ".join(row)).lower()
                if not normalized_row:
                    continue
                keyword_type = None
                if any(keyword and keyword in normalized_row for keyword in normalized_burn):
                    keyword_type = "burn"
                elif any(keyword and keyword in normalized_row for keyword in normalized_provided):
                    keyword_type = "provided"
                elif _OCF_IFRS_PATTERN.search(normalized_row):
                    keyword_type = "ifrs"
                else:
                    continue
                combined = " ".join(str(part or "") for part in row[1:])
                numeric_values: list[float] = []
                for match in _NUMBER_SEARCH_PATTERN.finditer(combined):
                    token = match.group(0)
                    try:
                        numeric_values.append(float(_to_float(token)))
                    except ValueError:
                        continue
                if not numeric_values:
                    continue
                adjusted = float(numeric_values[0])
                if adjusted > 0 and ("used" in normalized_row or "outflow" in normalized_row or keyword_type == "burn"):
                    adjusted = -abs(adjusted)
                elif adjusted > 0 and "generated" in normalized_row and "used" not in normalized_row:
                    adjusted = abs(adjusted)
                scale_val = int(table.get("scale", 1))
                ocf_value = adjusted * float(scale_val)
                ocf_scale_used = scale_val
                matched_in_table = True
                break
            if matched_in_table:
                break
            else:
                _log_parse_event(
                    logging.DEBUG,
                    f"miss: no 'Net cash ... operating activities' row in table {normalized_index}",
                    url=context_url,
                )

    if ocf_value is None:
        candidate_texts = [cashflow_section_text, text]

        def _first_match(texts: list[Optional[str]], keywords: Iterable[str]) -> Optional[float]:
            for candidate_text in texts:
                if not candidate_text:
                    continue
                value = _extract_number_after_keyword(candidate_text, keywords)
                if value is not None:
                    return value
            return None

        burn_value = _first_match(candidate_texts, burn_keywords)
        provided_value = _first_match(candidate_texts, provided_keywords)

        ocf_candidate = burn_value
        if ocf_candidate is None:
            ocf_candidate = provided_value
        elif burn_value is not None and burn_value > 0:
            ocf_candidate = -abs(burn_value)

        if burn_value is not None and burn_value > 0:
            burn_value = -abs(float(burn_value))

        scale_for_text = cashflow_scale or default_scale or 1

        if ocf_candidate is not None:
            if burn_value is not None:
                ocf_value = float(burn_value) * float(scale_for_text)
            else:
                ocf_value = float(ocf_candidate) * float(scale_for_text)
            ocf_scale_used = scale_for_text

    html_ocf_raw_value: Optional[float] = float(ocf_value) if ocf_value is not None else None

    period_months_inferred = _infer_months_from_text(cashflow_section_text) or _infer_months_from_text(text)

    html_units_scale = ocf_scale_used if ocf_value is not None else cash_scale_used or default_scale or 1

    evidence_parts = [
        f"cashflow_header={'yes' if cashflow_section_html else 'no'}",
        f"cash_found={'yes' if cash_value is not None else 'no'}",
        f"ocf_found={'yes' if ocf_value is not None else 'no'}",
    ]
    if html_units_scale and html_units_scale != 1:
        evidence_parts.append(f"scale={html_units_scale}")

    header_flag = "yes" if cashflow_section_html else "no"
    cash_flag = "yes" if cash_value is not None else "no"
    ocf_flag = "yes" if ocf_value is not None else "no"
    _log_parse_event(
        logging.DEBUG,
        f"runway: cashflow_header={header_flag} ocf_found={ocf_flag} cash_found={cash_flag} scale={html_units_scale}",
        url=context_url,
    )

    return {
        "found_cashflow_header": bool(cashflow_section_html),
        "cash_value": cash_value,
        "ocf_value": html_ocf_raw_value,
        "period_months_inferred": period_months_inferred,
        "units_scale": html_units_scale,
        "evidence": "; ".join(evidence_parts),
        "html_header": bool(cashflow_section_html),
    }


def _looks_like_exhibit_path(path: str) -> bool:
    if not path:
        return False
    last_segment = path.split("/")[-1]
    return bool(re.search(r"(?:ex|exhibit)99", last_segment, re.IGNORECASE))


def _extract_exhibit_candidates(index_html: str) -> list[dict]:
    candidates: list[dict] = []
    if not index_html:
        return candidates
    for row_match in _ROW_PATTERN.finditer(index_html):
        row_html = row_match.group(0)
        row_lower = row_html.lower()
        if not any(token in row_lower for token in ("ex-99", "ex99", "exhibit 99", "ex_99")):
            continue
        cells = list(_CELL_PATTERN.findall(row_html))
        if not cells:
            continue
        href: Optional[str] = None
        for cell_html in cells:
            link_match = re.search(r'href\s*=\s*"([^"]+)"', cell_html, re.IGNORECASE)
            if link_match:
                href = html.unescape(link_match.group(1))
                break
        if not href:
            continue
        texts = [_strip_html(cell_html) for cell_html in cells]
        doc_type = ""
        description = ""
        for text_part in texts:
            cleaned = (text_part or "").strip()
            if not cleaned:
                continue
            if not doc_type:
                doc_match = re.search(r"EX[\s\-_.]*99[\w\.\-]*", cleaned, re.IGNORECASE)
                if doc_match:
                    doc_type = doc_match.group(0).replace(" ", "").upper()
            if not description and len(cleaned) > 3:
                description = cleaned
        filename = href.split("/")[-1]
        candidates.append(
            {
                "href": href,
                "doc_type": doc_type or filename,
                "description": description,
                "filename": filename,
                "texts": texts,
            }
        )
    return candidates


def _score_exhibit_candidate(candidate: dict, form_type: Optional[str]) -> int:
    doc_type_raw = str(candidate.get("doc_type") or "")
    doc_type = doc_type_raw.upper()
    filename = str(candidate.get("filename") or "").lower()
    href = str(candidate.get("href") or "").lower()
    description_parts = candidate.get("texts") or []
    description_combined = " ".join(str(part or "") for part in description_parts)
    description_lower = description_combined.lower()

    normalized_doc = re.sub(r"[^a-z0-9]", "", doc_type.lower())
    normalized_filename = re.sub(r"[^a-z0-9]", "", filename)
    normalized_href = re.sub(r"[^a-z0-9]", "", href)

    score = 0

    if doc_type.startswith("EX-99"):
        score += 50

    for idx, pattern in enumerate(_EXHIBIT_FILENAME_PRIORITY):
        normalized_pattern = re.sub(r"[^a-z0-9]", "", pattern.lower())
        if (
            normalized_pattern in normalized_doc
            or normalized_pattern in normalized_filename
            or normalized_pattern in normalized_href
        ):
            score += (len(_EXHIBIT_FILENAME_PRIORITY) - idx) * 10

    financial_keywords = [
        "financial statements",
        "financial report",
        "audited financial",
        "condensed consolidated",
        "interim financial report",
        "interim consolidated",
    ]
    if any(keyword in description_lower for keyword in financial_keywords):
        score += 40
    if "interim" in description_lower:
        score += 10

    normalized_form = _normalize_form_type(form_type)
    if normalized_form == "40-F":
        if doc_type.startswith("EX-99.2"):
            score += 1000
        elif doc_type.startswith(("EX-99.5", "EX-99.1", "EX-99.3")) and (
            "financial statements" in description_lower or "interim" in description_lower
        ):
            score += 900
    elif normalized_form == "6-K":
        if doc_type.startswith("EX-99.1") and (
            "financial statements" in description_lower
            or "interim financial report" in description_lower
            or "condensed consolidated" in description_lower
        ):
            score += 900
    elif normalized_form == "20-F":
        if doc_type.startswith("EX-99") and "financial statements" in description_lower:
            score += 800

    return score


def follow_exhibits_and_parse(filing_url: str, html: Optional[str]) -> dict:
    try:
        parsed = urlparse(filing_url)
    except Exception:
        parsed = None

    path = parsed.path if parsed else ""
    if _looks_like_exhibit_path(path or ""):
        return {}

    cleaned_url = (filing_url or "").split("#", 1)[0]
    if "/" not in cleaned_url:
        return {}
    base_dir = cleaned_url.rsplit("/", 1)[0] + "/"

    form_candidates = []
    if html:
        try:
            form_candidates.append(_infer_form_type_from_text(_strip_html(html)))
        except Exception:
            pass
    form_candidates.append(_infer_form_type_from_url(filing_url))

    form_type: Optional[str] = None
    for candidate in form_candidates:
        normalized = _normalize_form_type(candidate)
        if normalized:
            form_type = normalized
            break

    index_html: Optional[str] = None
    index_url_used: Optional[str] = None
    for suffix in ("index.html", "index.htm"):
        candidate_url = urljoin(base_dir, suffix)
        try:
            raw_index = _fetch_url(candidate_url)
        except Exception:
            continue
        try:
            index_html = raw_index.decode("utf-8", errors="ignore")
        except Exception:
            continue
        index_url_used = candidate_url
        break

    if not index_html:
        try:
            parsed = urlparse(filing_url)
            if parsed.scheme == "file":
                from pathlib import Path

                base_path = Path(unquote(parsed.path or "")).parent
                glob_patterns = [
                    "ex99*.*",
                    "ex-99*.*",
                    "ex_99*.*",
                    "exhibit99*.*",
                    "exhibit_99*.*",
                ]
                file_hits = []
                for pat in glob_patterns:
                    file_hits.extend(base_path.glob(pat))
                local_candidates = []
                for p in sorted(set(file_hits)):
                    href = p.as_uri()
                    filename = p.name
                    local_candidates.append(
                        {
                            "href": href,
                            "doc_type": filename.upper(),
                            "description": filename,
                            "filename": filename,
                            "texts": [filename],
                        }
                    )
                candidates = local_candidates
            else:
                return {}
        except Exception:
            return {}
    else:
        index_html = _unescape_html_entities(index_html, context=index_url_used)
        candidates = _extract_exhibit_candidates(index_html)

    if not candidates:
        return {}

    for candidate in candidates:
        candidate["score"] = _score_exhibit_candidate(candidate, form_type)
        candidate["absolute_href"] = urljoin(base_dir, candidate.get("href", ""))

    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item.get("score", 0)),
            str(item.get("doc_type") or ""),
            str(item.get("filename") or ""),
        ),
        reverse=True,
    )

    labels = [str(item.get("doc_type") or item.get("filename") or item.get("href")) for item in ranked]
    _log_parse_event(
        logging.DEBUG,
        f"runway: exhibit index parsed, candidates={[label for label in labels if label]}",
        url=index_url_used,
    )

    defaults = _form_defaults(form_type)
    fallback_period = defaults.get("period_months_default")

    for candidate in ranked:
        exhibit_url = str(candidate.get("absolute_href") or "")
        if not exhibit_url:
            continue
        doc_label = str(candidate.get("doc_type") or candidate.get("filename") or exhibit_url).strip()
        try:
            raw_html = _fetch_url(exhibit_url)
        except Exception as exc:
            _log_parse_event(
                logging.DEBUG,
                f"runway: exhibit fetch failed {doc_label}",
                url=exhibit_url,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            continue

        exhibit_html = raw_html.decode("utf-8", errors="ignore")
        exhibit_html = _unescape_html_entities(exhibit_html, context=exhibit_url)
        parsed_html = _parse_html_cashflow_sections(exhibit_html, context_url=exhibit_url)

        cash_val = parsed_html.get("cash_value")
        ocf_val = parsed_html.get("ocf_value")
        period_inferred = parsed_html.get("period_months_inferred")
        units_scale = parsed_html.get("units_scale") or 1
        evidence = parsed_html.get("evidence") or ""

        if period_inferred not in {3, 6, 9, 12}:
            fallback = fallback_period if fallback_period in {3, 6, 9, 12} else None
            if fallback is not None:
                period_inferred = fallback
            else:
                inferred_from_text = _infer_months_from_text(_strip_html(exhibit_html))
                if inferred_from_text in {3, 6, 9, 12}:
                    period_inferred = inferred_from_text

        evidence_parts = [part for part in (evidence,) if part]
        if candidate.get("doc_type"):
            evidence_parts.append(f"exhibit={candidate.get('doc_type')}")
        evidence_full = "; ".join(evidence_parts)

        label_lower = doc_label.lower() if doc_label else exhibit_url

        if cash_val is not None and ocf_val is not None and period_inferred in {3, 6, 9, 12}:
            _log_parse_event(
                logging.DEBUG,
                f"runway: trying exhibit {label_lower} -> success",
                url=exhibit_url,
            )
            status = "OK (from exhibit)"
            if candidate.get("doc_type"):
                status = f"OK (from exhibit {candidate.get('doc_type')})"
            return {
                "source": "exhibit",
                "cash_value": cash_val,
                "ocf_value": ocf_val,
                "period_months": period_inferred,
                "units_scale": units_scale,
                "evidence": evidence_full,
                "found_cashflow_header": parsed_html.get("found_cashflow_header"),
                "html_info": parsed_html,
                "status": status,
                "exhibit_href": exhibit_url,
                "exhibit_doc_type": candidate.get("doc_type"),
            }

        if cash_val is None or ocf_val is None:
            _log_parse_event(
                logging.DEBUG,
                f"runway: exhibit {label_lower} had no cash-flow table",
                url=exhibit_url,
                cash_found=cash_val is not None,
                ocf_found=ocf_val is not None,
            )
        else:
            _log_parse_event(
                logging.DEBUG,
                f"runway: exhibit {label_lower} missing period",
                url=exhibit_url,
                period=period_inferred,
            )

    return {}


def _infer_months_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for pattern, months in _PERIOD_TEXT_PATTERNS:
        if pattern.search(text):
            return months
    return None


def _normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _extract_filing_identifiers(filing_url: str) -> tuple[Optional[str], Optional[str]]:
    parsed = urlparse(filing_url)
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    doc_path = path
    if "doc" in query and query["doc"]:
        doc_path = query["doc"][0]
    doc_path = unquote(doc_path)

    match = _ACCESSION_RE.search(doc_path) or _ACCESSION_RE.search(path)
    if not match:
        return None, None

    cik_digits = _normalize_digits(match.group(1))
    accession_digits = _normalize_digits(match.group(2))
    if not cik_digits or not accession_digits:
        return None, None

    cik_padded = cik_digits.zfill(10)
    if len(accession_digits) < 18:
        accession_digits = accession_digits.zfill(18)
    accession = f"{accession_digits[:10]}-{accession_digits[10:12]}-{accession_digits[12:]}"
    return cik_padded, accession


def _normalize_form_type(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    match = _FORM_TYPE_PATTERN.search(text)
    if not match:
        return None
    normalized = match.group(1).upper()
    canonical_map = {
        "10-Q": "10-Q",
        "10-QT": "10-Q",
        "10-K": "10-K",
        "10-KT": "10-K",
        "20-F": "20-F",
        "40-F": "40-F",
        "6-K": "6-K",
    }
    return canonical_map.get(normalized)


def _extract_form_hint_from_url(filing_url: str) -> Optional[str]:
    try:
        parsed = urlparse(filing_url)
    except Exception:
        return None
    query = parse_qs(parsed.query)
    for key, values in query.items():
        if key.lower() != "form":
            continue
        for value in values:
            normalized = _normalize_form_type(value)
            if normalized:
                return normalized
    return None


def _infer_form_type_from_url(filing_url: str) -> Optional[str]:
    hint = _extract_form_hint_from_url(filing_url)
    if hint:
        return hint
    try:
        parsed = urlparse(filing_url)
    except Exception:
        return None
    components = [parsed.path or ""]
    query = parse_qs(parsed.query)
    for values in query.values():
        components.extend(values)
    combined = " ".join(components)
    return _normalize_form_type(combined)


def _infer_form_type_from_text(text: str) -> Optional[str]:
    return _normalize_form_type(text)


def _form_defaults(form_type: Optional[str]) -> dict:
    normalized = _normalize_form_type(form_type)
    adapter = FORM_ADAPTERS.get(normalized) if normalized else None

    period_default: Optional[int] = None
    if adapter in {"US_Annual", "Foreign_Annual"}:
        period_default = 12
    elif adapter in {"US_Quarterly", "SixK"}:
        period_default = 3

    burn_keywords = list(_OCF_KEYWORDS_BURN_BASE)
    burn_keywords.extend(_OCF_KEYWORDS_BURN_EXTRA)

    provided_keywords = list(_OCF_KEYWORDS_PROVIDED_BASE)
    provided_keywords.extend(_OCF_KEYWORDS_PROVIDED_EXTRA)

    return {
        "period_months_default": period_default,
        "ocf_keywords_burn": burn_keywords,
        "ocf_keywords_provided": provided_keywords,
        "cashflow_headers": list(_CASHFLOW_HEADERS_BASE),
        "cashflow_header_patterns": list(_CASHFLOW_HEADER_PATTERNS),
    }


def _extract_note_suffix(note: Optional[str]) -> list[str]:
    if not note:
        return []
    text = str(note)
    prefix = "values parsed from XBRL"
    lowered = text.lower()
    prefix_lower = prefix.lower()
    if lowered.startswith(prefix_lower):
        colon_idx = text.find(":")
        if colon_idx != -1:
            remainder = text[colon_idx + 1 :].strip()
        else:
            remainder = text[len(prefix) :].strip()
        return [remainder] if remainder else []
    return [text]


def url_matches_form(url: str, form: str | None) -> bool:
    normalized = _normalize_form_type(form)
    if not normalized:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    components = [parsed.path or ""]
    query = parse_qs(parsed.query)
    for values in query.values():
        components.extend(values)
    combined = " ".join(components).upper().replace("-", "")
    target = normalized.upper().replace("-", "")
    if target in combined:
        return True

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if host.endswith("sec.gov"):
        if any(path.startswith(prefix) for prefix in ("/ix", "/ixviewer", "/cgi-bin/viewer")):
            return True

    return False


def _parse_fact_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        pass
    else:
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    for fmt in _FACT_DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _select_best_fact(
    facts: Iterable[dict], accession: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    best_value: Optional[float] = None
    best_form: Optional[str] = None
    best_fact: Optional[dict] = None
    best_key: tuple[int, datetime, int] | None = None
    for fact in facts:
        val = fact.get("val")
        if not isinstance(val, (int, float)):
            continue
        fact_accn = (fact.get("accn") or "").strip()
        match_score = 1 if accession and fact_accn == accession else 0
        end_dt = _parse_fact_timestamp(
            fact.get("end") or fact.get("filed") or fact.get("instant")
        )
        if end_dt is None:
            end_dt = datetime.min.replace(tzinfo=timezone.utc)
        fy_value = fact.get("fy")
        try:
            fy_score = int(fy_value)
        except (TypeError, ValueError):
            fy_score = 0
        key = (match_score, end_dt, fy_score)
        if best_key is None or key > best_key:
            best_key = key
            best_value = float(val)
            best_form = _normalize_form_type(fact.get("form"))
            best_fact = fact
    return best_value, best_form, best_fact


def _extract_fact_value(
    units: dict, accession: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    preferred_units = ["USD", "USD ", "USD millions", "USDm", "USD $", "USD $ millions"]
    for unit_name in preferred_units:
        facts = units.get(unit_name)
        if not facts:
            continue
        value, form_type, fact = _select_best_fact(facts, accession)
        if value is not None:
            return value, form_type, fact

    all_facts: list[dict] = []
    for facts in units.values():
        all_facts.extend(facts)
    if not all_facts:
        return None, None, None
    return _select_best_fact(all_facts, accession)


def _fetch_xbrl_value(
    cik: str, accession: str, concept: str
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    base_url = "https://data.sec.gov/api/xbrl/companyconcept"
    taxonomies = ["us-gaap", "ifrs-full"]
    for taxonomy in taxonomies:
        url = f"{base_url}/CIK{cik}/{taxonomy}/{concept}.json"
        try:
            data = _fetch_json(url)
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP error fetching JSON (404)" in message:
                continue
            raise

        units = data.get("units") or {}
        value, form_type, fact = _extract_fact_value(units, accession)
        if value is not None:
            return value, form_type, fact

    return None, None, None


def _derive_from_xbrl(
    filing_url: str, form_type_hint: Optional[str] = None
) -> tuple[Optional[dict], Optional[str]]:
    cik, accession = _extract_filing_identifiers(filing_url)
    if not cik or not accession:
        return None, form_type_hint

    cash = None
    form_type = form_type_hint
    cash_fact: Optional[dict] = None
    cash_concepts = [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsIncludingRestrictedCashCurrent",
        "CashAndCashEquivalentsAndOtherShortTermInvestments",
    ]
    errors: list[str] = []

    for concept in cash_concepts:
        try:
            cash_value, value_form, fact = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if cash_value is not None:
            cash = cash_value
            if form_type is None and value_form:
                form_type = value_form
            cash_fact = fact
            break

    burn = None
    burn_fact: Optional[dict] = None
    burn_concepts = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivitiesExcludingDiscontinuedOperations",
        "NetCashProvidedByUsedInOperatingActivitiesConverted",
        "NetCashFlowsFromUsedInOperatingActivities",
        "NetCashFlowsFromOperatingActivities",
        "NetCashFlowsFromOperations",
    ]
    for concept in burn_concepts:
        try:
            burn_value, value_form, fact = _fetch_xbrl_value(cik, accession, concept)
        except Exception as exc:
            errors.append(f"{concept}: {exc}")
            continue
        if burn_value is not None:
            burn = burn_value
            if form_type is None and value_form:
                form_type = value_form
            burn_fact = fact
            break

    if cash is None and burn is None:
        if errors:
            raise RuntimeError("; ".join(errors))
        return None, form_type

    period_months = _infer_period_months(burn_fact, form_type)
    ocf_normalized, period_months, assumption = _normalize_ocf_value(burn, period_months)

    status = "OK"
    if ocf_normalized is None:
        status = "Missing OCF"
    elif period_months not in {3, 6, 9, 12}:
        status = "Missing OCF period"

    result = _finalize_runway_result(
        cash=float(cash) if cash is not None else None,
        ocf_raw=float(burn) if burn is not None else None,
        ocf_quarterly=ocf_normalized,
        period_months=period_months,
        assumption=assumption,
        note=f"values parsed from XBRL: {filing_url}" + (" (partial XBRL data)" if cash is None or burn is None else ""),
        form_type=form_type,
        units_scale=1,
        status=status,
        source_tags=["XBRL"],
    )
    return result, form_type


def _infer_runway_estimate(form_type: Optional[str]) -> str:
    if not form_type:
        return ""
    normalized = form_type.upper()
    if normalized.startswith(("10-K", "20-F", "40-F")):
        return "annual_div4"
    return "interim"


def _finalize_runway_result(
    *,
    cash: Optional[float],
    ocf_raw: Optional[float],
    ocf_quarterly: Optional[float] = None,
    period_months: Optional[int],
    assumption: Optional[str],
    note: str,
    form_type: Optional[str],
    units_scale: Optional[int],
    status: Optional[str],
    source_tags: Optional[Iterable[str]] = None,
) -> dict:
    estimate = _infer_runway_estimate(form_type)

    status_final = status or ""

    quarterly_burn_raw: Optional[float] = None
    runway_quarters_raw: Optional[float] = None
    show_quarters = True

    ocf_for_burn = None
    if ocf_quarterly is not None:
        ocf_for_burn = float(ocf_quarterly)
    elif ocf_raw is not None:
        ocf_for_burn = float(ocf_raw)

    if ocf_for_burn is None:
        show_quarters = False
        if not status_final:
            status_final = "Missing OCF"
    else:
        if ocf_for_burn >= 0:
            quarterly_burn_raw = 0.0
            show_quarters = False
            status_final = "OCF positive (self-funding)"
        else:
            quarterly_burn_raw = abs(ocf_for_burn)
            if cash is not None and quarterly_burn_raw > 0:
                runway_quarters_raw = float(cash) / float(quarterly_burn_raw)
            if not status_final:
                status_final = "OK"

    cash_value = float(cash) if cash is not None else None

    runway_quarters_display: Optional[float] = None
    runway_months_display: Optional[float] = None
    if show_quarters and runway_quarters_raw is not None:
        runway_quarters_display = round(float(runway_quarters_raw), 2)
        runway_months_display = round(float(runway_quarters_raw) * 3, 2)

    result: dict = {
        "cash_raw": cash_value,
        "ocf_raw": ocf_raw,
        "ocf_quarterly_raw": ocf_for_burn,
        "quarterly_burn_raw": quarterly_burn_raw,
        "runway_quarters_raw": runway_quarters_raw,
        "runway_quarters_display": runway_quarters_display,
        "runway_months_display": runway_months_display,
        "note": note,
        "estimate": estimate,
        "status": status_final,
        "assumption": assumption or "",
        "period_months": period_months,
        "units_scale": units_scale or 1,
    }

    result["cash"] = _round_half_up(cash_value)
    result["quarterly_burn"] = (
        _round_half_up(quarterly_burn_raw) if quarterly_burn_raw is not None else None
    )
    result["runway_quarters"] = (
        _round_half_up(runway_quarters_raw)
        if show_quarters and runway_quarters_raw is not None
        else None
    )

    if form_type:
        result["form_type"] = form_type

    months_valid = period_months in {3, 6, 9, 12}
    complete = (cash_value is not None and ocf_raw is not None and months_valid)
    result["complete"] = complete

    if not complete and status_final in {"", "OK"}:
        status_final = "Incomplete"

    result["status"] = status_final

    if source_tags:
        unique_tags = []
        for tag in source_tags:
            if tag and tag not in unique_tags:
                unique_tags.append(tag)
        if unique_tags:
            result["source_tags"] = unique_tags

    return result


def _log_runway_outcome(
    canonical_url: str,
    form_type: Optional[str],
    result: dict,
    *,
    html_info: Optional[dict] = None,
    xbrl_result: Optional[dict] = None,
    xbrl_error: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    sources = None
    if "source_tags" in result:
        tags = result.get("source_tags") or []
        if isinstance(tags, (list, tuple)):
            sources = ",".join(str(tag) for tag in tags if tag)
        else:
            sources = str(tags)

    fields: dict[str, object] = {
        "url": canonical_url,
        "form_type": form_type or result.get("form_type"),
        "status": result.get("status"),
        "complete": result.get("complete"),
        "cash_raw": result.get("cash_raw"),
        "ocf_raw": result.get("ocf_raw"),
        "period_months": result.get("period_months"),
        "sources": sources,
    }

    if html_info is not None:
        fields.update(
            {
                "html_header": html_info.get("found_cashflow_header"),
                "html_cash": html_info.get("cash_value"),
                "html_ocf": html_info.get("ocf_value"),
                "html_period_inferred": html_info.get("period_months_inferred"),
                "html_units_scale": html_info.get("units_scale"),
                "html_evidence": html_info.get("evidence"),
            }
        )

    if xbrl_result is not None:
        fields.update(
            {
                "xbrl_cash": xbrl_result.get("cash_raw"),
                "xbrl_ocf": xbrl_result.get("ocf_raw"),
                "xbrl_period": xbrl_result.get("period_months"),
            }
        )

    if xbrl_error:
        fields["xbrl_error"] = xbrl_error

    if extra:
        fields.update({k: v for k, v in extra.items()})

    note = result.get("note")
    if isinstance(note, str):
        sanitized = note.replace("\n", " ")
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."
        fields["note"] = sanitized

    level = logging.INFO if result.get("complete") else logging.WARNING
    _log_parse_event(level, "runway parse outcome", **fields)


def get_runway_from_filing(filing_url: str) -> dict:
    original_input = filing_url
    canonical_url = _canonicalize_sec_filing_url(filing_url)

    html_text: Optional[str] = None
    is_local_file = False
    local_file_path: Optional[Path] = None

    parsed_initial = urlparse(canonical_url)
    if parsed_initial.scheme == "file":
        file_path = unquote(parsed_initial.path or "")
        if parsed_initial.netloc:
            file_path = f"//{parsed_initial.netloc}{file_path}"
        candidate = Path(file_path)
        if candidate.exists():
            resolved = candidate.resolve()
            is_local_file = True
            local_file_path = resolved
            base_uri = resolved.as_uri()
            if parsed_initial.query:
                canonical_url = f"{base_uri}?{parsed_initial.query}"
            else:
                canonical_url = base_uri
            parsed_initial = urlparse(canonical_url)
    elif not parsed_initial.scheme and not parsed_initial.netloc:
        path_only = parsed_initial.path or canonical_url
        query_only = parsed_initial.query
        candidate = Path(unquote(path_only))
        if candidate.exists():
            resolved = candidate.resolve()
            is_local_file = True
            local_file_path = resolved
            base_uri = resolved.as_uri()
            canonical_url = f"{base_uri}?{query_only}" if query_only else base_uri
            parsed_initial = urlparse(canonical_url)

    if canonical_url != original_input:
        _log_parse_event(
            logging.DEBUG,
            "canonical_url",
            original=original_input,
            canonical=canonical_url,
        )

    if is_local_file and local_file_path is not None:
        _log_parse_event(
            logging.DEBUG,
            "offline html detected",
            offline_html=str(local_file_path),
        )
        try:
            html_text = local_file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read local filing HTML: {local_file_path} ({exc})"
            ) from exc

    form_hint_query = _extract_form_hint_from_url(canonical_url)
    form_type_hint = _infer_form_type_from_url(canonical_url)

    xbrl_result: Optional[dict] = None
    xbrl_error: Optional[str] = None
    detected_form_type: Optional[str] = None

    if not is_local_file:
        try:
            xbrl_result, detected_form_type = _derive_from_xbrl(
                canonical_url, form_type_hint=form_type_hint
            )
        except Exception as exc:
            xbrl_error = f"XBRL parse failed ({exc.__class__.__name__}: {exc})"
            _log_parse_event(
                logging.WARNING,
                "runway parse XBRL failure",
                url=canonical_url,
                error=xbrl_error,
            )

    if xbrl_result and xbrl_error:
        xbrl_result = dict(xbrl_result)
        xbrl_result["note"] = f"{xbrl_result['note']} (with warning: {xbrl_error})"

    if xbrl_result and xbrl_result.get("complete"):
        result_copy = dict(xbrl_result)
        if not result_copy.get("source_tags"):
            result_copy["source_tags"] = ["XBRL"]
        _log_runway_outcome(
            canonical_url,
            detected_form_type or form_type_hint,
            result_copy,
            xbrl_result=xbrl_result,
            xbrl_error=xbrl_error,
            extra={"path": "xbrl_complete"},
        )
        return result_copy

    if html_text is None:
        try:
            raw_bytes = _fetch_url(canonical_url)
        except HTTPError as exc:
            _log_parse_event(
                logging.ERROR,
                "runway fetch HTTP error",
                url=canonical_url,
                status_code=exc.code,
            )
            raise RuntimeError(
                f"HTTP error fetching filing ({exc.code}): {canonical_url}"
            ) from exc
        except URLError as exc:
            _log_parse_event(
                logging.ERROR,
                "runway fetch URL error",
                url=canonical_url,
                reason=getattr(exc, "reason", exc),
            )
            raise RuntimeError(
                f"URL error fetching filing ({exc.reason}): {canonical_url}"
            ) from exc
        except Exception as exc:
            _log_parse_event(
                logging.ERROR,
                "runway fetch unexpected error",
                url=canonical_url,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise RuntimeError(
                f"Unexpected error fetching filing: {canonical_url} ({exc.__class__.__name__}: {exc})"
            ) from exc

        html_text = raw_bytes.decode("utf-8", errors="ignore")

    html_text = _unescape_html_entities(html_text, context=canonical_url)
    text = _strip_html(html_text)

    text_form = _infer_form_type_from_text(text)

    form_type_candidates = [
        detected_form_type,
        xbrl_result.get("form_type") if xbrl_result else None,
        form_hint_query,
        form_type_hint,
        text_form,
    ]
    form_type: Optional[str] = None
    for candidate in form_type_candidates:
        normalized_candidate = _normalize_form_type(candidate)
        if normalized_candidate:
            form_type = normalized_candidate
            break

    _log_parse_event(
        logging.DEBUG,
        "form inference",
        url=canonical_url,
        form_hint=form_hint_query,
        url_form=form_type_hint,
        text_form=text_form,
        final_form=form_type,
    )

    defaults = _form_defaults(form_type)
    cashflow_headers = defaults["cashflow_headers"]
    cashflow_header_patterns = defaults.get("cashflow_header_patterns", [])

    exhibit_override: Optional[dict] = None
    if form_type == "6-K":
        lower_html = html_text.lower()
        header_present = any(header.lower() in lower_html for header in cashflow_headers)
        if not header_present:
            for pattern in cashflow_header_patterns:
                try:
                    if pattern.search(html_text):
                        header_present = True
                        break
                except Exception:
                    continue
        if not header_present:
            exhibit_parse = follow_exhibits_and_parse(canonical_url, html_text)
            if exhibit_parse.get("source") == "exhibit":
                exhibit_override = exhibit_parse
            else:
                note_parts = [
                    f"6-K missing operating cash flow statement headers: {canonical_url}"
                ]
                if xbrl_result:
                    for suffix in _extract_note_suffix(xbrl_result.get("note")):
                        cleaned_suffix = suffix.replace(canonical_url, "").strip()
                        if cleaned_suffix:
                            note_parts.append(cleaned_suffix)
                elif xbrl_error:
                    note_parts.append(xbrl_error)

                period_value = (
                    xbrl_result.get("period_months")
                    if (xbrl_result and xbrl_result.get("period_months") in {3, 6, 9, 12})
                    else defaults.get("period_months_default")
                )
                assumption_value = (xbrl_result.get("assumption") if xbrl_result else "") or ""
                units_scale_value = (xbrl_result.get("units_scale") if xbrl_result else None) or 1
                source_tags = ["XBRL"] if (xbrl_result and xbrl_result.get("cash_raw") is not None) else None

                result = _finalize_runway_result(
                    cash=xbrl_result.get("cash_raw") if xbrl_result else None,
                    ocf_raw=None,
                    ocf_quarterly=None,
                    period_months=period_value,
                    assumption=assumption_value,
                    note="; ".join(part for part in note_parts if part),
                    form_type=form_type,
                    units_scale=units_scale_value,
                    status="6-K missing OCF (exhibits tried)",
                    source_tags=source_tags,
                )

                _log_runway_outcome(
                    canonical_url,
                    form_type,
                    result,
                    xbrl_result=xbrl_result,
                    xbrl_error=xbrl_error,
                    extra={
                        "path": "6-K header+exhibit check",
                        "header_present": header_present,
                    },
                )

                return result

    html_info = _parse_html_cashflow_sections(html_text, context_url=canonical_url)
    html_cash = html_info.get("cash_value")
    html_ocf_raw_value = html_info.get("ocf_value")
    html_period_inferred = html_info.get("period_months_inferred")
    html_units_scale = html_info.get("units_scale") or 1
    html_evidence = html_info.get("evidence") or ""
    html_source = "main"
    exhibit_status: Optional[str] = None
    exhibit_href: Optional[str] = None
    exhibit_doc_type: Optional[str] = None

    if html_cash is None or html_ocf_raw_value is None:
        exhibit_parse = exhibit_override or follow_exhibits_and_parse(canonical_url, html_text)
        if exhibit_parse.get("source") == "exhibit":
            html_source = "exhibit"
            html_cash = exhibit_parse.get("cash_value")
            html_ocf_raw_value = exhibit_parse.get("ocf_value")
            html_period_inferred = exhibit_parse.get("period_months")
            html_units_scale = exhibit_parse.get("units_scale") or 1
            html_evidence = exhibit_parse.get("evidence") or ""
            exhibit_status = exhibit_parse.get("status")
            exhibit_href = exhibit_parse.get("exhibit_href")
            exhibit_doc_type = exhibit_parse.get("exhibit_doc_type")
            source_html_info = exhibit_parse.get("html_info")
            if isinstance(source_html_info, dict):
                html_info = dict(source_html_info)
            else:
                html_info = {
                    "found_cashflow_header": exhibit_parse.get("found_cashflow_header"),
                    "cash_value": html_cash,
                    "ocf_value": html_ocf_raw_value,
                    "period_months_inferred": html_period_inferred,
                    "units_scale": html_units_scale,
                    "evidence": html_evidence,
                    "html_header": exhibit_parse.get("found_cashflow_header"),
                }
            html_info["source"] = "exhibit"
            if exhibit_href:
                html_info["exhibit_href"] = exhibit_href

    period_from_xbrl = None
    if xbrl_result and xbrl_result.get("period_months") in {3, 6, 9, 12}:
        period_from_xbrl = xbrl_result.get("period_months")

    period_from_html = (
        html_period_inferred if html_period_inferred in {3, 6, 9, 12} else None
    )

    final_period = period_from_xbrl or period_from_html or defaults.get("period_months_default")

    period_for_html_normalization = (
        html_period_inferred
        if html_period_inferred in {3, 6, 9, 12}
        else final_period
    )
    if period_for_html_normalization is None and html_period_inferred is not None:
        period_for_html_normalization = html_period_inferred

    html_ocf_normalized = None
    html_assumption = ""
    if html_ocf_raw_value is not None:
        html_ocf_normalized, normalized_period, html_assumption = _normalize_ocf_value(
            html_ocf_raw_value, period_for_html_normalization
        )
        if final_period is None and normalized_period is not None:
            final_period = normalized_period
        html_assumption = html_assumption or ""

    xbrl_cash = xbrl_result.get("cash_raw") if xbrl_result else None
    xbrl_ocf_raw_value = xbrl_result.get("ocf_raw") if xbrl_result else None
    xbrl_ocf_quarterly = xbrl_result.get("ocf_quarterly_raw") if xbrl_result else None
    xbrl_assumption = (xbrl_result.get("assumption") if xbrl_result else "") or ""
    xbrl_units_scale = (xbrl_result.get("units_scale") if xbrl_result else None) or 1

    final_cash = None
    cash_source: Optional[str] = None
    if xbrl_cash is not None:
        final_cash = xbrl_cash
        cash_source = "XBRL"
    elif html_cash is not None:
        final_cash = html_cash
        cash_source = "HTML"

    final_ocf_raw = None
    final_ocf_quarterly = None
    final_assumption = ""
    ocf_source: Optional[str] = None
    if xbrl_ocf_quarterly is not None:
        final_ocf_raw = xbrl_ocf_raw_value
        final_ocf_quarterly = xbrl_ocf_quarterly
        final_assumption = xbrl_assumption
        ocf_source = "XBRL"
    elif html_ocf_normalized is not None:
        final_ocf_raw = html_ocf_raw_value
        final_ocf_quarterly = html_ocf_normalized
        final_assumption = html_assumption
        ocf_source = "HTML"
    else:
        final_assumption = html_assumption or xbrl_assumption or ""

    if ocf_source == "HTML" and html_period_inferred in {3, 6, 9, 12}:
        final_period = html_period_inferred

    used_html_for_period = False
    if (
        final_period in {3, 6, 9, 12}
        and html_period_inferred in {3, 6, 9, 12}
        and final_period == html_period_inferred
    ):
        used_html_for_period = True

    used_xbrl = False
    if cash_source == "XBRL" or ocf_source == "XBRL":
        used_xbrl = True
    if period_from_xbrl in {3, 6, 9, 12} and final_period == period_from_xbrl:
        used_xbrl = True

    used_html = False
    if cash_source == "HTML" or ocf_source == "HTML" or used_html_for_period:
        used_html = True

    unit_candidates: list[int] = []
    if cash_source == "HTML" or ocf_source == "HTML":
        unit_candidates.append(html_units_scale)
    if cash_source == "XBRL" or ocf_source == "XBRL":
        unit_candidates.append(xbrl_units_scale)

    final_units_scale = 1
    for value in unit_candidates:
        if value and value != 1:
            final_units_scale = value
            break
    if final_units_scale == 1 and unit_candidates:
        final_units_scale = unit_candidates[0] or 1

    source_tags: list[str] = []
    if used_xbrl:
        source_tags.append("XBRL")
    if used_html:
        source_tags.append("HTML")

    merged_sources = used_xbrl and used_html
    note_parts: list[str] = []
    if merged_sources:
        note_parts.append(f"values parsed from XBRL and HTML (merged): {canonical_url}")
    elif used_xbrl:
        note_parts.append(f"values parsed from XBRL: {canonical_url}")
    else:
        note_parts.append(f"values parsed from filing HTML: {canonical_url}")

    if used_html and html_evidence:
        note_parts.append(html_evidence)

    if html_source == "exhibit" and exhibit_href:
        if exhibit_doc_type:
            note_parts.append(f"exhibit source: {exhibit_doc_type} {exhibit_href}")
        else:
            note_parts.append(f"exhibit source: {exhibit_href}")

    if xbrl_result:
        for suffix in _extract_note_suffix(xbrl_result.get("note")):
            cleaned_suffix = suffix.replace(canonical_url, "").strip()
            if cleaned_suffix:
                note_parts.append(cleaned_suffix)
    elif xbrl_error:
        note_parts.append(xbrl_error)

    note_text = "; ".join(part for part in note_parts if part)

    result = _finalize_runway_result(
        cash=final_cash,
        ocf_raw=final_ocf_raw,
        ocf_quarterly=final_ocf_quarterly,
        period_months=final_period,
        assumption=final_assumption,
        note=note_text,
        form_type=form_type,
        units_scale=final_units_scale,
        status=exhibit_status,
        source_tags=source_tags or None,
    )

    if html_source == "exhibit" and exhibit_href:
        result["exhibit_href"] = exhibit_href
        if exhibit_doc_type:
            result["exhibit_doc_type"] = exhibit_doc_type

    _log_runway_outcome(
        canonical_url,
        form_type,
        result,
        html_info=html_info,
        xbrl_result=xbrl_result,
        xbrl_error=xbrl_error,
        extra={"path": "html_merge"},
    )

    _log_parse_event(
        logging.DEBUG,
        f"runway: DONE status={result.get('status')} ocf_raw={result.get('ocf_raw')} cash_raw={result.get('cash_raw')} period_months={result.get('period_months')} runway_q={result.get('runway_quarters_raw')}",
        url=canonical_url,
    )

    return result


__all__ = ["get_runway_from_filing", "url_matches_form"]
