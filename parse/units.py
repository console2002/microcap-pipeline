"""Numeric normalization helpers for runway parsing."""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Iterable, Optional

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

_SCALE_MULTIPLIER = {
    "thousands": 1_000,
    "thousand": 1_000,
    "millions": 1_000_000,
    "million": 1_000_000,
}


def _strip_html_local(html_text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_text or "")
    return " ".join(without_tags.split())


def round_half_up(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    try:
        quant = Decimal("1").scaleb(-digits)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return float(value)
    return float(rounded)


def detect_scale_multiplier(text: Optional[str]) -> Optional[int]:
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
            snippet = _strip_html_local(fragment)[:200]
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


def normalize_ocf_value(
    value: Optional[float],
    period_months: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[str]]:
    if value is None:
        return None, period_months, None

    ocf_value = float(value)

    if period_months in {3, 6, 9, 12}:
        factor = float(period_months) / 3.0
        assumption_map = {3: "", 6: "6m/2", 9: "9m/3", 12: "annual/4"}
        normalized_value = ocf_value / factor
        return normalized_value, period_months, assumption_map.get(period_months, "")

    return ocf_value, period_months, None


__all__ = [
    "round_half_up",
    "normalize_ocf_value",
    "detect_scale_multiplier",
    "normalize_number",
    "find_scale_near",
    "_NUMERIC_TOKEN_PATTERN",
    "_NUMBER_SEARCH_PATTERN",
    "_TOKEN_LOOKAHEAD",
    "_CURRENCY_CONTEXT_WORDS",
]
