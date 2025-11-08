"""HTML parsing helpers for runway extraction."""
from __future__ import annotations

import html
import logging
import re
from typing import Iterable, Optional

from .logging import log_parse_event
from .units import (
    _CURRENCY_CONTEXT_WORDS,
    _NUMERIC_TOKEN_PATTERN,
    _TOKEN_LOOKAHEAD,
    _NUMBER_SEARCH_PATTERN,
    find_scale_near,
    normalize_number,
)

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

_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<t[hd][^>]*>.*?</t[hd]>", re.IGNORECASE | re.DOTALL)


def preview_text(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    sanitized = text.replace("\r", " ").replace("\n", " ")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > limit:
        return sanitized[: limit - 3] + "..."
    return sanitized


def unescape_html_entities(html_text: str, *, context: Optional[str] = None) -> str:
    if not html_text:
        return html_text
    before = preview_text(html_text)
    unescaped = html.unescape(html_text)
    after = preview_text(unescaped)
    log_parse_event(
        logging.DEBUG,
        "html entity unescape",
        url=context,
        unescape_before=before,
        unescape_after=after,
    )
    return unescaped


def format_score_tuple(score: Optional[tuple[object, ...]]) -> str:
    if not score:
        return "()"
    return "(" + ", ".join(str(part) for part in score) + ")"


def normalize_for_match(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"-\s*/\s*-", "-", text)
    return re.sub(r"\s+", " ", text)


def extract_number_after_keyword(text: str, keywords: Iterable[str]) -> Optional[float]:
    if not text:
        return None
    norm_text = normalize_for_match(text)
    for keyword in keywords:
        k = normalize_for_match(keyword)
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
                value = float(normalize_number(token))
            except (TypeError, ValueError):
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

            block_id = 0
            if token.startswith("(") or token.endswith(")"):
                block_id = 1

            numeric_tokens.append(
                {
                    "idx": idx,
                    "token": token,
                    "value": value,
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
                    "block_id": block_id,
                }
            )

        strong_flags = [0] * len(numeric_tokens)
        for i, token_info in enumerate(numeric_tokens):
            digit_count = int(token_info["digit_count"])
            is_negative = bool(token_info["is_negative"])
            has_separator = bool(token_info["has_separator"])
            has_currency = bool(token_info["has_currency"])
            block_id = int(token_info.get("block_id", 0))
            if digit_count >= 3 or has_separator or has_currency or is_negative:
                strong_flags[i] = 1
            elif block_id == 0:
                strong_flags[i] = 0
            else:
                strong_flags[i] = 1

        strong_suffix = [0] * len(numeric_tokens)
        running = 0
        for i in range(len(numeric_tokens) - 1, -1, -1):
            strong_suffix[i] = running
            if strong_flags[i]:
                running += 1

        scored_tokens: list[dict[str, object]] = []
        for i, token_info in enumerate(numeric_tokens):
            idx = int(token_info["idx"])
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
                and re.match(r"^\d{1,3}([,\.\u00A0\u202F'’]\d{3})+", normalized_group)
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
            key=lambda info: (-abs(float(info["value"])), int(info["idx"]))
        )
        for entry in sorted_for_log[:5]:
            score_repr = format_score_tuple(entry.get("score"))
            log_parse_event(
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
            score_repr = format_score_tuple(best_candidate.get("score"))
            log_parse_event(
                logging.DEBUG,
                "caption_selected",
                keyword=keyword,
                candidate_token=str(best_candidate["token"]),
                value=float(best_candidate["value"]),
                score=score_repr,
            )
            return float(best_candidate["value"])
    return None


def strip_html(html_text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_text or "")
    return " ".join(without_tags.split())


def extract_html_section(
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


def infer_months_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    patterns = [
        (re.compile(r"three\s+months\s+ended", re.IGNORECASE), 3),
        (re.compile(r"six\s+months\s+ended", re.IGNORECASE), 6),
        (re.compile(r"nine\s+months\s+ended", re.IGNORECASE), 9),
        (re.compile(r"twelve\s+months\s+ended", re.IGNORECASE), 12),
        (re.compile(r"twelve\s+month\s+period", re.IGNORECASE), 12),
        (re.compile(r"year\s+ended", re.IGNORECASE), 12),
    ]
    for pattern, months in patterns:
        if pattern.search(text):
            return months
    return None


def parse_html_cashflow_sections(html_text: str, *, context_url: Optional[str] = None) -> dict:
    from .router import _form_defaults  # local import to avoid cycles

    log_parse_event(logging.DEBUG, "runway: parse_html start", url=context_url)

    text = strip_html(html_text)
    defaults = _form_defaults(None)

    cashflow_headers = list(defaults["cashflow_headers"])
    header_candidates = cashflow_headers + list(_CASHFLOW_HEADER_PATTERNS)

    balance_section_html = extract_html_section(html_text, _BALANCE_HEADERS, window=60000)
    cashflow_section_html = extract_html_section(html_text, header_candidates, window=120000)

    balance_section_text = strip_html(balance_section_html) if balance_section_html else None
    cashflow_section_text = strip_html(cashflow_section_html) if cashflow_section_html else None

    balance_scale = find_scale_near(balance_section_html, html_text) if balance_section_html else 1
    cashflow_scale = find_scale_near(cashflow_section_html, html_text) if cashflow_section_html else 1
    default_scale = find_scale_near(html_text, html_text)

    burn_keywords = defaults["ocf_keywords_burn"]
    provided_keywords = defaults["ocf_keywords_provided"]

    normalized_burn = [normalize_for_match(k).lower() for k in burn_keywords]
    normalized_provided = [normalize_for_match(k).lower() for k in provided_keywords]
    normalized_cash_flow_keywords = [normalize_for_match(k).lower() for k in _CASH_KEYWORDS_FLOW]

    def _rows_from_table(table_html: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_match in _ROW_PATTERN.finditer(table_html):
            row_html = row_match.group(0)
            cells = [strip_html(cell) for cell in _CELL_PATTERN.findall(row_html)]
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
        balance_match = extract_number_after_keyword(balance_section_text, _CASH_KEYWORDS_BALANCE)
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
                    normalized = normalize_number(token)
                except (TypeError, ValueError):
                    continue
                if normalized is not None:
                    values.append(float(normalized))
            return values

        for row in rows:
            normalized_row = normalize_for_match(" ".join(row)).lower()
            if not normalized_row:
                prev_cash_label = False
                continue

            has_cash_phrase = "cash and cash equivalents" in normalized_row
            has_restricted_phrase = "cash & restricted cash" in normalized_row or "cash and restricted cash" in normalized_row
            if "net decrease" in normalized_row or "net increase" in normalized_row:
                prev_cash_label = False
                continue
            if has_cash_phrase and ("end of" in normalized_row or "period" in normalized_row or "year" in normalized_row):
                numbers = _row_numbers(row)
                if numbers:
                    return numbers[-1]
                prev_cash_label = True
                continue
            if prev_cash_label and has_restricted_phrase:
                numbers = _row_numbers(row)
                if numbers:
                    return numbers[-1]
            prev_cash_label = has_cash_phrase
        return None

    tables_with_cash = []
    for table in tables:
        rows = table.get("rows") or []
        if not rows:
            continue
        cash_candidate = _match_cash_from_rows(rows)
        if cash_candidate is not None:
            tables_with_cash.append({"table": table, "cash": cash_candidate})

    if tables_with_cash and cash_value is None:
        best = tables_with_cash[0]
        cash_value = float(best["cash"]) * float(best["table"]["scale"])
        cash_scale_used = int(best["table"]["scale"])

    ocf_candidates: list[tuple[float, str]] = []

    def _match_ocf_from_rows(rows: list[list[str]]) -> list[tuple[float, str]]:
        matches: list[tuple[float, str]] = []
        for row in rows:
            normalized_row = normalize_for_match(" ".join(row)).lower()
            if not normalized_row:
                continue
            if any(keyword in normalized_row for keyword in normalized_burn):
                values = [normalize_number(token) for token in row[1:]]
                filtered = [v for v in values if v is not None]
                if filtered:
                    matches.append((float(filtered[-1]), "burn"))
            elif any(keyword in normalized_row for keyword in normalized_provided):
                values = [normalize_number(token) for token in row[1:]]
                filtered = [v for v in values if v is not None]
                if filtered:
                    matches.append((-float(filtered[-1]), "provided"))
        return matches

    for table in tables:
        rows = table.get("rows") or []
        if not rows:
            continue
        matches = _match_ocf_from_rows(rows)
        if matches:
            ocf_candidates.extend((value * float(table.get("scale", 1)), kind) for value, kind in matches)

    html_ocf_raw_value: Optional[float] = None
    if ocf_candidates:
        html_ocf_raw_value = float(ocf_candidates[-1][0])

    if html_ocf_raw_value is None and cashflow_section_text:
        text_match = extract_number_after_keyword(
            cashflow_section_text,
            burn_keywords + provided_keywords,
        )
        if text_match is not None:
            html_ocf_raw_value = float(text_match) * float(cashflow_scale or default_scale)

    period_months_inferred = infer_months_from_text(text)
    if period_months_inferred not in {3, 6, 9, 12}:
        period_months_inferred = infer_months_from_text(cashflow_section_text)

    html_units_scale = cashflow_scale or balance_scale or default_scale or 1

    cash_flag = cash_value is not None
    ocf_flag = html_ocf_raw_value is not None
    header_flag = bool(cashflow_section_html)

    evidence_parts = []
    if header_flag:
        evidence_parts.append("cash flow table located")
    if cash_flag:
        evidence_parts.append(f"cash={cash_value}")
    if ocf_flag:
        evidence_parts.append(f"ocf={html_ocf_raw_value}")
    if period_months_inferred:
        evidence_parts.append(f"period={period_months_inferred}")
    evidence_text = "; ".join(evidence_parts)

    log_parse_event(
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
        "evidence": evidence_text,
        "html_header": bool(cashflow_section_html),
    }


__all__ = [
    "unescape_html_entities",
    "parse_html_cashflow_sections",
    "extract_html_section",
    "strip_html",
    "extract_number_after_keyword",
    "infer_months_from_text",
    "normalize_for_match",
    "format_score_tuple",
    "preview_text",
]
