"""EX-99 exhibit traversal helpers."""
from __future__ import annotations

import html
import logging
import re
from typing import Iterable, Optional
from urllib.parse import unquote, urljoin, urlparse

from .htmlutil import (
    infer_months_from_text,
    parse_html_cashflow_sections,
    strip_html,
    unescape_html_entities,
)
from .logging import log_parse_event
from .router import (
    _form_defaults,  # type: ignore[attr-defined]
    _infer_form_type_from_text,  # type: ignore[attr-defined]
    _infer_form_type_from_url,  # type: ignore[attr-defined]
    _normalize_form_type,  # type: ignore[attr-defined]
)
_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<t[hd][^>]*>.*?</t[hd]>", re.IGNORECASE | re.DOTALL)

# ---- EX-99 filename hint (tie-breaker only) ----
# Numbers here are only a HINT. Do NOT rely on them exclusively.
_EXHIBIT_PRIORITY_NUMBERS: Iterable[str] = ("1", "2", "3", "5")

_EX99_PATTERNS = [
    re.compile(fr'(?i)\bex(?:hibit)?[-_. ]?99[-_. ]?{n}[a-z]?(\.\d+)?\b')
    for n in _EXHIBIT_PRIORITY_NUMBERS
]


def filename_has_priority_hint(name: str) -> bool:
    """Return True if filename suggests EX-99.[1/2/3/5] style. Case-insensitive."""
    if not name:
        return False
    return any(rx.search(name) for rx in _EX99_PATTERNS)


def exhibit_filename_hint_score(name: str) -> int:
    """
    Small tie-breaker score for EX-99 filenames that look like financial exhibits.
    Keep tiny (+8) so content-based scoring always dominates.
    """
    return 8 if filename_has_priority_hint(name) else 0


# ---- end hint block ----


def looks_like_exhibit_path(path: str) -> bool:
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
        texts = [strip_html(cell_html) for cell_html in cells]
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

    score = 0

    if doc_type.startswith("EX-99"):
        score += 50

    score += exhibit_filename_hint_score(filename)

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


def follow_exhibits_and_parse(filing_url: str, html_text: Optional[str]) -> dict:
    try:
        parsed = urlparse(filing_url)
    except Exception:
        parsed = None

    path = parsed.path if parsed else ""
    if looks_like_exhibit_path(path or ""):
        return {}

    cleaned_url = (filing_url or "").split("#", 1)[0]
    if "/" not in cleaned_url:
        return {}
    base_dir = cleaned_url.rsplit("/", 1)[0] + "/"

    form_candidates = []
    if html_text:
        try:
            form_candidates.append(_infer_form_type_from_text(strip_html(html_text)))
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
            if parsed and parsed.scheme == "file":
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
        index_html = unescape_html_entities(index_html, context=index_url_used)
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
    log_parse_event(
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
            log_parse_event(
                logging.DEBUG,
                f"runway: exhibit fetch failed {doc_label}",
                url=exhibit_url,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            continue

        exhibit_html = raw_html.decode("utf-8", errors="ignore")
        exhibit_html = unescape_html_entities(exhibit_html, context=exhibit_url)
        parsed_html = parse_html_cashflow_sections(exhibit_html, context_url=exhibit_url)

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
                inferred_from_text = infer_months_from_text(strip_html(exhibit_html))
                if inferred_from_text in {3, 6, 9, 12}:
                    period_inferred = inferred_from_text

        evidence_parts = [part for part in (evidence,) if part]
        if candidate.get("doc_type"):
            evidence_parts.append(f"exhibit={candidate.get('doc_type')}")
        evidence_full = "; ".join(evidence_parts)

        label_lower = doc_label.lower() if doc_label else exhibit_url

        if cash_val is not None and ocf_val is not None and period_inferred in {3, 6, 9, 12}:
            log_parse_event(
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
            log_parse_event(
                logging.DEBUG,
                f"runway: exhibit {label_lower} had no cash-flow table",
                url=exhibit_url,
                cash_found=cash_val is not None,
                ocf_found=ocf_val is not None,
            )
        else:
            log_parse_event(
                logging.DEBUG,
                f"runway: exhibit {label_lower} missing period",
                url=exhibit_url,
                period=period_inferred,
            )

    return {}


from .router import _fetch_url  # noqa: E402  # late import to avoid cycles

__all__ = ["follow_exhibits_and_parse", "looks_like_exhibit_path"]
