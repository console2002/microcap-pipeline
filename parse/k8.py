"""Parser and classifier for 8-K filings."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urllib_request
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from .htmlutil import strip_html, unescape_html_entities
from .logging import log_parse_event
from .router import _fetch_url, _user_agent


_ITEM_PATTERN = re.compile(r"item[\s\u00A0]*([0-9]{1,2}\.[0-9]{2})", re.IGNORECASE)
_TARGET_ITEMS = {"1.01", "2.02", "3.02", "5.02", "7.01", "8.01", "9.01"}

_CONTRACT_KEYWORDS = [
    "obligated",
    "obligation",
    "guaranteed minimum",
    "min. guarantee",
    "minimum guarantee",
    "base value",
    "ceiling",
    "task order",
    "award id",
    "notice to proceed",
]

_GUIDANCE_UP_KEYWORDS = [
    "raises guidance",
    "guidance raised",
    "guidance increased",
    "outlook raised",
    "outlook increased",
    "above prior guidance",
    "revenue guidance up",
    "eps guidance up",
    "ebitda guidance up",
    "revenue guidance increased",
    "eps guidance increased",
    "ebitda guidance increased",
]

_REGULATORY_KEYWORDS = [
    "fda",
    "510(k)",
    "de novo",
    "pdufa",
    "ind accepted",
    "ind clearance",
    "ema",
    "ce mark",
    "advisory committee",
    "clearance",
    "approval",
]

_NON_BINDING_KEYWORDS = [
    "non-binding",
    "letter of intent",
    "loi",
]

_NON_BINDING_WEAK_WORDS = ["may", "intend", "proposed", "propose"]

_DILUTION_KEYWORDS = {
    "at-the-market": "ATM",
    "atm": "ATM",
    "sales agreement": "ATM",
    "equity distribution": "ATM",
    "underwritten": "UNDERWRITTEN",
    "registered direct": "RD",
    "shelf": "SHELF",
    "rule 415": "SHELF",
    "warrant": "WARRANTS",
    "convertible": "CONVERT",
    "pipe": "PIPE",
    "senior notes": "NOTES",
    "note offering": "NOTES",
    "notes offering": "NOTES",
    "424b": "SHELF",
    "s-3": "SHELF",
    "s-4": "UNDERWRITTEN",
    "s-1": "UNDERWRITTEN",
    "pre-funded": "WARRANTS",
}

_CURRENCY_PATTERN = re.compile(r"\$?\s?\d[\d,]*\.?\d*\s*(million|thousand|m|k)?", re.IGNORECASE)


@dataclass
class _ItemSection:
    code: str
    text: str


def _canonicalize_sec_url(url: str) -> str:
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    lower_url = url.lower()
    if lower_url.startswith("https://www.sec.gov/archives/"):
        return url
    netloc_lower = (parsed.netloc or "").lower()
    path_lower = (parsed.path or "").lower()
    viewer_paths = ("/ix", "/ixviewer", "/cgi-bin/viewer", "/viewer")
    if netloc_lower.endswith("sec.gov") and any(
        path_lower.startswith(prefix) for prefix in viewer_paths
    ):
        query = parse_qs(parsed.query or "")
        param_value = None
        for key in ("doc", "filename"):
            values = query.get(key)
            if values:
                param_value = values[0]
                break
        if not param_value:
            return url
        p = unquote(param_value)
        if p.startswith("/Archives/"):
            canonical = f"https://www.sec.gov{p}"
        elif p.startswith("Archives/"):
            canonical = f"https://www.sec.gov/{p}"
        elif p.startswith("http"):
            canonical = p
        else:
            return url
        try:
            parsed_target = urlparse(canonical)
        except Exception:
            return canonical
        scheme = "https"
        netloc = parsed_target.netloc or "www.sec.gov"
        if netloc.lower().endswith("sec.gov"):
            netloc = "www.sec.gov"
        path_only = parsed_target.path or ""
        return f"{scheme}://{netloc}{path_only}"
    return url


def _accession_directory(path: str) -> Optional[str]:
    if not path:
        return None
    marker = "/archives/edgar/data/"
    lower_path = path.lower()
    idx = lower_path.find(marker)
    if idx == -1:
        return None
    remainder = path[idx + len(marker) :]
    parts = remainder.split("/")
    if len(parts) < 2:
        return None
    cik = parts[0].strip()
    accession = parts[1].strip()
    if not cik or not accession:
        return None
    prefix = path[: idx + len(marker)]
    return f"{prefix}{cik}/{accession}/"


def _read_local_file(parsed) -> Optional[str]:
    try:
        path_text = parsed.path or ""
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        candidate = Path(path_text)
    except Exception:
        return None
    if not candidate.exists():
        return None
    try:
        return candidate.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _fetch_html(url: str, html: Optional[str]) -> tuple[Optional[str], bool]:
    url_plain = bool(url and url.lower().endswith(".txt"))
    if html:
        return html, url_plain
    if not url:
        return None, False
    parsed = urlparse(url)
    if parsed.scheme == "file":
        content = _read_local_file(parsed)
        return content, url_plain or bool((parsed.path or "").lower().endswith(".txt"))
    try:
        req = urllib_request.Request(url, headers={"User-Agent": _user_agent()})
        with urllib_request.urlopen(req) as response:  # pragma: no cover - network
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except Exception as exc:  # pragma: no cover - network errors
        log_parse_event(logging.DEBUG, "8k fetch failed", url=url, error=str(exc))
        return None, False
    text = raw.decode("utf-8", errors="ignore")
    is_plain = "text/plain" in (content_type or "").lower()
    return text, is_plain or url_plain


def _html_to_text(html_text: str) -> str:
    if not html_text:
        return ""
    replaced = re.sub(
        r"(?is)<\s*(br|p|div|li|tr|td|table|h[1-6])[^>]*>",
        "\n",
        html_text,
    )
    no_tags = re.sub(r"<[^>]+>", " ", replaced)
    return re.sub(r"\s+", " ", no_tags.replace("\xa0", " ")).strip()


def _split_item_sections(text: str) -> list[_ItemSection]:
    sections: list[_ItemSection] = []
    if not text:
        return sections
    matches = list(_ITEM_PATTERN.finditer(text))
    for idx, match in enumerate(matches):
        code = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        snippet = text[start:end].strip()
        sections.append(_ItemSection(code=code, text=snippet))
    return sections


def _gather_target_items(sections: Iterable[_ItemSection]) -> list[_ItemSection]:
    collected: list[_ItemSection] = []
    for section in sections:
        if section.code in _TARGET_ITEMS:
            collected.append(section)
    return collected


def _same_filing_link(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None
    canonical_base = _canonicalize_sec_url(base_url)
    resolved = href
    parsed_candidate = urlparse(href)
    if not parsed_candidate.scheme:
        resolved = urljoin(canonical_base, href)
    resolved = _canonicalize_sec_url(resolved)
    parsed_base = urlparse(canonical_base)
    parsed_resolved = urlparse(resolved)
    base_dir = _accession_directory(parsed_base.path or "")
    if base_dir:
        prefix_lower = base_dir.lower()
        resolved_path = parsed_resolved.path or ""
        if parsed_resolved.netloc.lower() != "www.sec.gov":
            return None
        if resolved_path.lower().startswith(prefix_lower):
            return resolved
        return None
    # Fallback for local fixtures (file:// paths)
    if parsed_base.scheme == parsed_resolved.scheme == "file":
        base_path = parsed_base.path or ""
        resolved_path = parsed_resolved.path or ""
        if not base_path or not resolved_path:
            return None
        base_dir = base_path.rsplit("/", 1)[0] if "/" in base_path else base_path
        resolved_dir = resolved_path.rsplit("/", 1)[0] if "/" in resolved_path else resolved_path
        if base_dir.lower() != resolved_dir.lower():
            return None
        return resolved
    return None


def _extract_exhibits(html_text: str, base_url: str) -> list[dict[str, str]]:
    exhibits: list[dict[str, str]] = []
    if not html_text:
        return exhibits
    canonical_base = _canonicalize_sec_url(base_url)
    link_pattern = re.compile(r"<a[^>]+href\s*=\s*['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    for match in link_pattern.finditer(html_text):
        href = match.group(1)
        text_html = match.group(2) or ""
        normalized_text = strip_html(text_html).lower()
        normalized_href = href.lower()
        keywords = [
            "ex-99",
            "exhibit 99",
            "press release",
            "press-release",
            "pressrelease",
            "presentation",
            "deck",
            "slides",
        ]
        if not any(keyword in normalized_href or keyword in normalized_text for keyword in keywords):
            continue
        resolved = _same_filing_link(canonical_base, href)
        if not resolved:
            continue
        exhibits.append({"url": resolved, "name": Path(resolved).name or resolved})
    return exhibits


def _fetch_exhibit_text(exhibit_url: str) -> Optional[str]:
    parsed = urlparse(exhibit_url)
    if parsed.scheme == "file":
        content = _read_local_file(parsed)
    else:
        try:
            raw = _fetch_url(exhibit_url)
        except Exception as exc:  # pragma: no cover - network errors
            log_parse_event(
                logging.DEBUG,
                "8k exhibit fetch failed",
                url=exhibit_url,
                error=str(exc),
            )
            return None
        content = raw.decode("utf-8", errors="ignore")
    if not content:
        return None
    unescaped = unescape_html_entities(content, context=exhibit_url)
    return _html_to_text(unescaped)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _extract_currency_trigger(text: str) -> Optional[str]:
    if not text:
        return None
    preferred: Optional[str] = None
    fallback: Optional[str] = None
    for match in _CURRENCY_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        lower = candidate.lower()
        fallback = candidate
        if "$" in candidate or any(unit in lower for unit in ["million", "thousand", "m", "k"]):
            preferred = candidate
            break
    return preferred or fallback


def _classify(items: list[_ItemSection], combined_text: str) -> dict:
    lower_map = {section.code: section.text.lower() for section in items}
    lower_all = combined_text.lower()

    is_dilution = False
    dilution_tags: list[str] = []
    for keyword, tag in _DILUTION_KEYWORDS.items():
        if keyword in lower_all:
            if tag not in dilution_tags:
                dilution_tags.append(tag)
            if tag in {"ATM", "UNDERWRITTEN", "RD", "SHELF", "PIPE", "NOTES", "WARRANTS", "CONVERT"}:
                is_dilution = True

    classification = {
        "is_catalyst": False,
        "tier": None,
        "tier1_type": None,
        "tier1_trigger": None,
        "is_dilution": is_dilution,
        "dilution_tags": dilution_tags,
        "ignore_reason": None,
    }

    if not items:
        return classification

    if _contains_any(lower_all, ["reg fd"]) and {section.code for section in items} == {"7.01"}:
        classification["ignore_reason"] = "Reg FD only"
        return classification

    if any(keyword in lower_all for keyword in _NON_BINDING_KEYWORDS):
        if any(word in lower_all for word in _NON_BINDING_WEAK_WORDS):
            classification["ignore_reason"] = "Non-binding LOI"

    def _promote(tier: str, tier1_type: Optional[str], trigger: Optional[str] = None) -> None:
        if classification["ignore_reason"]:
            return
        classification["is_catalyst"] = True
        classification["tier"] = tier
        classification["tier1_type"] = tier1_type
        classification["tier1_trigger"] = trigger

    item_101 = lower_map.get("1.01", "")
    item_801 = lower_map.get("8.01", "")
    item_202 = lower_map.get("2.02", "")

    def _section_text(code: str) -> str:
        for section in items:
            if section.code == code:
                return section.text
        return ""

    if not classification["is_dilution"]:
        if item_101 and _contains_any(item_101, _CONTRACT_KEYWORDS):
            trigger = _extract_currency_trigger(_section_text("1.01"))
            _promote("Tier-1", "FUNDED_AWARD", trigger)
        elif item_801 and _contains_any(item_801, _CONTRACT_KEYWORDS):
            trigger = _extract_currency_trigger(_section_text("8.01"))
            _promote("Tier-1", "FUNDED_AWARD", trigger)
        elif item_202 and _contains_any(item_202, _GUIDANCE_UP_KEYWORDS):
            _promote("Tier-1", "GUIDANCE_UP")
        elif item_801 and _contains_any(item_801, _REGULATORY_KEYWORDS):
            _promote("Tier-1", "REGULATORY")
        elif any(section.code == "2.02" for section in items):
            _promote("Tier-2", None)
        elif any(section.code in {"1.01", "8.01", "5.02"} for section in items):
            _promote("Tier-2", None)

    if classification["is_dilution"]:
        classification["is_catalyst"] = False
        classification["tier"] = None
        classification["tier1_type"] = None
        classification["tier1_trigger"] = None

    if classification["tier"] != "Tier-1":
        classification["tier1_type"] = None
        classification["tier1_trigger"] = None

    if classification["ignore_reason"]:
        classification["is_catalyst"] = False
        classification["tier"] = None
        classification["tier1_type"] = None
        classification["tier1_trigger"] = None

    return classification


def _best_effort_filing_date(text: str) -> Optional[str]:
    patterns = [
        re.compile(r"Filing Date[:\s\u00A0]+([0-9]{4}-[0-9]{2}-[0-9]{2})", re.IGNORECASE),
        re.compile(r"Filed on[:\s\u00A0]+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", re.IGNORECASE),
        re.compile(r"Date of Report[^\d]*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1)
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                dt = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return dt.strftime("%Y-%m-%d")
    return None


def parse(url: str, html: str | None = None, form_hint: str | None = None) -> dict:
    form_type = form_hint or "8-K"
    canonical_url = _canonicalize_sec_url(url)
    html_text, is_plain = _fetch_html(canonical_url, html)
    if not html_text:
        log_parse_event(logging.DEBUG, "8k missing html", url=url)
        return {
            "complete": False,
            "url": url,
            "form_type": form_type,
            "filing_date": None,
            "items": [],
            "exhibits": [],
            "classification": {
                "is_catalyst": False,
                "tier": None,
                "tier1_type": None,
                "tier1_trigger": None,
                "is_dilution": False,
                "dilution_tags": [],
                "ignore_reason": "html_unavailable",
            },
        }

    if is_plain:
        plain_text = unescape_html_entities(html_text.replace("\u00A0", " "), context=canonical_url)
        text = plain_text
    else:
        unescaped = unescape_html_entities(html_text, context=canonical_url)
        text = _html_to_text(unescaped)
    sections = _split_item_sections(text)
    target_items = _gather_target_items(sections)

    exhibits_info = []
    appended_text_segments: list[str] = []
    if not is_plain:
        for exhibit in _extract_exhibits(unescaped, canonical_url):
            exhibit_text = _fetch_exhibit_text(exhibit["url"])
            if not exhibit_text:
                continue
            exhibit_entry = {
                "name": exhibit["name"],
                "text": exhibit_text,
            }
            exhibits_info.append(exhibit_entry)
            appended_text_segments.append(f"[Exhibit {exhibit_entry['name']}] {exhibit_text}")

    combined_text_parts = [section.text for section in target_items] + appended_text_segments
    combined_text = " \n".join(part for part in combined_text_parts if part)

    classification = _classify(target_items, combined_text)

    filing_date = _best_effort_filing_date(text)

    items_payload = [
        {"item": section.code, "text": section.text}
        for section in target_items
    ]

    log_parse_event(
        logging.DEBUG,
        "8k parsed",
        url=url,
        form=form_type,
        filing_date=filing_date,
        items=";".join(section.code for section in target_items),
        catalyst=classification["tier"],
        dilution=";".join(classification["dilution_tags"]),
        ignore=classification["ignore_reason"],
    )

    return {
        "complete": bool(target_items),
        "url": url,
        "form_type": form_type,
        "filing_date": filing_date,
        "items": items_payload,
        "exhibits": exhibits_info,
        "classification": classification,
    }


__all__ = ["parse"]
