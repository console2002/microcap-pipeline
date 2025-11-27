from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from edgar import Filing
from edgar.company_reports import EightK

from app.edgar_adapter import EdgarAdapter, _parse_accession_from_url

logger = logging.getLogger(__name__)


def _normalize_item_label(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^Item\\s+", "", text, flags=re.IGNORECASE)
    return text


def _join_list(values: Iterable[str]) -> str:
    cleaned = [str(value).strip() for value in values if value and str(value).strip()]
    return "; ".join(cleaned)


@dataclass
class EdgarEightKParseResult:
    filing: Filing
    eight_k: EightK
    items_raw: list[str]
    items_normalized: list[str]
    exhibits: list[dict[str, str]]
    press_releases: list[dict[str, str]]
    press_release_text: str
    event_text: str
    filing_url_txt: str
    primary_ex99_docs: str
    primary_ex10_docs: str
    has_press_release: bool
    has_exhibits: bool
    has_xbrl: bool
    raw_submission_text: str


class EdgarEightKParser:
    """Parser that uses edgartools to extract 8-K structure and metadata."""

    def __init__(self, adapter: Optional[EdgarAdapter] = None):
        self.adapter = adapter or EdgarAdapter()

    def _build_text_url(self, cik: str, accession: str) -> str:
        accession_digits = re.sub(r"\D", "", accession)
        accession_nodash = accession_digits.zfill(18)
        accession_formatted = f"{accession_nodash[:10]}-{accession_nodash[10:12]}-{accession_nodash[12:]}"
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_nodash}/{accession_formatted}.txt"
        )

    def _fetch_filing(self, url: str) -> tuple[Optional[Filing], str, str]:
        cik, accession = _parse_accession_from_url(url)
        if not accession:
            return None, "", "missing_accession"

        text_url = self._build_text_url(cik, accession)
        raw_text = ""

        try:
            raw_text = self.adapter.download_filing_text(text_url)
            filing = Filing.from_sgml_text(raw_text)
            return filing, text_url, ""
        except Exception as exc:  # pragma: no cover - defensive network handling
            logger.warning("edgar 8-K text fetch failed %s: %s", text_url, exc)

        try:
            # Fallback resolution may still perform network requests; honor SEC
            # throttling before attempting.
            if hasattr(self.adapter, "_rate_limit"):
                self.adapter._rate_limit()
            filing = self.adapter._resolve_filing(accession)  # type: ignore[attr-defined]
            if filing is None:
                return None, text_url, "fetch_failed"
            return filing, text_url, ""
        except Exception as exc:  # pragma: no cover - defensive network handling
            logger.error("edgar 8-K filing resolve failed %s: %s", accession, exc)
            return None, text_url, "fetch_failed"

    def parse(self, url: str, form_hint: str = "") -> tuple[Optional[EdgarEightKParseResult], str]:
        filing, text_url, error = self._fetch_filing(url)
        if filing is None:
            return None, error or "filing_missing"

        form_clean = (filing.form or "").strip().upper()
        if form_clean not in {"8-K", "8-K/A"}:
            return None, "unsupported_form"

        try:
            eight_k: EightK = filing.obj()
        except Exception as exc:  # pragma: no cover - unexpected parsing failures
            logger.warning("edgar 8-K object parse failed for %s: %s", url, exc)
            return None, "parse_exception"

        items_raw = list(eight_k.items or [])
        items_normalized = [_normalize_item_label(item) for item in items_raw]

        exhibits_data: list[dict[str, str]] = []
        primary_ex99_docs: list[str] = []
        primary_ex10_docs: list[str] = []

        for exhibit in filing.exhibits:
            doc_type = getattr(exhibit, "document_type", "") or ""
            entry = {
                "exhibit_number": doc_type or getattr(exhibit, "sequence_number", ""),
                "document": getattr(exhibit, "document", ""),
                "description": getattr(exhibit, "description", ""),
                "url": getattr(exhibit, "url", ""),
                "document_type": doc_type,
            }
            exhibits_data.append(entry)

            doc_name = entry.get("document", "")
            if doc_type.upper().startswith("EX-99") and doc_name:
                primary_ex99_docs.append(doc_name)
            if doc_type.upper().startswith(("EX-10", "EX-1", "EX-2")) and doc_name:
                primary_ex10_docs.append(doc_name)

        press_releases: list[dict[str, str]] = []
        press_texts: list[str] = []
        for pr in getattr(eight_k, "press_releases", []) or []:
            text = ""
            try:
                text = pr.text() or ""
            except Exception:
                text = ""
            entry = {
                "document": getattr(pr, "document", ""),
                "description": getattr(pr, "description", ""),
                "url": pr.url() if hasattr(pr, "url") else "",
                "text": text,
            }
            press_releases.append(entry)
            if text.strip():
                press_texts.append(text.strip())

        event_parts: list[str] = []
        for label, normalized in zip(items_raw, items_normalized):
            if normalized.startswith("9.01"):
                continue
            try:
                text = eight_k[label] or ""
            except Exception:
                text = ""
            if text and text.strip():
                event_parts.append(text.strip())

        event_parts.extend(press_texts)
        event_text = "\n\n".join(part for part in event_parts if part)

        press_release_text = "\n\n".join(press_texts)

        result = EdgarEightKParseResult(
            filing=filing,
            eight_k=eight_k,
            items_raw=items_raw,
            items_normalized=items_normalized,
            exhibits=exhibits_data,
            press_releases=press_releases,
            press_release_text=press_release_text,
            event_text=event_text,
            filing_url_txt=getattr(filing, "filing_url", "") or getattr(filing, "text_url", text_url),
            primary_ex99_docs=_join_list(primary_ex99_docs),
            primary_ex10_docs=_join_list(primary_ex10_docs),
            has_press_release=bool(press_releases),
            has_exhibits=bool(exhibits_data),
            has_xbrl=bool(filing.xbrl()),
            raw_submission_text=getattr(filing, "full_text_submission", "") or "",
        )

        return result, ""

