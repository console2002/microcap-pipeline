from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Sequence


def _normalize_patterns(patterns: Iterable[str]) -> list[str]:
    return [str(p or "").upper() for p in patterns if str(p or "").strip()]


def load_drop_filters(cfg: dict) -> tuple[list[str], list[str]]:
    """Return substring and whole-word drop patterns from configuration."""

    universe_cfg = cfg.get("Universe", {}) if isinstance(cfg, dict) else {}
    substrings = _normalize_patterns(universe_cfg.get("DropPatterns", []))
    word_patterns = _normalize_patterns(universe_cfg.get("DropWordPatterns", []))
    return substrings, word_patterns


@lru_cache(maxsize=None)
def _word_pattern_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern)
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)


def should_drop_record(
    company: str,
    ticker: str,
    substring_patterns: Sequence[str],
    word_patterns: Sequence[str],
) -> bool:
    """Determine whether the record should be excluded based on configured patterns."""

    haystack = f"{company or ''} {ticker or ''}".upper().strip()

    if haystack and substring_patterns and any(p in haystack for p in substring_patterns):
        return True

    if word_patterns and haystack:
        for pattern in word_patterns:
            if _word_pattern_regex(pattern).search(haystack):
                return True

    return False
