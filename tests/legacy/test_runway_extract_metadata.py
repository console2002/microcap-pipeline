from datetime import datetime, timezone

from runway_extract import _apply_source_metadata_defaults


def test_apply_source_metadata_defaults_uses_fallback_info() -> None:
    filed_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
    candidate_infos = [
        {
            "form": "6-K",
            "original_form": "6-K",
            "filed_at": filed_at,
            "filing_url": "https://example.com/filing.htm",
        }
    ]

    source_form, source_date, source_url, source_days_ago = _apply_source_metadata_defaults(
        source_form="",
        source_date="",
        source_url="",
        source_days_ago="",
        info_to_use=None,
        candidate_infos=candidate_infos,
        result_to_use=None,
    )

    assert source_form == "6-K"
    assert source_date == filed_at.date().isoformat()
    assert source_url == "https://example.com/filing.htm"
    assert source_days_ago == str((datetime.utcnow().date() - filed_at.date()).days)
