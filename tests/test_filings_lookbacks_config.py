from app.config import filings_form_lookbacks, filings_max_lookback, load_config


def test_filings_lookbacks_cover_whitelist():
    cfg = load_config()
    lookbacks = cfg.get("FilingsLookbacks") or {}

    missing: list[str] = []
    for forms in cfg.get("FilingsWhitelistByRole", {}).values():
        for form in forms:
            key = str(form).strip().upper()
            if key and key not in lookbacks:
                missing.append(key)

    assert not missing, f"Missing lookback_days for forms: {missing}"


def test_filings_lookbacks_values_match_expected():
    cfg = load_config()
    mapping = filings_form_lookbacks(cfg)

    expected = {
        "10-Q": 180,
        "6-K": 180,
        "10-K": 400,
        "8-K": 400,
        "8-K/A": 180,
        "S-3": 400,
        "6-K/A": 400,
        "3": 400,
        "13D": 365,
    }

    for form, days in expected.items():
        assert mapping.get(form) == days


def test_filings_max_lookback_matches_mapping():
    cfg = load_config()
    mapping = filings_form_lookbacks(cfg)

    assert filings_max_lookback(cfg) == max(mapping.values())
