from app.config import load_config, weekly_allowed_forms


def test_weekly_allowed_forms_uses_by_role():
    cfg = {
        "FilingsWhitelistByRole": {
            "RunwayCore": ["10-Q"],
            "Catalyst": ["8-K"],
            "Dilution": ["S-3"],
        },
    }

    assert weekly_allowed_forms(cfg) == {"10-Q", "8-K", "S-3"}


def test_weekly_allowed_forms_fallback_to_flat():
    cfg = {"FilingsWhitelist": ["8-K", "10-Q", "8-A12B"]}
    assert weekly_allowed_forms(cfg) == {"8-K", "10-Q", "8-A12B"}


def test_weekly_allowed_forms_with_loaded_config():
    # Ensure load_config integration still works with the helper
    cfg = load_config()
    forms = weekly_allowed_forms(cfg)
    assert forms, "expected allowed forms from config"
