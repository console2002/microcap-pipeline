from app.config import load_config, weekly_allowed_forms


def test_weekly_allowed_forms_roles_include_required_and_exclude_ownership():
    cfg = load_config()
    forms = weekly_allowed_forms(cfg)

    core_forms = {
        "10-Q",
        "10-Q/A",
        "10-QT",
        "10-QT/A",
        "10-K",
        "10-K/A",
        "10-KT",
        "10-KT/A",
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
        "6-K",
        "6-K/A",
    }
    assert core_forms.issubset(forms)

    dilution_forms = {
        "S-3",
        "S-3/A",
        "S-8",
        "S-8/A",
        "S-4",
        "S-4/A",
        "424B1",
        "424B3",
        "424B4",
        "424B5",
        "424B7",
        "424B8",
    }
    assert dilution_forms.issubset(forms)

    for ownership_form in [
        "13D",
        "13D/A",
        "13G",
        "13G/A",
        "13F-HR",
        "13F-HR/A",
        "8-A12B",
    ]:
        assert ownership_form not in forms
