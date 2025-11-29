import pandas as pd

from app.weekly_deep_research import _conviction_from_subscores
from app.weekly_validated import evaluate_validation


def test_validation_rules_and_conviction_logic():
    base = pd.Series(
        {
            "Dilution": "Low",
            "Runway (qtrs)": 6,
            "Catalyst": "Tier-1",
            "DilutionEvidencePrimary": "https://www.sec.gov/s3",
            "RunwayEvidencePrimary": "https://www.sec.gov/10q",
            "CatalystEvidencePrimary": "https://www.sec.gov/8k",
            "Subscores Evidenced (x/5)": 5,
            "Materiality (pass/fail + note)": "PASS - Tier1",
            "Biotech Peer Read-Through (Y/N + link)": "N",
        }
    )

    status, reason = evaluate_validation(base)
    assert status == "Validated"
    assert reason == ""

    biotech_missing = base.copy()
    biotech_missing["Biotech Peer Read-Through (Y/N + link)"] = "TBD"
    status, reason = evaluate_validation(biotech_missing)
    assert status == "TBD — exclude"
    assert "Biotech" in reason

    missing_subscore = base.copy()
    missing_subscore["DilutionEvidencePrimary"] = ""
    status, reason = evaluate_validation(missing_subscore)
    assert status == "TBD — exclude"
    assert reason.startswith("Mandatory subscore missing")

    assert _conviction_from_subscores(4, True, "PASS - Tier1") == "Medium"
    assert _conviction_from_subscores(5, True, "PASS - Tier1") == "High"
    assert _conviction_from_subscores(3, True, "PASS - Tier1") == "Low"
    assert _conviction_from_subscores(4, False, "PASS - Tier1") == "Low"
