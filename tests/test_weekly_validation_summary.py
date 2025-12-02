import pandas as pd

import pandas as pd

import app.weekly_validated as weekly_validated
from app.weekly_validated import evaluate_validation, summarize_validation_gates


def _base_row():
    return {
        "Price": 10.0,
        "ADV20": 50_000,
        "MarketCap": 100_000_000,
        "RunwayQuarters": 4,
        "Runway (qtrs)": 4,
        "RunwayEvidencePrimary": "http://runway",
        "Dilution": "Low",
        "DilutionScore": "",
        "DilutionEvidencePrimary": "http://dilution",
        "Catalyst": "Strong",
        "CatalystScore": "",
        "CatalystEvidencePrimary": "http://catalyst",
        "Subscores Evidenced (x/5)": 4,
        "SubscoresEvidencedCount": "",
        "Materiality": "Pass - note",
        "Materiality (pass/fail + note)": "Pass - note",
        "BiotechPeerRead": "N",
        "Biotech Peer Read-Through (Y/N + link)": "",
    }


def test_summarize_validation_gates_counts(monkeypatch):
    monkeypatch.setattr(weekly_validated, "BIOTECH_PEER_REQUIRED_FOR_VALIDATION", True)
    rows = []

    # All gates pass
    rows.append(_base_row())

    # Universe gate fail
    universe_fail = _base_row()
    universe_fail["Price"] = 0.5
    rows.append(universe_fail)

    # Mandatory subscores fail (remove dilution evidence and value)
    mandatory_fail = _base_row()
    mandatory_fail["Dilution"] = ""
    mandatory_fail["DilutionScore"] = ""
    mandatory_fail["DilutionEvidencePrimary"] = ""
    rows.append(mandatory_fail)

    # Runway numeric fail (evidence present but no numeric value)
    runway_numeric_fail = _base_row()
    runway_numeric_fail["RunwayQuarters"] = ""
    runway_numeric_fail["Runway (qtrs)"] = ""
    rows.append(runway_numeric_fail)

    # Subscore count fail
    subscore_fail = _base_row()
    subscore_fail["Subscores Evidenced (x/5)"] = 3
    rows.append(subscore_fail)

    # Biotech peer fail
    biotech_fail = _base_row()
    biotech_fail["Sector"] = "Healthcare"
    biotech_fail["Industry"] = "Biotechnology"
    biotech_fail["BiotechPeerRead"] = "TBD - pending"
    rows.append(biotech_fail)

    # Materiality fail
    materiality_fail = _base_row()
    materiality_fail["Materiality"] = "Fail - risk"
    materiality_fail["Materiality (pass/fail + note)"] = ""
    rows.append(materiality_fail)

    df = pd.DataFrame(rows)

    stats = summarize_validation_gates(df)

    assert stats["N_rows"] == 7
    for label in [
        "GATE_UNIVERSE",
        "GATE_MANDATORY_SUBSCORES",
        "GATE_RUNWAY_NUMERIC",
        "GATE_SUBSCORE_COUNT",
        "GATE_BIOTECH_PEER",
        "GATE_MATERIALITY",
    ]:
        assert stats[f"{label}_pass"] == 6
        assert stats[f"{label}_fail"] == 1
    assert stats["N_validated_all_true"] == 1
    assert stats["N_any_gate_false"] == 6


def test_evaluate_validation_alignment():
    df = pd.DataFrame([_base_row()])
    status, reason = evaluate_validation(df.iloc[0])
    assert status == "Validated"
    assert reason == ""

    failing_row = _base_row()
    failing_row["RunwayEvidencePrimary"] = ""
    failing_row["RunwayQuarters"] = ""
    status_fail, reason_fail = evaluate_validation(pd.Series(failing_row))
    assert status_fail.startswith("TBD")
    assert "Runway" in reason_fail
