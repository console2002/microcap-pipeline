import pandas as pd

from app.weekly_validated import evaluate_validation, summarize_validation_gates


def _base_row():
    return {
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


def test_summarize_validation_gates_counts():
    rows = []

    # All gates pass
    rows.append(_base_row())

    # C1 fail (runway evidence missing and no values)
    runway_fail = _base_row()
    runway_fail["RunwayQuarters"] = ""
    runway_fail["Runway (qtrs)"] = ""
    runway_fail["RunwayEvidencePrimary"] = ""
    rows.append(runway_fail)

    # C2 fail (dilution evidence missing)
    dilution_fail = _base_row()
    dilution_fail["Dilution"] = ""
    dilution_fail["DilutionScore"] = ""
    dilution_fail["DilutionEvidencePrimary"] = ""
    rows.append(dilution_fail)

    # C3 fail (catalyst evidence missing)
    catalyst_fail = _base_row()
    catalyst_fail["Catalyst"] = ""
    catalyst_fail["CatalystScore"] = ""
    catalyst_fail["CatalystEvidencePrimary"] = ""
    rows.append(catalyst_fail)

    # C6 fail (materiality)
    materiality_fail = _base_row()
    materiality_fail["Materiality"] = "Fail - risk"
    materiality_fail["Materiality (pass/fail + note)"] = ""
    rows.append(materiality_fail)

    # C5 fail (biotech TBD)
    biotech_fail = _base_row()
    biotech_fail["BiotechPeerRead"] = "TBD - pending"
    rows.append(biotech_fail)

    # C4 fail (insufficient subscores)
    subscore_fail = _base_row()
    subscore_fail["Subscores Evidenced (x/5)"] = 3
    rows.append(subscore_fail)

    df = pd.DataFrame(rows)

    stats = summarize_validation_gates(df)

    assert stats["N_rows"] == 7
    assert stats["C1_pass"] == 6 and stats["C1_fail"] == 1
    assert stats["C2_pass"] == 6 and stats["C2_fail"] == 1
    assert stats["C3_pass"] == 6 and stats["C3_fail"] == 1
    assert stats["C4_pass"] == 6 and stats["C4_fail"] == 1
    assert stats["C5_pass"] == 6 and stats["C5_fail"] == 1
    assert stats["C6_pass"] == 6 and stats["C6_fail"] == 1
    assert stats["N_validated_all_true"] == 1
    assert stats["N_any_gate_false"] == 6


def test_evaluate_validation_alignment():
    df = pd.DataFrame([_base_row()])
    status, reason = evaluate_validation(df.iloc[0])
    assert status == "Validated"
    assert reason == ""

    failing_row = _base_row()
    failing_row["RunwayEvidencePrimary"] = ""
    status_fail, reason_fail = evaluate_validation(pd.Series(failing_row))
    assert status_fail.startswith("TBD")
    assert "Runway" in reason_fail
