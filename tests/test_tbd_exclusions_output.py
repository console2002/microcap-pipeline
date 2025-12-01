import pandas as pd

from app.weekly_validated import build_validated_selections


def test_tbd_exclusions_propagate_fields_and_status(tmp_path):
    data_dir = tmp_path

    deep_rows = [
        {
            "Ticker": "AAA",
            "Company": "Alpha",
            "CIK": 1,
            "Sector": "Health Care",
            "Industry": "Biotechnology",
            # Missing evidence forces a TBD outcome
            "RunwayQuarters": None,
            "Runway (qtrs)": "",
            "RunwayEvidencePrimary": "",
            "Dilution": "",
            "DilutionEvidencePrimary": "",
            "Catalyst": "",
            "CatalystEvidencePrimary": "",
            "SubscoresEvidencedCount": 0,
            "Materiality": "fail",
            "BiotechPeerRead": "TBD",
        }
    ]

    pd.DataFrame(deep_rows).to_csv(data_dir / "30_deep_research.csv", index=False)

    validated_df, tbd_df = build_validated_selections(data_dir=str(data_dir))

    assert validated_df.empty
    assert len(tbd_df) == 1

    row = tbd_df.iloc[0]
    assert row["Sector"] == "Health Care"
    assert row["Industry"] == "Biotechnology"
    assert row["Status"] == "TBD - exclude"
    assert "—" not in row["Status"]

    output_path = data_dir / "40_tbd_exclusions.csv"
    written = pd.read_csv(output_path)
    assert "—" not in "".join(written["Status"].astype(str))
