import pandas as pd

from app.biotech_utils import classify_peer_events, get_biotech_peers, is_biotech
from app.weekly_deep_research import (
    _biotech_peer_read_from_events,
    _prepare_events,
    _status_from_row,
)


def test_is_biotech_detection():
    assert is_biotech("Healthcare", "Biotechnology")
    assert is_biotech("Health Care", "Drug Manufacturers")
    assert not is_biotech("Technology", "Software")
    assert not is_biotech("Financials", "Banks")


def test_peer_selection_prefers_same_industry():
    universe = pd.DataFrame(
        [
            {"Ticker": "BIO1", "Sector": "Healthcare", "Industry": "Biotechnology"},
            {"Ticker": "BIO2", "Sector": "Healthcare", "Industry": "Biotechnology"},
            {"Ticker": "BIO3", "Sector": "Healthcare", "Industry": "Pharmaceuticals"},
            {"Ticker": "TECH", "Sector": "Technology", "Industry": "Software"},
        ]
    )

    peers = get_biotech_peers(universe, "BIO1", "Healthcare", "Biotechnology")
    assert "BIO2" in peers
    assert "BIO1" not in peers
    assert "TECH" not in peers


def test_classify_peer_events_positive_negative_mixed_none():
    base_date = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    events = pd.DataFrame(
        [
            {"Ticker": "PEER1", "event_type": "Phase III positive", "Tier": "Tier-1", "event_date": base_date},
            {"Ticker": "PEER2", "event_type": "CRL issued", "Tier": "Tier-1", "event_date": base_date},
        ]
    )

    pos_only = classify_peer_events(events.iloc[[0]])
    assert pos_only.code == "POSITIVE"

    neg_only = classify_peer_events(events.iloc[[1]])
    assert neg_only.code == "NEGATIVE"

    mixed = classify_peer_events(events)
    assert mixed.code == "MIXED"

    none = classify_peer_events(pd.DataFrame())
    assert none.code == "NONE"


def test_biotech_peer_read_integration_positive_and_none():
    today = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    universe = pd.DataFrame(
        [
            {"Ticker": "BIOA", "Sector": "Healthcare", "Industry": "Biotechnology"},
            {"Ticker": "BIOB", "Sector": "Healthcare", "Industry": "Biotechnology"},
        ]
    )

    events = _prepare_events(
        pd.DataFrame(
            [
                {
                    "Ticker": "BIOB",
                    "event_type": "Phase III positive",
                    "Tier": "Tier-1",
                    "event_date": today,
                }
            ]
        )
    )

    peer_read, evidence = _biotech_peer_read_from_events("BIOA", "Healthcare", "Biotechnology", events, universe)
    assert peer_read.startswith("Y_POS")
    assert "BIOB" in peer_read
    assert evidence

    peer_read_none, evidence_none = _biotech_peer_read_from_events(
        "BIOA", "Healthcare", "Biotechnology", _prepare_events(pd.DataFrame()), universe
    )
    assert peer_read_none == "N:NoPeerEvent"
    assert evidence_none == ""


def test_validation_flag_effects():
    # When biotech peer evidence is required, lack of evidence forces exclusion.
    status_excluded = _status_from_row(
        mandatory_ok=True,
        subscore_count=5,
        materiality="PASS - catalyst",
        biotech_peer="N:NoPeerEvent",
        is_biotech_candidate=True,
        require_biotech_peer=True,
    )
    assert status_excluded.startswith("TBD")

    # Without the requirement, status passes through.
    status_allowed = _status_from_row(
        mandatory_ok=True,
        subscore_count=5,
        materiality="PASS - catalyst",
        biotech_peer="N:NoPeerEvent",
        is_biotech_candidate=True,
        require_biotech_peer=False,
    )
    assert status_allowed == "Validated"

    # Non-biotech remains unaffected even when requirement is on.
    status_non_biotech = _status_from_row(
        mandatory_ok=True,
        subscore_count=5,
        materiality="PASS - catalyst",
        biotech_peer="N:NonBiotech",
        is_biotech_candidate=False,
        require_biotech_peer=True,
    )
    assert status_non_biotech == "Validated"


def test_weekly_validated_enforces_biotech_peer_when_enabled(monkeypatch):
    import app.weekly_validated as weekly_validated

    monkeypatch.setattr(weekly_validated, "BIOTECH_PEER_REQUIRED_FOR_VALIDATION", True)

    biotech_row = pd.Series(
        {
            "Sector": "Healthcare",
            "Industry": "Biotechnology",
            "RunwayQuarters": 4,
            "RunwayEvidencePrimary": "url",
            "Dilution": "High",
            "DilutionEvidencePrimary": "url",
            "Catalyst": "Tier-1",
            "CatalystEvidencePrimary": "url",
            "Subscores Evidenced (x/5)": 4,
            "Materiality (pass/fail + note)": "PASS - test",
            "BiotechPeerRead": "N:NoPeerEvent",
        }
    )

    status_biotech, reason_biotech = weekly_validated.evaluate_validation(biotech_row)
    assert status_biotech == "TBD - exclude"
    assert "Biotech peer missing" in reason_biotech

    non_biotech_row = biotech_row.copy()
    non_biotech_row["Sector"] = "Technology"
    non_biotech_row["Industry"] = "Software"
    non_biotech_row["BiotechPeerRead"] = "N:NonBiotech"

    status_non_biotech, _ = weekly_validated.evaluate_validation(non_biotech_row)
    assert status_non_biotech == "Validated"
