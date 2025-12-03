import json

from app import logging_utils


def test_log_diag_writes_json_line(tmp_path, monkeypatch):
    path = tmp_path / "diag.jsonl"
    cfg = {
        "Diagnostics": {"Enabled": True, "Path": str(path)},
        "Paths": {"data": str(tmp_path), "logs": str(tmp_path)},
    }

    logging_utils._DIAG_CONFIG = None

    logging_utils.log_diag(
        stage="universe",
        ticker="TEST",
        cik="0000000000",
        decision="gate_pass",
        details="unit test",
        fields={"sample": True},
        cfg=cfg,
    )

    assert path.exists(), "diagnostics file should be created"
    lines = path.read_text().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["stage"] == "universe"
    assert payload["decision"] == "gate_pass"
    assert payload["fields"]["sample"] is True


def test_log_diag_produces_true_jsonl(tmp_path):
    path = tmp_path / "diag.jsonl"
    cfg = {
        "Diagnostics": {"Enabled": True, "Path": str(path)},
        "Paths": {"data": str(tmp_path), "logs": str(tmp_path)},
    }

    logging_utils._DIAG_CONFIG = None

    logging_utils.log_diag(
        stage="runway",
        ticker="ABOS",
        cik=None,
        decision="runway_missing",
        details="no filings",
        fields={"forms": []},
        cfg=cfg,
    )

    line = path.read_text().splitlines()[0]
    assert not line.startswith("\"")

    payload = json.loads(line)
    assert payload["ticker"] == "ABOS"
    assert payload["stage"] == "runway"
    assert payload["decision"] == "runway_missing"
    assert "fields" in payload


def test_log_diag_includes_error_metadata(tmp_path):
    path = tmp_path / "diag.jsonl"
    cfg = {
        "Diagnostics": {"Enabled": True, "Path": str(path)},
        "Paths": {"data": str(tmp_path), "logs": str(tmp_path)},
    }

    logging_utils._DIAG_CONFIG = None

    try:
        raise ValueError("boom")
    except ValueError as exc:
        logging_utils.log_diag(
            stage="events",
            ticker="TEST",
            cik=None,
            decision="event_parse_error",
            details="parse failure",
            fields={},
            cfg=cfg,
            error=exc,
        )

    payload = json.loads(path.read_text().splitlines()[0])
    assert payload.get("error_type") == "ValueError"
    assert payload.get("error_message") == "boom"
    assert payload.get("traceback")
