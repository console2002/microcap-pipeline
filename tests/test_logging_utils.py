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
