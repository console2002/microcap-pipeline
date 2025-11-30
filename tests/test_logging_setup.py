import logging

from app import logging_utils
from app.logging_utils import setup_logging


def test_setup_logging_attaches_file_handler(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_utils, "_LOGGING_CONFIGURED", False)

    def fake_load_config():
        return {"Paths": {"logs": str(tmp_path)}}

    monkeypatch.setattr(logging_utils, "load_config", fake_load_config)

    setup_logging()

    logger = logging.getLogger("edgar_core")
    handlers = logger.handlers or logging.getLogger().handlers
    assert handlers, "Expected at least one handler to be attached"
    assert any(getattr(h, "level", logging.NOTSET) <= logging.INFO for h in handlers)
