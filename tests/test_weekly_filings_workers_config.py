from app.config import load_config
from app.edgar_adapter import EdgarAdapter


def _fake_fetch(self, ticker, *, whitelist, start_expr, progress_fn=None, stop_flag=None, idx=0, total=0):
    return [], {"ticker": ticker, "raw_count": 0, "kept_count": 0, "duration_ms": 0}


def test_filings_workers_sequential_path(monkeypatch, tmp_path):
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg.setdefault("Workers", {}).setdefault("EDGAR", {})["Workers"] = 1

    called = {"executor": False}

    class FailExecutor:  # pragma: no cover - defensive
        def __init__(self, *args, **kwargs):
            called["executor"] = True
            raise AssertionError("ThreadPoolExecutor should not be used when workers=1")

    monkeypatch.setattr("app.edgar_adapter.ThreadPoolExecutor", FailExecutor)
    monkeypatch.setattr(EdgarAdapter, "_fetch_filings_for_ticker", _fake_fetch, raising=False)

    adapter = EdgarAdapter(cfg)
    adapter.fetch_recent_filings(["AAA"], progress_fn=None)

    assert called["executor"] is False


def test_filings_workers_parallel_path(monkeypatch, tmp_path):
    cfg = load_config()
    cfg["Paths"]["data"] = str(tmp_path)
    cfg["Paths"]["logs"] = str(tmp_path / "logs")
    cfg.setdefault("Workers", {}).setdefault("EDGAR", {})["Workers"] = 4

    called = {"executor": False}

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class TrackingExecutor:
        def __init__(self, *args, **kwargs):
            called["executor"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    def fake_as_completed(futures):
        for fut in list(futures):
            yield fut

    monkeypatch.setattr("app.edgar_adapter.ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr("app.edgar_adapter.as_completed", fake_as_completed)
    monkeypatch.setattr(EdgarAdapter, "_fetch_filings_for_ticker", _fake_fetch, raising=False)

    adapter = EdgarAdapter(cfg)
    adapter.fetch_recent_filings(["AAA", "BBB"], progress_fn=None)

    assert called["executor"] is True
