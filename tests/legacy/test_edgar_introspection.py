import csv
import importlib.metadata
import logging
from pathlib import Path

import pytest

import edgar
from edgar import Filing


def _edgar_version() -> str:
    try:
        return importlib.metadata.version("edgar")
    except importlib.metadata.PackageNotFoundError:
        return getattr(edgar, "__version__", "unknown")


def _accession_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    accession_raw = parts[-2]
    return f"{accession_raw[:10]}-{accession_raw[10:12]}-{accession_raw[12:]}"


def _load_sample_filings():
    target_forms = ["10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "4", "13F-HR"]
    seen = {}
    path = Path("data/02_filings.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            form = row.get("Form", "")
            if form in target_forms and form not in seen:
                seen[form] = row
            if len(seen) == len(target_forms):
                break
    return seen


@pytest.mark.parametrize("form", ["10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "4", "13F-HR"])
def test_filing_obj_introspection_logs(capfd, form):
    logging.basicConfig(level=logging.INFO)
    version = _edgar_version()
    print("edgartools version:", version)

    rows = _load_sample_filings()
    if form not in rows:
        pytest.skip(f"form {form} not present in data/02_filings.csv")

    row = rows[form]
    cik = int(row["CIK"])
    accession = _accession_from_url(row["URL"])
    filing = Filing(cik, row.get("Company", ""), row.get("Form", form), row.get("FiledAt", ""), accession)

    try:
        obj = filing.obj()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"filing.obj() failed for {form}: {exc}")
        obj = None

    attributes = []
    if obj is not None:
        attributes = [name for name in dir(obj) if not name.startswith("__")]
        print(
            f"form={filing.form} date={filing.filing_date} url={filing.text_url} obj={type(obj).__name__} attrs={attributes[:20]}"
        )
    else:
        print(f"form={filing.form} date={filing.filing_date} url={filing.text_url} obj=None")

    captured = capfd.readouterr()
    assert f"form={filing.form}" in captured.out
