from pathlib import Path
import sys

import pandas as pd

from app.csv_names import csv_filename

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import build_watchlist


def test_run_drops_dilution_only_catalyst(tmp_path):
    data_dir = Path(tmp_path)

    research_rows = [
        {
            "Ticker": "AISP",
            "CIK": "1842566",
            "Company": "Airship AI Holdings, Inc.",
            "Sector": "Technology",
            "Price": 4.33,
            "MarketCap": 135_000_000,
            "ADV20": 103_000,
            "RunwayQuartersRaw": 3.2,
            "RunwayCashRaw": 6_306_274,
            "RunwayCash": 6_306_274,
            "RunwayQuarterlyBurnRaw": 1_959_409.5,
            "RunwayQuarterlyBurn": 1_959_409.5,
            "RunwayEstimate": "interim",
            "RunwayNotes": "values parsed from XBRL",
            "RunwaySourceForm": "10-Q",
            "RunwaySourceDate": "2025-08-05",
            "RunwaySourceURL": "https://www.sec.gov/ix?doc=/Archives/example/10q.htm",
            "Catalyst": "2025-10-28 S-3 https://www.sec.gov/ix?doc=/Archives/example/s3.htm",
            "Dilution": "2025-10-28 S-3 https://www.sec.gov/ix?doc=/Archives/example/s3.htm",
            "Governance": "2025-10-27 DEF 14A https://www.sec.gov/ix?doc=/Archives/example/def14a.htm",
            "FilingsSummary": "2025-10-28 S-3 https://www.sec.gov/ix?doc=/Archives/example/s3.htm",
        }
    ]

    research_path = data_dir / csv_filename("dr_populate_results")
    pd.DataFrame(research_rows).to_csv(research_path, index=False)

    rows_written, status = build_watchlist.run(data_dir=str(data_dir))

    assert status == "ok"
    assert rows_written == 0

    output_path = data_dir / csv_filename("validated_watchlist")
    output = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(output) == 1  # header only
