import pandas as pd

from app.filings_filters import prune_filings_by_role


def _base_cfg():
    return {
        "FilingsWhitelistByRole": {
            "RunwayAndGovernance": ["10-Q", "10-K", "6-K", "DEF 14A"],
            "Dilution": ["S-3", "S-8", "424B5", "424B3"],
            "Catalyst": ["8-K", "6-K"],
            "Insider": ["4", "4/A"],
        },
        "FilingsGroups": {
            "Governance": {
                "forms": ["DEF 14A"],
            }
        },
        "FilingsPerTickerLimits": {
            "Runway": 2,
            "Dilution": 1,
            "Catalyst": 1,
            "Insider": 1,
            "Governance": 4,
        },
    }


def test_pruning_respects_role_caps_and_keeps_runway():
    cfg = _base_cfg()
    df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Form": "10-Q", "FiledAt": "2024-05-10", "URL": "u1"},
            {"Ticker": "AAA", "Form": "10-K", "FiledAt": "2024-03-01", "URL": "u2"},
            {"Ticker": "AAA", "Form": "10-Q", "FiledAt": "2023-12-01", "URL": "u3"},
            {"Ticker": "AAA", "Form": "6-K", "FiledAt": "2024-04-01", "URL": "u4"},
            {"Ticker": "AAA", "Form": "S-3", "FiledAt": "2024-05-02", "URL": "u5"},
            {"Ticker": "AAA", "Form": "S-8", "FiledAt": "2024-02-02", "URL": "u6"},
            {"Ticker": "AAA", "Form": "8-K", "FiledAt": "2024-05-04", "URL": "u7"},
            {"Ticker": "AAA", "Form": "4", "FiledAt": "2024-05-03", "URL": "u8"},
            {"Ticker": "AAA", "Form": "DEF 14A", "FiledAt": "2024-01-01", "URL": "u9"},
            {"Ticker": "AAA", "Form": "OTHER", "FiledAt": "2024-06-01", "URL": "u10"},
        ]
    )

    pruned, applied = prune_filings_by_role(df, cfg)

    runway_forms = pruned[pruned["Form"].str.startswith("10-", na=False)]
    assert len(runway_forms) == 2
    assert "u1" in runway_forms["URL"].values
    assert "u2" in runway_forms["URL"].values

    dilution = pruned[pruned["Form"].str.startswith("S-", na=False)]
    assert len(dilution) == 1
    assert "u5" in dilution["URL"].values

    catalyst = pruned[pruned["Form"] == "8-K"]
    assert len(catalyst) == 1

    insider = pruned[pruned["Form"] == "4"]
    assert len(insider) == 1

    governance = pruned[pruned["Form"] == "DEF 14A"]
    assert not governance.empty

    assert "OTHER" in pruned["Form"].values
    assert "AAA" in applied
    assert applied["AAA"]["Runway"]["cap"] == 2


def test_pruning_is_per_ticker_and_retains_runway_even_with_zero_cap():
    cfg = _base_cfg()
    cfg["FilingsPerTickerLimits"]["Runway"] = 0
    cfg["FilingsPerTickerLimits"]["Dilution"] = 2

    df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Form": "10-Q", "FiledAt": "2024-05-10", "URL": "a1"},
            {"Ticker": "AAA", "Form": "10-Q", "FiledAt": "2024-04-10", "URL": "a2"},
            {"Ticker": "BBB", "Form": "10-K", "FiledAt": "2024-05-09", "URL": "b1"},
            {"Ticker": "BBB", "Form": "S-3", "FiledAt": "2024-05-01", "URL": "b2"},
            {"Ticker": "BBB", "Form": "S-3", "FiledAt": "2024-04-01", "URL": "b3"},
            {"Ticker": "BBB", "Form": "S-8", "FiledAt": "2024-03-01", "URL": "b4"},
        ]
    )

    pruned, applied = prune_filings_by_role(df, cfg)

    aaa_runway = pruned[(pruned["Ticker"] == "AAA") & (pruned["Form"].str.startswith("10-", na=False))]
    assert len(aaa_runway) >= 1
    assert "a1" in aaa_runway["URL"].values

    bbb_dilution = pruned[(pruned["Ticker"] == "BBB") & (pruned["Form"].str.startswith("S-", na=False))]
    assert len(bbb_dilution) == 2
    assert set(bbb_dilution["URL"]) == {"b2", "b3"}

    assert set(pruned["Ticker"]) == {"AAA", "BBB"}
    assert set(df.columns).issubset(set(pruned.columns))

