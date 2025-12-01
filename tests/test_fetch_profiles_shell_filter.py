from app.fmp import fetch_profiles


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, payloads: list[list[dict]]):
        self._payloads = payloads
        self.calls = []

    def get(self, url, params, ratelimit):
        self.calls.append((url, params, ratelimit))
        payload = self._payloads.pop(0)
        return DummyResponse(payload)

    def stats_string(self):
        return "dummy-stats"


def test_fetch_profiles_drops_shell_companies():
    client = DummyClient(
        [
            [
                {
                    "symbol": "CRAQ",
                    "companyName": "Shell Example",
                    "exchangeShortName": "NASDAQ",
                    "industry": "Shell Companies",
                    "price": 2.5,
                    "mktCap": 1_000_000,
                },
                {
                    "symbol": "GOOD",
                    "companyName": "Good Co",
                    "exchangeShortName": "NYSE",
                    "industry": "Software",
                    "price": 5.0,
                    "mktCap": 50_000_000,
                },
            ]
        ]
    )

    cfg = {
        "FMPKey": "test-key",
        "BatchSizes": {"Profiles": 10},
        "RateLimitsPerMin": {"FMP": 1200},
        "Universe": {
            "Exchanges": ["NASDAQ", "NYSE"],
            "DropPatterns": [],
            "DropWordPatterns": [],
        },
        "HardGates": {"MinPrice": 0, "CapMin": 0, "CapMax": 1_000_000_000},
    }

    profiles = fetch_profiles(client, cfg, ["CRAQ", "GOOD"])

    assert len(profiles) == 1
    assert profiles[0]["Ticker"] == "GOOD"
    assert profiles[0]["Industry"] == "Software"
