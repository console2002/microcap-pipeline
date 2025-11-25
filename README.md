# microcap-pipeline

Goal:
- Pull micro-cap universe data from SEC, FMP, and openFDA
- Maintain rolling CSV caches (profiles, prices, filings, fda)
- Build hydrated candidates + shortlist
- Run manually (Weekly / Daily) or via GUI
- Enforce hard gates and rate limits, and log everything

## Prereqs

- Windows 10/11
- Python 3.11+ installed and on PATH (`python --version`)

## 1. Create venv

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. Install dependencies

Install the core requirements (now including [edgartools](https://pypi.org/project/edgartools/) for SEC data ingestion):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure EDGAR identity

Set a valid EDGAR user agent in `config.json` under the `Edgar` section (falls back to the top-level `UserAgent`).

```json
"Edgar": {
  "UserAgent": "Firstname Lastname email@example.com",
  "ThrottlePerMin": 60
}
```

This identity is required for all `edgartools` requests. Update the throttle to align with your SEC usage policy.
