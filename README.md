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
