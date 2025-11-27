# Weekly Pipeline Status

## Stage Coverage (W0–W4)

| Stage | Implemented? (Y/Partial/N) | Module(s) | Notes |
| --- | --- | --- | --- |
| W0 | Partial | `app/pipeline.run_weekly_pipeline` | Run-level controls (config load, single-run lock, stage resume list) exist; spec file `weekly.txt` not present in repo, so coverage is inferred from code paths.【F:app/pipeline.py†L1450-L1494】 |
| W1 | Partial | `app/pipeline` | Uses config-driven gates (e.g., spreads/ADV filters, lookbacks) during profile/filing ingestion but no explicit weekly control spec available for verification.【F:app/pipeline.py†L645-L759】【F:app/pipeline.py†L900-L1079】 |
| W2.A Counts | Partial | `app/pipeline.universe_step`, `profiles_step`, `filings_step` | Universe/profiles/filings caches are pulled and logged; counts tracked in runlog but no dedicated summary table per spec visible.【F:app/pipeline.py†L633-L642】【F:app/pipeline.py†L645-L759】【F:app/pipeline.py†L900-L1079】 |
| W2.B Candidate Shortlist | Y | `app/pipeline.hydrate_and_shortlist_step`, `app/hydrate.py`, `app/shortlist.py` | Hydrates candidates from caches and filters into `06_shortlist_candidates.csv`.【F:app/pipeline.py†L1205-L1238】【F:app/hydrate.py†L1-L87】【F:app/shortlist.py†L1-L24】 |
| W3 (subscores/evidence) | Partial | `deep_research.py`, `runway_extract.py` via `parse_q10_step`, `app/build_watchlist.py` (8-K parse + scoring) | Deep research builds catalyst/dilution bundles but runway is stubbed; runway extraction and 8-K parsing run separately; governance/insider/ownership logic lives in watchlist builder rather than a standalone W3 output.【F:deep_research.py†L141-L199】【F:deep_research.py†L368-L399】【F:app/pipeline.py†L1241-L1367】【F:app/build_watchlist.py†L1555-L1657】 |
| W4 (Validated Candidate Set) | Partial | `app/build_watchlist.run`, `app/pipeline.build_watchlist_step` | Produces `11_validated_watchlist.csv` with validation checks, but final selection rubric vs spec not confirmed; depends on upstream dr_populate and market filters.【F:app/build_watchlist.py†L1555-L1657】【F:app/build_watchlist.py†L1997-L2120】【F:app/pipeline.py†L1430-L1447】 |

## Data & CSV Flow

| Output | CSV Filename | Producer Module/Function | Consumer Stage(s) | Notes |
| --- | --- | --- | --- | --- |
| Profiles cache | `01_profiles.csv` | `app/pipeline.profiles_step` | Prices, Hydrate/Shortlist | Fetches FMP profiles/quotes and applies spread/price gates before caching.【F:app/pipeline.py†L645-L759】【F:app/csv_names.py†L4-L15】 |
| Filings cache | `02_filings.csv` | `app/pipeline.filings_step` (EdgarAdapter batches) | FDA filter, Hydrate, Runway/8-K parsing | Applies form lookbacks and runway gate, writes deduped filings with Age column.【F:app/pipeline.py†L900-L1079】【F:app/csv_names.py†L4-L15】 |
| Prices cache | `03_prices.csv` | `app/pipeline.prices_step` | Hydrate, Build Watchlist | Pulls historical prices/volume and computes ADV20 before caching.【F:app/pipeline.py†L1174-L1202】【F:app/csv_names.py†L4-L15】 |
| FDA events | `04_fda.csv` | `app/pipeline.fda_step` | Hydrate, Deep research catalyst context | Filters to filings tickers and caches device/drug events.【F:app/pipeline.py†L1082-L1171】【F:app/csv_names.py†L4-L15】 |
| Hydrated candidates | `05_hydrated_candidates.csv` | `app/pipeline.hydrate_and_shortlist_step` via `app/hydrate.hydrate_candidates` | Shortlist, Build Watchlist (market merge) | Joins profiles, prices, filings, FDA to latest signals before shortlist.【F:app/pipeline.py†L1205-L1231】【F:app/hydrate.py†L1-L87】【F:app/csv_names.py†L4-L15】 |
| Shortlist | `06_shortlist_candidates.csv` | `app/pipeline.hydrate_and_shortlist_step` via `app/shortlist.build_shortlist` | Deep Research | Applies price/cap/ADV and catalyst presence filters to hydrated set.【F:app/pipeline.py†L1226-L1238】【F:app/shortlist.py†L1-L24】【F:app/csv_names.py†L9-L15】 |
| Deep research results | `07_deep_research_results.csv` | `app/pipeline.deep_research_step` invoking `deep_research.run` | Runway extract, dr_populate | Generates catalyst/dilution bundles with placeholder runway notes.【F:app/pipeline.py†L1241-L1275】【F:deep_research.py†L341-L399】【F:app/csv_names.py†L10-L15】 |
| Runway extract | `08_runway_extract_results.csv` | `app/pipeline.parse_q10_step` calling `runway_extract.run` | dr_populate | Parses 10-Q/10-K for runway metrics after deep research output present.【F:app/pipeline.py†L1278-L1331】【F:app/csv_names.py†L11-L15】 |
| 8-K events | `09_8k_events.csv` | `app/pipeline.parse_8k_step` via `app.build_watchlist.generate_eight_k_events` | Build Watchlist (catalyst/trigger checks) | Uses edgartools parser to classify 8-K items and exhibits.【F:app/pipeline.py†L1333-L1367】【F:app/csv_names.py†L12-L15】 |
| DR populate | `10_dr_populate_results.csv` | `app/pipeline.dr_populate_step` invoking `app.dr_populate.run` | Build Watchlist | Aggregates filings/runway outputs into research records with subscores/materiality hints.【F:app/pipeline.py†L1369-L1423】【F:app/csv_names.py†L13-L15】 |
| Validated watchlist | `11_validated_watchlist.csv` | `app/build_watchlist.run` via `app/pipeline.build_watchlist_step` | Final selections artifact | Applies tiering/checklists and writes validated survivors with subscores/evidence columns.【F:app/build_watchlist.py†L1555-L1657】【F:app/build_watchlist.py†L1997-L2120】【F:app/csv_names.py†L14-L15】 |

## EDGAR/edgartools Usage

- **EdgarAdapter** pulls ticker universe, filings, and filing text through `edgar` helpers (`get_company_tickers`, `get_by_accession_number_enriched`, `download_text`) with rate limiting and identity configuration.【F:app/edgar_adapter.py†L8-L129】【F:app/edgar_adapter.py†L131-L179】 The weekly pipeline sets this adapter before universe/filings steps.【F:app/pipeline.py†L1456-L1494】
- **8-K parsing** uses `edgar.company_reports.EightK` and `Filing.from_sgml_text` to extract items, exhibits, and press releases for catalyst classification.【F:app/eight_k_parser.py†L8-L176】
- **Direct SEC HTTP (non-edgartools)** persists in `app/sec.load_sec_universe`, which fetches the raw JSON ticker list via `HttpClient`; this path is not wired into the weekly runner and is a candidate for later conversion to edgartools or removal.【F:app/sec.py†L5-L46】
- **Runway parsing** relies on local `runway_extract`/`parser_10q` helpers rather than edgartools objects; potential future alignment could wrap these around edgartools filing retrievals.

## Path to Final Weekly Selections

- **Current Final Selections Artefact:** `data/11_validated_watchlist.csv` written by `app/build_watchlist.run` after dr_populate/8-K parsing, serving as the closest “validated candidate set.”【F:app/build_watchlist.py†L1555-L1657】【F:app/build_watchlist.py†L1997-L2120】
- **Target Final Selections Artefact:** Standardise on `data/validated_selections.csv` unless the existing `11_validated_watchlist.csv` remains the canonical output; current code expects the latter name via `csv_names` mapping.【F:app/csv_names.py†L14-L15】
- **Gaps toward W3.c & W4:** Runway values in deep research are stubbed, and dilution/governance/insider evidence is mostly assembled inside the watchlist builder without a standalone W3 evidence bundle. Validation rules exist, but the rubric vs. `weekly.txt` cannot be confirmed because the spec file is absent from the tree.【F:deep_research.py†L141-L199】【F:app/build_watchlist.py†L1555-L1657】

## Legacy / Obsolete Candidates

### Modules Likely Legacy

| Module Path | Reason (No imports? Old pipeline? Duplicated?) |
| --- | --- |
| `app/sec.py` | Provides direct SEC JSON universe fetch but is unused by the weekly runner, which relies on `EdgarAdapter.load_company_universe`; good candidate for consolidation or removal later.【F:app/sec.py†L5-L46】【F:app/pipeline.py†L633-L642】 |

### Tests Likely Legacy

| Test File | Module(s) Tested | Reason |
| --- | --- | --- |
| _None identified_ | — | Current test suite targets active adapters/parsers; no obviously obsolete tests detected in this pass. |
