# Calibration data — Data agent notes

## City choice: Bengaluru, Karnataka

Bengaluru was chosen as the geographic universe for ATM/POS terminals because:
- It has dense, well-maintained OSM tagging (large local mapping community,
  tech-hub city) — an `amenity=atm` bbox query returns over a thousand nodes,
  comfortably above the "at least a few hundred" bar.
- It is also the single Indian city most associated with cybercrime case
  volume in NCRB's own published data (highest cybercrime rate among metro
  cities, 2023 and 2024 both), which makes it a natural, defensible choice
  for a fraud/mule-network detection prototype — real-world plausibility, not
  just data availability.
- Karnataka state-level NCRP numbers are available (see below), giving a
  clean state -> city calibration chain.

## OSM ATM fetch

- Query: Overpass API, `node["amenity"="atm"]` within bounding box
  `(12.83, 77.45, 13.14, 77.78)` (south, west, north, east) — covers greater
  Bengaluru / BBMP limits plus margin.
- A bbox query was used instead of an OSM administrative-boundary `area`
  query for reliability (avoids depending on a specific boundary relation
  being correctly tagged/found).
- Result: **1150 ATM nodes**, saved unmodified to
  `data/raw/osm_atms_bengaluru.json` (SHA256 in `manifest.json`).
- First fetch attempt hit a transient `RemoteDisconnected` error; the script's
  retry/backoff logic succeeded on attempt 2. No fabricated data was used at
  any point — the script aborts rather than inventing coordinates if Overpass
  stays unreachable after retries.
- Reproduce via `scripts/fetch_calibration.py` (idempotent — skips the fetch
  if the raw file already exists, to respect Overpass rate limits).

## Terminals table — synthetic augmentation disclosure

Per SCHEMA_CONTRACT.md, `terminals` needs `bank` and `install_date` columns
that OSM does not provide for ATM nodes. These two columns ONLY are
synthesized:
- `bank`: sampled uniformly (seeded RNG, seed=42) from 6 real Indian bank
  names (State Bank of India, HDFC Bank, ICICI Bank, Axis Bank, Punjab
  National Bank, Canara Bank).
- `install_date`: sampled uniformly from the past 1-5 years (seeded RNG).

All other columns (`terminal_id`, `lat`, `lon`, `h3_cell`, `type`) are derived
directly from real OSM coordinates — `type` is hardcoded `"ATM"` since we only
queried `amenity=atm` (did not add POS/shop nodes; 1150 ATM points already
comfortably exceeded the minimum bar, so we kept scope thin per the "don't
over-build" guidance).

H3 indexing: resolution 9, via `h3.latlng_to_cell(lat, lon, 9)` (h3 package
v4 API — note the v4 rename from `h3.geo_to_h3`).

Build script: `scripts/build_terminals.py`. Output written to both
`data/simulated/fixture/terminals.parquet` (doubles as the SCHEMA_CONTRACT.md
"fixture" for early Model/API integration — all 1150 real terminals, not a
subsampled 50, since the full set is already small) and
`data/raw/terminals_full.parquet` (raw copy, per task instructions).

## Calibration numbers found (real, cited, hand-transcribed)

No downloadable CSV/API dataset with a clear open licence was found for
either NCRP or RBI figures in the time available (NCRP's own portal
dashboards are not publicly downloadable; RBI's DBIE / Handbook of Statistics
publishes as PDF tables, not a simple public bulk CSV/API within a hackathon
time budget). Per the task instructions, numbers were hand-transcribed from
citable public sources (government Rajya Sabha statements, RBI-sourced
figures as reported by mainstream Indian business press, NCRB "Crime in
India" figures as reported by Deccan Herald / Business Standard /
TheNewsMinute). Every number has a source URL and access date recorded in
`manifest.json`. No number here was invented — all were fetched via
WebSearch/WebFetch from a real, citable page.

Files:
- `ncrp_state_totals.csv` — 2024 state-wise NCRP complaint totals, top 5
  states (Maharashtra, UP, Karnataka, Gujarat, Delhi).
- `ncrp_national_totals.csv` — national NCRP incident totals, 2021 / 2024 /
  H1-2025.
- `ncrp_financial_loss.csv` — NCRP/I4C reported financial loss figures
  (total losses, amount under lien, amount returned), plus 2023/2024 annual
  loss totals.
- `bengaluru_karnataka_cybercrime.csv` — Karnataka state and Bengaluru city
  cybercrime case counts, 2022-2024, from NCRB-sourced news coverage. This is
  the most directly relevant calibration series since it matches our chosen
  terminal geography.
- `rbi_atm_counts.csv` — RBI ATM deployment statistics: total ATM counts,
  onsite/offsite split (FY23/FY24), white-label ATM counts, micro-ATM counts,
  ATMs-per-100k-population, bank branch counts. Primary cited source: RBI's
  "Handbook of Statistics on the Indian Economy 2023-24".

**Caveats / minor cross-source variance**: a few figures (e.g. Bengaluru 2023
case count: 17631 vs 17623; white-label ATM 2024 count: 34602 vs 36216
depending on which secondary article is cited) differ slightly between news
outlets reporting on the same underlying official data. Both variants are
noted in the `notes` column rather than silently picking one. These are
secondary-source citation discrepancies, not agent-invented numbers.

## Sources explicitly skipped

- **NCRP official portal (cybercrime.gov.in) dashboards** — no bulk
  download/API found that's accessible without login; skipped rather than
  scraping an unclear-licence dashboard.
- **PIB press release page (pib.gov.in/PressReleaseIframePage.aspx?PRID=2003505)**
  — returned HTTP 403 on fetch attempts; skipped.
- **MHA Lok Sabha PDF (mha.gov.in/.../452.pdf)** — returned HTTP 403 on fetch
  attempts; skipped. (A second MHA PDF at a Punycode-mangled URL also was not
  pursued.)
- **RBI's own ATMView.aspx statistics pages (rbi.org.in/Scripts/ATMView.aspx)**
  — these are RBI's real bank-wise ATM/POS/card statistics pages, but are
  interactive ASP.NET pages, not a flat downloadable table, and were not
  fetched directly within the time budget; the RBI-sourced numbers we did use
  are as reported (with explicit source attribution) by business press
  quoting the RBI Handbook of Statistics, not directly scraped from RBI's
  site. Flagging this so the Model/Simulator agents know these are
  secondhand-cited RBI numbers, not a direct RBI API pull.

## Real-data benchmark note (out of this agent's scope)

Chicago Crimes / AMLworld / Elliptic benchmark datasets mentioned in
PROJECT_SPEC.md are the Benchmark agent's responsibility, not fetched here.
