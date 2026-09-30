# Data Agent Report

## City chosen: Bengaluru, Karnataka
Dense OSM ATM tagging (query returned 1150 nodes, well above the "few hundred" bar)
and it is also the metro with the highest published cybercrime case rate in NCRB
data — a defensible, real-world-plausible choice for a fraud/mule-network
prototype, not just a data-availability pick. Rationale documented in
`data/calibration/README.md`.

## Terminals
- Fetched 1150 real `amenity=atm` nodes from OpenStreetMap via Overpass API
  (bbox `12.83,77.45,13.14,77.78`, covers greater Bengaluru). Raw response frozen
  unmodified at `data/raw/osm_atms_bengaluru.json` (SHA256 in
  `data/calibration/manifest.json`; file is gitignored per existing repo
  convention — `data/raw/*` and `*.parquet` are excluded from git, reproducible
  via script instead).
- Built `terminals` table (1150 rows) exactly per SCHEMA_CONTRACT.md:
  `terminal_id, lat, lon, h3_cell (H3 res 9, h3 v4 `latlng_to_cell`), type, bank,
  install_date`. Only `bank` and `install_date` are synthetic (seeded, sampled
  from 6 real Indian bank names / random date in past 1-5 years) — everything
  else is real OSM geography. Disclosed clearly in README.
- Saved to `data/simulated/fixture/terminals.parquet` (doubles as the
  SCHEMA_CONTRACT fixture — used the full 1150 rather than subsampling to 50,
  since the full set is already small) and `data/raw/terminals_full.parquet`.
- Reproduction scripts: `scripts/fetch_calibration.py` (idempotent OSM fetch,
  retry/backoff — first attempt hit a transient disconnect, second succeeded)
  and `scripts/build_terminals.py`.

## Calibration numbers
No cleanly downloadable/open-licence NCRP or RBI dataset was found in the time
available (NCRP portal has no public bulk export; RBI publishes as PDF tables).
Per task instructions, hand-transcribed real numbers from cited public sources
into `data/calibration/*.csv`:
- `ncrp_state_totals.csv` — 2024 state-wise NCRP complaints, top 5 states.
- `ncrp_national_totals.csv` — national NCRP totals 2021/2024/H1-2025.
- `ncrp_financial_loss.csv` — NCRP/I4C reported fraud loss figures.
- `bengaluru_karnataka_cybercrime.csv` — Karnataka + Bengaluru city case counts
  2022-2024 (NCRB-sourced, most directly relevant to our chosen geography).
- `rbi_atm_counts.csv` — RBI ATM/onsite-offsite/white-label/micro-ATM counts,
  primary cited source RBI's "Handbook of Statistics on the Indian Economy
  2023-24".

Every figure has a source URL + access date in `manifest.json`. All were
fetched via WebSearch/WebFetch from real citable pages — none invented.
**Skipped**: PIB press release page and MHA Lok Sabha PDF (both HTTP 403),
NCRP's own portal dashboards (no accessible bulk export), and RBI's
ATMView.aspx pages (interactive, not flat-downloadable) — logged in README
rather than guessed around.

## Deviations from spec
- Used a bbox Overpass query instead of an OSM admin-boundary `area` query,
  for reliability. Documented.
- Did not add `shop`/POS nodes — 1150 ATM points already exceeded the minimum,
  kept scope thin per spec's own guidance. All terminals have `type="ATM"`;
  Model/Simulator agents should know there are currently no `"POS"` rows in
  this table if they test that branch of logic.
- `install_date` written as `datetime64` (parquet has no separate pure-date
  physical type); values are midnight-UTC dates, functionally equivalent to
  `date`.

## Note for other agents
**Git housekeeping note**: mid-task, a `git commit` I ran picked up files from
the Benchmark agent that were concurrently staged in the shared index
(`model/benchmarks/*.py`, `results/benchmark_*`, `requirements.txt` diff) —
all files are intact on disk and correctly committed, just bundled into my
commit rather than the Benchmark agent's own. No data was lost or overwritten;
flagging so nobody is confused by the commit contents not matching its message
exactly.

Terminals table location for Simulator/Model agents:
`data/simulated/fixture/terminals.parquet` — 1150 rows, matches
SCHEMA_CONTRACT.md exactly, ready to build against now.
