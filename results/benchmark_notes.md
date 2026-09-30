# Real-Data Benchmark Notes (Benchmark Agent)

Per `PROJECT_SPEC.md` → "Real-data benchmarks" and "Reporting rules": every number below is
explicitly labeled as coming from **real, publicly obtainable data** (never simulated), kept
separate from the simulator/circularity-protocol results, and only reported when produced by
an actual run saved to disk. Raw downloaded data lives under `data/raw/` with SHA256 checksums.
Code lives under `model/benchmarks/`.

---

## 1. Graph/linkage module — AMLworld / Elliptic — SKIPPED (documented, not faked)

**Status: skipped cleanly. No substitute dataset was used.**

### What was tried

| Candidate source | Result |
|---|---|
| Elliptic Data Set, official (`kaggle.com/datasets/ellipticco/elliptic-data-set`) | Requires Kaggle login; no `kaggle` API credentials configured in this non-interactive environment (`~/.kaggle` absent, `kaggle` package not installed). Kaggle dataset pages also do not render licence text without JS/auth, so the licence itself could not even be confirmed programmatically. |
| Elliptic mirror on Hugging Face (`yhoma/elliptic-bitcoin-dataset`) | Third-party re-upload. The uploader's own dataset card self-declares `license: mit`, but this is **not** the original Elliptic licence — Elliptic's actual terms (via Elliptic Co. / Kaggle) are more restrictive than a re-uploader's own MIT tag, and there is no verifiable chain of custody proving the re-uploader had rights to relicense it. Treated as an unclear/unverifiable licence per the task's explicit instruction → **not used**. |
| IBM AMLworld, official repo (`github.com/IBM/AML-Data`) | This is IBM's own repo and clearly documents licence: code = Apache-2.0, **data = CDLA-Sharing-1.0** (clear, documented licence). However the repo does not host the CSVs itself — it only links out to (a) Kaggle (same auth wall as above) or (b) an IBM Box folder. |
| IBM Box folder (`ibm.box.com/v/AML-Anti-Money-Laundering-Data`) | Redirects to `ibm.ent.box.com/...`, which requires JS rendering / returned a connection reset on non-interactive fetch (`curl`), consistent with an interactive-login-gated Box share. Could not be completed non-interactively. |
| AMLworld mirror on Hugging Face (`bbfizp/AMLSim-HI-Small`) | Same problem as the Elliptic HF mirror: unverified third-party re-upload, no traceable relicensing rights back to IBM's CDLA-Sharing-1.0 terms. Also 476MB, oversized for a "small sanity check". **Not used.** |

### Why we stopped here

Both datasets named in the task have exactly one authoritative distribution channel apiece
(Kaggle for Elliptic; Kaggle or an IBM-controlled Box folder for AMLworld), and both are
gated behind interactive auth/JS that this non-interactive agent session cannot complete. The
only channels reachable without auth were unofficial third-party re-uploads whose licensing
claims cannot be verified against the originals. Per the task instructions ("If both require
login/auth you can't complete non-interactively, or licensing terms are unclear/restrictive,
DO NOT use them... Do not fabricate a substitute dataset pretending it's AMLworld/Elliptic"),
this part of the benchmark is **skipped, not faked**. No graph/linkage numbers are reported
against Elliptic or AMLworld.

**If Kaggle credentials become available later** (a `kaggle.json` API token placed at
`~/.kaggle/kaggle.json`), re-run is straightforward: `kaggle datasets download
ellipticco/elliptic-data-set` or `kaggle datasets download
ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, then build a NetworkX graph from
the edgelist and test whether node degree / ego-network size separates illicit vs. licit
labelled nodes (AUC) better than chance — the same honest-sanity-check design used for the
ST-DBSCAN check below. This script is not written yet since there is no data to run it on.

---

## 2. ST-DBSCAN module — Chicago Crimes (Socrata API) — COMPLETED, real data

**Status: completed successfully on real, public data.**

### Data

- Source: City of Chicago Data Portal, "Crimes - 2001 to present" (`ijzp-q8t2`), Socrata Open
  Data API. Public government dataset, no authentication required, free reuse
  (Socrata Open Data Terms of Use).
- Query: last 120 days ending 2026-09-29, rows with non-null lat/lon, `$limit=15000`.
- Raw file: `data/raw/chicago_crimes_sample_2026.json` (15,000 rows, 4,488,282 bytes)
- Checksum manifest: `data/raw/chicago_crimes_sample_2026.manifest.json`
  - **SHA256: `ff27c254ccd6f8d4a2f3d3467d4a121bbea56d2332d908314af43ea692223a31`**
- Fetch script: `model/benchmarks/fetch_chicago_crimes.py`

### Method

`model/benchmarks/stdbscan_chicago.py` — scikit-learn `DBSCAN` over a rescaled 3D feature
space `(x_km/eps_spatial, y_km/eps_spatial, t_hours/eps_temporal)` with `eps=1.0`, which
approximates an ST-DBSCAN space-AND-time neighborhood (a standard, lightweight way to combine
spatial and temporal closeness in one DBSCAN call, as suggested in the task). Parameters
chosen to be roughly comparable to the project's terminal/2-hour-window unit:

- `spatial_eps_km = 0.4` (~400m)
- `temporal_eps_hours = 6.0`
- `min_samples = 5`

### Results (REAL DATA — Chicago Crimes, not simulated)

Saved to `results/benchmark_stdbscan_chicago.json`.

| Metric | Value |
|---|---|
| Input points | 15,000 real crime incidents |
| Date range | 2026-06-02 to 2026-09-29 (~120 days) |
| Clusters found | 84 |
| Clustered points | 689 (4.59%) |
| Noise points | 14,311 (95.41%) |
| Cluster size: min / median / mean / max | 3 / 6 / 8.2 / 29 |

Example hotspots (top by size):

| Cluster | Size | Centroid (lat, lon) | Time span | Dominant crime types |
|---|---|---|---|---|
| 29 | 29 | 41.8901, -87.6295 | 2026-09-12 15:30 → 09-13 07:21 (~16h) | THEFT (7), BURGLARY (3), BATTERY (3) |
| 45 | 26 | 41.8054, -87.5855 | 2026-09-08 23:00 → 09-09 10:53 (~12h) | CRIMINAL DAMAGE (22 of 26) — a parking lot/garage |
| 14 | 22 | 41.8844, -87.6261 | 2026-09-16 08:15 → 09-17 00:06 (~16h) | THEFT (9), BATTERY (4) — street |
| 44 | 22 | 41.8852, -87.6264 | 2026-09-09 10:30 → 09-09 22:47 (~12h) | THEFT (13), BATTERY (3) — sidewalk |

### Takeaway

On real Chicago crime data, the space+time-rescaled DBSCAN approach produces a small number
(84) of tight, interpretable clusters — consistent with real spatio-temporal hotspots (e.g. a
single-location, single-crime-type burst like the 22/26-incident "CRIMINAL DAMAGE" cluster at
one parking lot within a 12-hour window) rather than noise artifacts. Most points (95%) remain
unclustered, which is expected and sensible for citywide crime data that is spatially diffuse
outside of genuine hotspots. This is a good sign that the same clustering machinery the Model
agent will use for candidate-terminal detection behaves reasonably on a real point process, not
just the simulator's synthetic mule bursts — though this check only validates cluster
*formation behavior*, not fraud-specific precision, since Chicago crime data has no ground-truth
"mule terminal" labels to score against.

---

## Overall honest summary

- **ST-DBSCAN module**: validated against real data (Chicago Crimes, Socrata API) — produces
  sensible, interpretable spatio-temporal clusters. Confirms the clustering mechanism itself
  is sound on a real point process; does not (and cannot, absent labels) confirm fraud
  detection precision on this dataset.
- **Graph/linkage module**: **not validated against real data** — both Elliptic and AMLworld
  were blocked by non-interactive auth walls on their only authoritative distribution
  channels, and no unverified substitute was used, per the task's explicit instruction not to
  fabricate a substitute. This is an open gap; flagged honestly rather than papered over.
  A path to close it is documented above if Kaggle credentials become available before the
  deadline.
