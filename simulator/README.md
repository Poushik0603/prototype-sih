# Simulator

Seeded, config-driven generator for Cassandra AI's synthetic fraud/mule-network data.
No LLM is used to generate numeric/tabular data. Three configs
(A/B/C) produce genuinely different mule topology + timing for the circularity protocol:
**train on A, test on B and C** — if a model trained on A generalizes to B and C, it has
learned the underlying "money converges at a terminal near complaint time" signal rather
than memorizing config A's specific generator fingerprint.

## Running

```powershell
.venv\Scripts\python.exe simulator\generate.py --config A
.venv\Scripts\python.exe simulator\generate.py --all      # runs A, B, C
```

Outputs go to `data/simulated/config_A/`, `config_B/`, `config_C/`:
- `accounts.parquet`, `complaints.parquet`, `terminals.parquet` (unpartitioned)
- `transactions/date=YYYY-MM-DD/part.parquet` (partitioned by day)
- `run_manifest.json` (seed, calibration hash, rules version, row-count stats)
- best-effort MLflow logging to the repo's local `mlruns/` file store (experiment
  `simulator`), skipped gracefully with a log line if `mlflow` isn't installed

## The three configs

All three share the same terminal universe (`data/simulated/fixture/terminals.parquet`,
real OSM-derived Bengaluru ATM/POS coordinates) and the same label
mechanism: **label(terminal, 2h window) = 1 iff a mule account's final cash-out hop lands
at that terminal in that window, and the ring's complaint was actually filed** (subject to
false-negative noise — see below). What differs is how the money gets there.

| | **A — fast short-chain** (canonical/train) | **B — slow long-chain** (test) | **C — fan-out structuring** (test) |
|---|---|---|---|
| Topology | Narrow: Poisson(lambda=2) hops per branch, 2-5 mule branches/victim | Narrow but deep: Poisson(lambda=4) hops, 3-7 branches | Wide/shallow: Poisson(lambda=1) hop, 6-14 branches |
| Timing between hops | Fast: lognormal(mu=1.5, sigma=0.8) — median ~4.5h | Slow & spread out: lognormal(mu=2.6, sigma=1.3) — median ~13h, heavy tail of days | Fast-ish: lognormal(mu=1.1, sigma=1.0) |
| Cash-out pattern | `single_large_withdrawal` — one big pull | `structured_below_threshold` — multiple sub-threshold withdrawals per mule account over time | `structured_below_threshold` — achieved via many parallel small mule accounts instead |
| Terminal choice | `nearest_to_last_hop` — naive, minimizes mule travel | `hotspot_avoidant` — deliberately avoids terminals with recent heavy mule traffic | `random_in_city` — uniformly random, no spatial structure at all |
| False-negative rate | 5% | 8% (slower rings, victims notice later) | 10% (many small accounts, easier to miss) |
| False-positive rate | 2% | 2% | 4% (wide fan-out sweeps up more innocents) |
| Label delay | uniform(0, 14) days | uniform(0, 21) days | uniform(0, 10) days |
| Time range | 2026-01-05 to 2026-01-26 (22 days) | 2026-01-12 to 2026-02-02 (22 days, offset later) | 2026-01-05 to 2026-01-19 (15 days, offset earlier) |
| Seed | 42 | 43 | 44 |

Ranges overlap but are not identical, so B and C are not just "the same calendar days with
different accounts" — see each config's YAML comments for the full rationale per parameter.

**Why this matters for circularity:** A model that keys off "chain depth" or "delay between
hops" as a feature will do fine on A (its training distribution) but should specifically
struggle on C, whose fraud graphs are shallow/wide rather than deep/narrow. A model that keys
off tight timing windows will struggle on B, whose hops are slow and irregular. Only a model
that has learned the *structural* invariant — several transactions from a complaint-linked
account chain converging at one terminal in a short window — should transfer across all three.

## Label mechanism detail

`complaints.account_id` refers to the **victim** account (per `docs/DATA_SCHEMA.md`: "victim or
reported mule account"). The generator's ground truth linkage (victim's complaint -> mule
chain -> cash-out terminal/window) is available directly in this run as
`run_manifest.json.stats.n_positive_terminal_windows`, but is **not** written out as an
explicit joined label table — that reconstruction (graph traversal from complaint to
transaction chain to terminal/window) happens in the feature-build step (`model/features.py`),
mirroring what a real investigator has to do (they start from a victim's complaint, not from a
labeled mule list). `transactions.is_mule_hop` is full ground truth (simulator-only, not
available to a real-world model) for validating that reconstruction and for the rule-only
baseline.

False-negative noise (`label_noise.false_negative_rate`) is implemented by dropping the
complaint row for a fraction of rings entirely — the mule transactions still exist
(`is_mule_hop=True`) but no complaint links to them, so they correctly produce label=0 unless
another ring's complaint happens to hit the same (terminal, window). False-positive noise adds
extra complaint rows against randomly chosen innocent benign accounts that have no matching
mule transactions, injecting complaint-level noise without fabricating positive windows.

## Terminal-choice-induced time drift (documented deviation)

Config B's long chains + heavy-tailed delay distribution can, in rare cases, push a hop's
timestamp well past the configured `time_range.end`. The generator caps this at `end + 10
days` (clipping, not resampling) so output partitions stay bounded and close to the documented
range — this is why B's actual transaction date partitions run through 2026-02-12 rather than
stopping exactly at 2026-02-02. This is noted here rather than silently tuned away.

## Scale (thin-slice, hackathon demo)

Real terminals table: 1150 terminals (real OSM-derived Bengaluru ATMs/POS, see
`data/calibration/README.md`), at `data/simulated/fixture/terminals.parquet`.
`daily_txn_volume_per_terminal` lambda is set to 8 (vs. an earlier draft's 40) specifically
because of that real terminal count, to keep total benign transaction volume in the low
hundreds of thousands rather than 1M+ per config — still far more than enough signal for the
feature-build step, while keeping generation runtime and file size thin.

## Row counts and label balance (actual run outputs)

| Config | Accounts | Mule accounts | Transactions | Mule-hop txns | Complaints | Days |
|---|---|---|---|---|---|---|
| A (fast short-chain) | 2,101 | 1,061 | 194,072 | 1,061 | 136 | 21 |
| B (slow long-chain) | 3,187 | 2,177 | 196,452 | 3,241 | 107 | 32* |
| C (fan-out structuring) | 2,254 | 1,264 | 131,948 | 3,096 | 84 | 14 |

\* B's configured range is 22 days (2026-01-12 to 2026-02-02); actual partitions run to
2026-02-12 because long/slow mule chains are clipped at `end + 10 days` rather than resampled
(see "terminal-choice-induced time drift" above).

Label balance, computed against the full (terminal, 2h-window) population implied by each
config's terminal count and day span:

| Config | label=1 windows | approx. total (terminal, window) population | positive rate |
|---|---|---|---|
| A | 464 | ~289,800 | 0.160% |
| B | 900 | ~441,600 | 0.204% |
| C | 1,464 | ~193,200 | 0.758% |

This is intentionally a rare-event problem, not artificially balanced — each config gives
several hundred to ~1.5K positive examples against a much larger negative population, and the
positive rate differs meaningfully across configs (C's wide fan-out produces proportionally far
more positive windows than A's narrow chains), as expected for the circularity protocol.

## Reproducibility

Verified bit-for-bit: running `generate()` twice with the same seed (via `random.seed` +
`np.random.seed`, plus `Faker.seed`) produces identical accounts/transactions/complaints
DataFrames. Each run's config file hash, terminals calibration hash, seed, and rules_version
are logged to `data/simulated/config_<X>/run_manifest.json` and (best-effort) to MLflow's
local file store under experiment `simulator`.

## Config loader

`simulator/lib/config.py` (`load_config`) parses a `rules.yaml`-shaped file into a
`SimConfig` dataclass with typed accessors (`.mule_network`, `.benign_background`,
`.label_noise`, `.time_range`, `.scale`, `.rules_version`). `simulator/lib/dist.py` samples
the three distribution shapes used throughout (`poisson`, `lognormal`, `uniform`) from a
single seeded `numpy.random.Generator` for full reproducibility.
