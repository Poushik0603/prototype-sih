# Simulator Agent — Report

## Terminals table used
**Real data.** The Data agent's `data/simulated/fixture/terminals.parquet` (1150 real
OSM-derived ATM/POS terminals in Bengaluru) landed partway through this run. I started
development against an auto-generated placeholder (synthetic points over a Bengaluru bounding
box, same schema) so I didn't block, then re-ran the final generation against the real file —
all three config outputs below use it (`run_manifest.json.terminals_source == "real"` for
A/B/C). The placeholder generator (`simulator/lib/terminals.py::generate_placeholder_terminals`)
is kept in the code as an automatic fallback for anyone who runs the generator before/without
the real fixture — no action needed from Model/API agents since real data was used for the
actual outputs.

## Row counts per config
| Config | Accounts | Mule accounts | Transactions | Mule-hop txns | Complaints | Days |
|---|---|---|---|---|---|---|
| A (fast short-chain) | 2,101 | 1,061 | 194,072 | 1,061 | 136 | 21 |
| B (slow long-chain) | 3,187 | 2,177 | 196,452 | 3,241 | 107 | 32* |
| C (fan-out structuring) | 2,254 | 1,264 | 131,948 | 3,096 | 84 | 14 |

\* B's configured range is 22 days (2026-01-12 to 2026-02-02); actual partitions run to
2026-02-12 because long/slow mule chains are clipped at `end + 10 days` rather than resampled
— documented in `simulator/README.md`.

## Label balance (terminal, 2h-window)
Using the generator's own ground-truth linkage (complaint -> mule chain -> cash-out
terminal/window), computed against the full (terminal x 2h-window) population implied by each
config's terminal count and day span:

| Config | label=1 windows | approx. total (terminal,window) population | positive rate |
|---|---|---|---|
| A | 464 | ~289,800 | 0.160% |
| B | 900 | ~441,600 | 0.204% |
| C | 1,464 | ~193,200 | 0.758% |

This is intentionally a rare-event problem (not balanced), per spec — each config gives the
Model agent several hundred to ~1.5K positive examples against a much larger negative
population, and the positive rate differs meaningfully across configs (C's wide fan-out
produces proportionally far more positive windows than A's narrow chains), which is expected
and desired for the circularity protocol.

## Deviations from spec / notable decisions
1. **`daily_txn_volume_per_terminal` lambda reduced from template's 40 to 8.** With the real
   terminal count (1150, higher than the "50-300" thin-slice guidance — but it's the real Data
   agent output, so I used it as-is), lambda=40 over a 3-week window produced ~1M+ benign rows
   per config. Reduced to keep runtime/volume thin per hackathon-scope guidance while still
   giving ~130-200K transactions per config.
2. **Time-drift clipping in config B.** Long chains x heavy-tailed hop delays could push
   timestamps arbitrarily far past `time_range.end` in rare cases. Fixed by (a) biasing
   incident start times into the first ~70% of the range and (b) hard-clipping any hop
   timestamp at `end + 10 days`. Documented in `simulator/README.md`; B's actual output spans
   10 days past its configured end as a result.
3. **`complaints.account_id` is the victim, not the mule.** Matches SCHEMA_CONTRACT.md's own
   note ("victim or reported mule account"). The simulator does NOT emit a pre-joined
   complaint-to-terminal label table — that graph reconstruction (victim complaint -> chain ->
   terminal/window) is left as the Model agent's feature-build step, mirroring the real
   investigative task. Full ground truth (`transactions.is_mule_hop`) is available for
   validating that reconstruction and for the rule-only baseline.
4. **MLflow compatibility.** The installed MLflow (3.16.1) refuses to write to a plain
   filesystem store by default ("maintenance mode") unless `MLFLOW_ALLOW_FILE_STORE=true` is
   set — set this in `generate.py` before import, per PROJECT_SPEC.md's local-file-store
   requirement. Logging is wrapped in try/except so a missing or misbehaving mlflow never
   fails the run; the JSON manifest is always written regardless.
5. **Free-text complaint narratives skipped** (explicitly out of scope per task instructions).
6. Splink/NetworkX graph linkage, feature table, ST-DBSCAN, etc. are explicitly out of scope
   for this agent (Model agent's responsibility) — not attempted here.

## Reproducibility
Verified bit-for-bit: running `generate()` twice with the same seed (via `random.seed` +
`np.random.seed`, plus `Faker.seed`) produces identical accounts/transactions/complaints
DataFrames. Each run's config file hash, terminals calibration hash, seed, and rules_version
are logged to `data/simulated/config_<X>/run_manifest.json` and (best-effort) to MLflow's
local file store under experiment `simulator`.
