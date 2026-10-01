# Verification notes — leakage audit and metric recomputation

A second, independent pass over this prototype's pipeline, done without reusing any
of the evaluation code (`evaluate.py`/`baselines.py`) — every "recomputed" number
below comes from a standalone script that loads `data/processed/features_config_*.parquet`
and `model/artifacts/xgb_config_A.json` directly and recomputes the metric from scratch.

## Summary verdict

The reported results hold up. Every headline number in `results/metrics_config_A.json`,
`_B.json`, `_C.json` was independently recomputed from the saved parquet feature
tables and the saved XGBoost model artifact, and matched the saved JSON **exactly**
(to the last printed digit) in all three configs. The leakage discipline described in
`model/features.py`'s docstrings is real, not just asserted: the rolling-origin split
has a genuine >=7-day gap with zero overlap, the historical "hotspot frequency" feature
correctly excludes the current window's own label via `.shift(1)` before
`.expanding()`, and the as-of graph-distance and complaint-density features both use
strict `<` timestamp comparisons sourced from a cumulative, time-ordered structure.
The claim that the model does not beat the baseline anywhere, with ROC AUC in the
0.39-0.58 range, is **verified independently and is accurate** — this is not an
underclaim or overclaim, it is a correctly and honestly reported negative result (and
is now additionally confirmed stable across 5 random seeds — see
`results/metrics_multiseed.json`). The circularity protocol is substantially complete
(4.5 of 5 steps solid), the real-data benchmark section is a genuine, specific account
of blocked channels (not a vague excuse), and the two screenshots are real,
appropriately-sized renders.

One finding from the first verification pass has since been fixed: the two
screenshots were originally captured against mock data before the real model
finished training, so the SHAP feature names shown didn't match the real model's
actual features. Both screenshots have since been re-captured against the live,
trained model (`USE_MOCK=false`) after also fixing a real performance bug uncovered
in the process — the API's real-data path was scoring the entire ~290k-row feature
history through SHAP on every request instead of just the current window, which made
it take minutes per call. See `api/README.md` for the fix.

## Leakage findings

| # | Area | File:line | Finding | Severity |
|---|---|---|---|---|
| 1 | Past-hotspot feature (classic self-referential leak pattern) | `model/features.py:193-205` (`compute_past_hotspot_freq`) | Verified by direct unit test: `s.shift(1).expanding().mean()` excludes the current row's own label before averaging. Confirmed on a synthetic 5-row sequence that row *i*'s computed value only reflects rows `0..i-1`. **No leakage.** | None found |
| 2 | Rolling-origin split embargo | `model/features.py:391-430` (`make_rolling_origin_splits`) | Verified directly on `features_config_A.parquet`: train ends 2026-01-09 02:00, val starts 2026-01-16 04:00 (gap = 7d02h); val ends 2026-01-17 04:00, test starts 2026-01-24 06:00 (gap = 7d02h). Both gaps are >=7 days, rows in between are tagged `embargo` and carry no train/val/test tag, confirmed by direct parquet inspection, not just by reading the code. **No leakage, no overlap.** | None found |
| 3 | As-of rolling txn count/amount features | `model/features.py:147-190` | Uses `np.searchsorted(ts, ws, side="left")` for the window end, i.e. transactions with `timestamp < window_start` only. Correct. | None found |
| 4 | Complaint density (spatial) feature | `model/features.py:208-273` | Uses `max(filed_at, first_seen_near_terminal)` compared via `< window_start`; both components are as-of (complaints filed before `ws`, and the complainant's account first seen transacting near this terminal before `ws`). Logic traced manually and found correct — no feature can reflect activity that happens in or after the window being scored. | None found |
| 5 | Graph-distance-to-known-mule feature | `model/features.py:276-388` | Cumulative graph only adds edges for transactions strictly before `ws` (`np.searchsorted(ts_arr, ws_ts, side="left")`); known-mule set is complaints filed strictly before `ws`. This was rewritten mid-project specifically to fix a leakage *risk* in an earlier fixed-lookback version per the code comments — the current form is correct and was spot-checked by reading the cursoring logic end-to-end. | None found |
| 6 | Label definition itself | `model/features.py:19-29, 127-144` | Label is built from `complaints.account_id` (what a bank would actually learn via a complaint), not the simulator's internal `is_mule` ground-truth flag — this is explicitly called out as the leakage-avoidance decision in the docstring and matches what the code does. Good practice, correctly implemented. | None found |
| 7 | Hyperparameter tuning only on val | `model/train.py:72-104` | `tune_on_val` only reads `df[df.split_tag=="val"]`; test split is never touched during training or tuning. `evaluate.py` loads the test split only once, after the model is already persisted to disk, and nothing in the codebase re-trains after seeing test numbers. | None found |
| 8 | Cross-config ("circularity") test construction | `model/evaluate.py:100-117` | For config A, only the `test` split tag is used (never train/val). For configs B/C, the **entire** feature table is used as test, which is legitimate (and explicitly justified in comments) because the model was never trained or tuned on *any* row of B or C — there is no risk of this being train-on-test, it's a true out-of-distribution check. One caveat: this means B/C's own internal train/val/embargo/test split tags are irrelevant to this comparison, which is correct but worth being explicit about (their `split_tag` column reflects *their own* would-be rolling-origin split, not anything to do with the A->B/A->C evaluation). | Minor (documentation clarity only, not a bug) |
| 9 | Config C has zero rows under its own internal `test` tag | `data/processed/features_config_C.parquet` (verified directly: `split_tag` value_counts show only `embargo`, `train`, `val`, no `test`) | This is a genuine consequence of C's 13-day span being too short to carve out train+2x embargo+val+test, and is openly documented in `features.py`'s own docstring and `model/README.md`. Not a bug, not hidden — but it means the "beats_baseline" result for config C in `metrics_config_C.json` is **entirely a cross-config check** (evaluated on C's whole table, not a true C-internal holdout), consistent with point 8 above. Worth being precise about this distinction in a deck if config C is cited. | Minor (disclosure quality) |

**No blocking leakage issues were found.** This is a cleaner-than-typical leakage
discipline for a thin-slice prototype — the as-of discipline is enforced at the
mechanical level (searchsorted/shift on sorted arrays), not just asserted in comments.

## Metric recomputation

All three configs were recomputed from `data/processed/features_config_{A,B,C}.parquet`
+ `model/artifacts/xgb_config_A.json` directly, with a standalone script, not by
reusing `evaluate.py`/`baselines.py`. Every value below matched the saved JSON exactly.

| Config | Metric | Reported | Independently recomputed | Match |
|---|---|---|---|---|
| A | n_test_rows / n_positive | 24,150 / 48 | 24,150 / 48 | Yes |
| A | ROC AUC — model | 0.52425 | 0.52425 | Yes |
| A | ROC AUC — baseline (past-hotspot) | 0.48612 | 0.48612 | Yes |
| A | Precision@10/window (mean) — model | 0.000 | 0.000 | Yes |
| A | Precision@10/window (mean) — baseline | 0.000 | 0.000 | Yes |
| A | beats_baseline | false | false (0.000 is not > 0.000) | Yes |
| B | n_test_rows / n_positive | 430,100 / 443 | 430,100 / 443 | Yes |
| B | ROC AUC — model | 0.39063 | 0.39063 | Yes |
| B | ROC AUC — baseline | 0.49169 | 0.49169 | Yes |
| B | Precision@10/window — model | 0.0002674 | 0.0002674 | Yes |
| B | Precision@10/window — baseline | 0.0005348 | 0.0005348 | Yes |
| B | beats_baseline | false | false | Yes |
| C | n_test_rows / n_positive | 193,200 / 537 | 193,200 / 537 | Yes |
| C | ROC AUC — model | 0.50311 | 0.50311 | Yes |
| C | ROC AUC — baseline | 0.50972 | 0.50972 | Yes |
| C | Precision@10/window — model | 0.002381 | 0.002381 | Yes |
| C | Precision@10/window — baseline | 0.003571 | 0.003571 | Yes |
| C | beats_baseline | false | false | Yes |

**No discrepancies found anywhere.** The claim that the model does not beat the
baseline anywhere (ROC AUC 0.39-0.58, near chance, occasionally worse than chance on
B) is **verified independently and is accurate, not an underclaim or exaggeration**.
`metrics_baselines.json` was also checked and is a pure re-serialization of the
already-verified per-config files (no new computation, matches source files
field-for-field).

Also verified: the Chicago Crimes real-data benchmark's SHA256 checksum
(`ff27c254ccd6f8d4a2f3d3467d4a121bbea56d2332d908314af43ea692223a31`) was recomputed
directly from `data/raw/chicago_crimes_sample_2026.json` on disk and matches exactly
what's recorded in `data/raw/chicago_crimes_sample_2026.manifest.json` and quoted in
`results/benchmark_notes.md`.

A second verification pass later confirmed, via 5-seed re-training
(`results/metrics_multiseed.json`), that seed variance is small relative to how far
every value sits from "beats baseline" — ROC AUC stays in the 0.39-0.58 near-chance
band across all 5 seeds, so the original single-seed finding is not a lucky/unlucky
seed artifact.

## Overclaims found and resolved

One issue was found and has since been fixed, not just flagged:

1. **Screenshot SHAP feature names didn't match the real model's feature columns.**
   `results/screenshots/officer_alert_view.png` originally showed "Top contributing
   factors (SHAP)" as `past_hotspot_freq`, `mule_hop_count_7d`, `avg_txn_amount`,
   `time_since_last_comp[laint]` — but the real trained model's `feature_cols` (per
   `model/artifacts/xgb_config_A_meta.json`) are `txn_count_2h`, `txn_amount_sum_2h`,
   `txn_count_6h`, `txn_amount_sum_6h`, `txn_count_24h`, `txn_amount_sum_24h`,
   `txn_count_168h`, `txn_amount_sum_168h`, `past_hotspot_freq`,
   `complaint_density_kring`, `graph_dist_to_known_mule`. This happened because the
   screenshots were captured while the API was running on `mock_data.py`, before the
   real model artifacts existed — an honest, documented reason at the time, not a
   secret fabrication, but it would have become a real overclaim if presented in a
   deck captioned as "the model's actual output." **Fixed**: both screenshots have
   been re-captured against the live, trained model, after also fixing a SHAP
   performance bug found in the same pass (see `api/README.md`).

Everything else checked out clean:
- Every specific numeric claim across the module READMEs and `results/benchmark_notes.md`
  was checked against a verifiable artifact on disk (parquet row counts, JSON metric
  files, manifest checksums, config YAML diffs) and all were accurate.
- The claim that Elliptic/AMLworld were skipped due to licensing is a specific,
  falsifiable account — named URLs (Kaggle, IBM's GitHub repo, IBM Box, two named
  Hugging Face mirrors), named failure modes (auth wall, JS-gated redirect,
  unverifiable third-party relicensing), not a vague "couldn't find data" excuse.
  This reads as a genuine attempt, not a fabricated excuse.
- Both screenshots are real, substantial PNGs (1.08MB and 1.37MB originally; both
  re-captured since), confirmed non-trivial by direct image inspection (full
  MapLibre-rendered Bengaluru map with OSM tile labels, markers, legend, and a
  populated detail panel) — not blank or placeholder images.
- The 50-value `all_latencies_ms` array in `results/api_latency.json` is internally
  consistent with its own reported p50/p95/mean (independently recomputable from the
  array itself).

## Circularity protocol completeness

1. **Train A / test B,C with genuinely different topology** — **Solid.** Diffed
   `simulator/configs/A.yaml` vs `B.yaml` vs `C.yaml` directly: these differ in
   `hop_count_distribution` (Poisson lambda=2 vs 4 vs 1), `delay_between_hops_hours`
   (lognormal params differ materially), `cash_out_pattern` (single large withdrawal
   vs structured-below-threshold via repeated withdrawal vs structured-below-threshold
   via wide fan-out), and critically `terminal_choice` (`nearest_to_last_hop` vs
   `hotspot_avoidant` vs `random_in_city` — three genuinely different spatial
   heuristics), plus different seeds (42/43/44) and different time ranges. This is a
   real topology/timing change, not a renamed seed.
2. **Rule-only baseline present** — **Solid.** `model/baselines.py:rule_only_baseline`
   implements per-config-specific logic matching each config's own `terminal_choice`
   rule, present in all three `metrics_config_*.json` files.
3. **Label noise / delayed labels tested with a reported number** — **Partial.** Noise
   rates differ per config (A: 5%/2% FN/FP; B: 8%/2%; C: 10%/4%) and this is baked into
   the generated data (not a separately re-runnable ablation), and the degradation from
   A->B->C is reported as the robustness proxy. This is honestly flagged as conflating
   topology/timing shift with noise-rate shift in `model/circularity.py`'s own
   docstring and `metrics_circularity.json`'s step3 note — a real limitation, clearly
   disclosed rather than glossed over, but it does mean there's no isolated "same
   topology, only noise changed" ablation number.
4. **Real-data benchmarks reported separately** — **Solid.** Confirmed
   `results/benchmark_stdbscan_chicago.json` carries `"is_real_data": true` and a
   `label_note` stating it's real data; it is never referenced from or merged into
   any `metrics_config_*.json`/`metrics_circularity.json`/`metrics_baselines.json`
   file (checked all four simulated-metrics files for cross-references — none found).
5. **Realism gap stated plainly** — **Solid.** `model/circularity.py`'s
   `REALISM_GAP_STATEMENT` (echoed into `results/metrics_circularity.json`) is a
   substantive, specific statement — it explicitly names the mechanism by which
   simulated performance is inflated (model and simulator draw from the same
   heuristic family) rather than a generic disclaimer.

**4 of 5 steps fully solid, 1 (label noise) honestly partial with the limitation
clearly disclosed.**

## Reproducibility spot-check

Checked `simulator/lib/generate_core.py:generate()`. Seeding is complete, not
partial: `rng = np.random.default_rng(seed)` (numpy sampling), `py_rng =
random.Random(seed)` (stdlib random — used for `.choice`/`.sample`/`.randint`
throughout, confirmed by grep showing ~20 call sites all using `py_rng.*`, not the
unseeded global `random` module), and `Faker.seed(seed)` (synthetic name/text
generation). `simulator/generate.py` additionally sets global `random.seed(cfg.seed)`
and `np.random.seed(cfg.seed)` as a belt-and-suspenders measure. Using locally-scoped
RNG instances (`default_rng`/`Random(seed)`) rather than relying purely on global
state is actually a *more* robust reproducibility pattern than global-only seeding,
since it isolates this generator's randomness from any other code that might also
consume the global RNG. The claim of verified bit-for-bit reproducibility across two
runs with the same seed is plausible given this seeding structure, and is also
consistent with the multi-seed evaluation producing clean, non-degenerate results per
seed.

## Recommendation

**Nothing here blocks presenting these results as "results."** The headline finding
— the model does not beat either baseline, with near-chance ROC AUC, stable across
seeds — is real, independently verified, and reported honestly rather than hidden.
This is a thin-slice prototype with an honest negative result and a working,
leak-free evaluation harness, which is itself a legitimate deliverable for a project
judged on rigor.

**Already fixed**: screenshots re-captured against the real model (previously flagged
as a risk if presented without context — resolved, not just noted).

**Acceptable to present as-is**: the near-chance model performance (already labeled
honestly), the config-C zero-internal-test-split limitation (already documented),
and the conflated noise/topology ablation in step 3 (already disclosed as a known
limitation) — none of these need further work before a deck, they need only to be
stated with the same honesty they already carry in the underlying JSON/markdown
files. The realism-gap statement in `metrics_circularity.json` is strong enough to
quote directly in a deck's limitations slide.
