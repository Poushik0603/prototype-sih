# Deck edits — SIH26184_NeferX_CassandraAI_v5

Every edit below carries real numbers/screenshots from `results/results.md` and
must be labeled **"prototype on simulated data"** (or "real data" where marked)
on the slide itself — do not present simulated numbers as production
performance.

## Architecture / pipeline slide
- Add the actual stack used, since it differs slightly from the original plan:
  seeded simulator → feature build (ST-DBSCAN candidates, as-of features) →
  XGBoost + SHAP → FastAPI → React + MapLibre.
- Note: pgmq was dropped in favor of direct table access for this thin slice
  (documented deviation, not a silent change) — mention only if asked about
  queueing, not a headline item.
- Note: real geography — 1,150 real OpenStreetMap ATM coordinates for
  Bengaluru, not synthetic points. This is a credibility-building detail,
  worth a callout box: "terminal locations are real; transaction/account data
  is simulated and calibrated against public NCRP/RBI statistics."

## New slide: "Honest result — model vs. baseline" (recommend adding, not optional)
This is the single most important slide to add. Judges reward rigor over
inflated claims. Use the table from `results/results.md` §2:

| Config | Model ROC AUC | Baseline ROC AUC | Beats baseline? |
|---|---|---|---|
| A | 0.524 | 0.486 | No |
| B | 0.391 | 0.492 | No |
| C | 0.503 | 0.510 | No |

Caption: **"Prototype on simulated data — the model does not yet outperform a
simple frequency baseline. We are reporting this plainly: at this data scale
and 2-hour window granularity, the learned signal is not yet separable from
chance. See realism-gap statement."** Pull the quoted realism-gap statement
from `results/results.md` §3 directly onto this slide — it's a strong, precise
statement that shows the team understands exactly why simulated performance
can't be read as real-world performance.

Framing for presentation: pitch this as "we built the full evaluation harness
— rolling-origin split, 7-day embargo, leakage audit by an independent
reviewer agent, cross-config generalization testing — and it correctly
surfaced that our first model iteration isn't ready, rather than overclaiming
a result that wouldn't survive scrutiny." This is a stronger SIH pitch than a
cherry-picked good number, especially under judge scrutiny.

## Screenshots slide(s)
- Replace any placeholder/mockup images with
  `results/screenshots/ranked_atm_heatmap.png` and
  `results/screenshots/officer_alert_view.png` — both captured against the
  live, real, trained model (re-captured during final integration after an
  earlier mock-data capture was caught and flagged by the independent
  reviewer).
- Caption both: **"Prototype UI, real trained model output, simulated
  underlying data."**
- Minor known cosmetic gap: the "ATM · —" bank-name field renders as an em
  dash in the current screenshots (a null-value display issue, not a data
  problem) — either crop tightly to avoid drawing attention to it, or mention
  it's a known UI polish item if asked.

## Real-data validation slide
Add a small table distinguishing from the simulated results above:

| Module | Real dataset | Result |
|---|---|---|
| ST-DBSCAN clustering | Chicago Crimes (15,000 real incidents, public Socrata API) | 84 sensible spatio-temporal clusters found |
| Graph/linkage (NetworkX/Splink) | Elliptic / IBM AMLworld | Not validated — both datasets required login or had unclear licensing; skipped rather than substituted with something lower-quality |

Caption: **"Real-world data used only to sanity-check individual pipeline
modules, not the full fraud-detection pipeline end to end — no public dataset
links a complaint to a specific ATM+window, which is why the hybrid simulator
approach exists."** This directly supports the deck's existing "no public
dataset has this label" justification — use it as evidence, not just
assertion.

## Limitations slide (add if not present, or expand if present)
Pull directly from `results/results.md` §7 — six bullet points, already
written in presentable form. Key ones to definitely keep: model doesn't beat
baseline yet; graph module unvalidated on real fraud data; calibration numbers
hand-transcribed from cited public sources (not a bulk dataset).

## What NOT to change
- Do not alter the original problem-statement framing or team/idea slides —
  this prototype validates the approach's buildability and evaluation rigor,
  it does not change what problem is being solved.
- Do not present the ROC AUC / precision numbers without the "simulated data"
  label and the realism-gap caption — doing so would be exactly the kind of
  overclaim the project's own rules (and the independent reviewer) are
  designed to prevent.
