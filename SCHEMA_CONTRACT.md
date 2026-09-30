# Schema Contract (agreed up front — do not change without updating this file)

All agents build against these exact column names/types so integration doesn't break.
Parquet files use these schemas; Postgres tables mirror them.

## terminals
| column       | type    | notes                                  |
|--------------|---------|-----------------------------------------|
| terminal_id  | string  | e.g. "T00001"                           |
| lat          | float64 |                                          |
| lon          | float64 |                                          |
| h3_cell      | string  | H3 resolution 9                         |
| type         | string  | "ATM" or "POS"                          |
| bank         | string  |                                          |
| install_date | date    |                                          |

## accounts (mule + benign)
| column      | type   | notes                          |
|-------------|--------|----------------------------------|
| account_id  | string |                                   |
| is_mule     | bool   | ground truth, simulator-only     |
| opened_date | date   |                                   |

## transactions
| column       | type     | notes                                        |
|--------------|----------|-----------------------------------------------|
| txn_id       | string   |                                                |
| terminal_id  | string   | FK -> terminals                               |
| account_id   | string   | FK -> accounts                                |
| amount       | float64  |                                                |
| timestamp    | datetime | UTC                                            |
| is_mule_hop  | bool     | ground truth: part of a mule cash-out chain    |

## complaints
| column        | type     | notes                              |
|---------------|----------|--------------------------------------|
| complaint_id  | string   |                                       |
| account_id    | string   | victim or reported mule account      |
| filed_at      | datetime |                                       |
| state         | string   | Indian state name                    |

## feature table (terminal x 2h window) — model input/output
| column        | type     | notes                                                  |
|---------------|----------|----------------------------------------------------------|
| terminal_id   | string   |                                                            |
| window_start  | datetime | 2-hour bucket start, UTC                                  |
| label         | int      | 1 if mule withdrawal at (terminal, window), else 0         |
| split_tag     | string   | "train" / "val" / "test"                                   |
| sim_config    | string   | "A" / "B" / "C" — which simulator config generated this   |
| <feature cols>| float64  | e.g. txn_count_24h, past_hotspot_freq, complaint_density... |

## model output / API response (ranked terminals for officer view)
```json
{
  "terminal_id": "T00001",
  "lat": 0.0,
  "lon": 0.0,
  "window_start": "2026-01-01T00:00:00Z",
  "risk_score": 0.87,
  "rank": 1,
  "top_shap_features": [{"feature": "past_hotspot_freq", "value": 0.31}],
  "baseline_score": 0.42
}
```

## File locations
- `data/simulated/config_A/transactions/date=YYYY-MM-DD/part.parquet` (partitioned by day)
- same pattern for config_B, config_C, and for accounts/complaints (not partitioned, single file ok)
- `data/simulated/config_A/terminals.parquet` (shared across configs — same terminal universe)
- `data/processed/features_config_A.parquet` etc.
- `results/metrics_*.json` — model agent's saved run outputs, one file per run

## Fixture for early integration (Model/API agents don't need to wait on Data/Simulator)
Data agent should produce a SMALL fixture first (e.g. 50 terminals, 3 days, config A only)
at `data/simulated/fixture/` matching the exact schema above, so Model and API/UI agents can
start immediately. Full-scale generation (3 configs, full time range) follows after.
