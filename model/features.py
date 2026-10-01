"""Feature engineering: build the (terminal x 2h-window) feature table.

Reads transactions/accounts/complaints/terminals for a single simulator config
(or the dev fixture) and emits data/processed/features_config_<X>.parquet
matching SCHEMA_CONTRACT.md's "feature table" section.

LEAKAGE DISCIPLINE (mandatory per PROJECT_SPEC.md):
  - Every feature for window [window_start, window_start + 2h) uses ONLY
    transactions/complaints with timestamp < window_start (strictly as-of).
  - The account-graph feature (shortest path to a complaint-linked account) is
    built from a graph containing only edges (account, terminal) observed
    BEFORE window_start.
  - Split is rolling-origin by calendar time with a 7-day embargo gap between
    the end of train and the start of test (and between val and test). No
    window that falls inside the embargo gets any split_tag other than
    "embargo" (dropped before saving... actually we keep it but never assign
    train/val/test so it can never leak). See `make_rolling_origin_splits`.

Label definition (PROJECT_SPEC.md / SCHEMA_CONTRACT.md):
  label = 1 if a complaint-linked mule account withdraws (has a transaction)
  at that (terminal, window) bucket. "Complaint-linked" = account_id that is
  the subject of a complaint (SCHEMA_CONTRACT: complaints.account_id) OR that
  is flagged is_mule=True in accounts (ground truth flag the simulator writes
  for dev/eval purposes only -- NOT used as a feature, only for labeling,
  matching how a real bank would learn about mule accounts: via complaints).
  We label from complaints.account_id to mirror the real-world signal path
  (a complaint arrives, and *then* we know an account is mule-linked) rather
  than from the simulator's internal is_mule ground truth, which would be a
  form of label leakage from data the bank would not actually have.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOW = pd.Timedelta(hours=2)
EMBARGO = pd.Timedelta(days=7)
H3_RES = 9
H3_KRING = 2  # neighbor ring for spatial complaint density

# rolling windows (lookback from window_start) for txn count/amount features
LOOKBACKS_H = [2, 6, 24, 24 * 7]


def _data_root_for_config(config: str) -> Path:
    """Resolve where a given config's raw tables live.

    Prefers the real simulator output under data/simulated/<config>/ if it
    exists; otherwise falls back to the dev fixture at
    data/processed/dev_fixture/fx_<letter>/ (config="A" -> fx_A, etc.).
    """
    real = REPO_ROOT / "data" / "simulated" / f"config_{config}"
    if (real / "accounts.parquet").exists():
        return real
    fixture = REPO_ROOT / "data" / "processed" / "dev_fixture" / f"fx_{config}"
    if fixture.exists():
        return fixture
    raise FileNotFoundError(
        f"No data found for config {config!r}: checked {real} and {fixture}"
    )


def _terminals_path_for(data_root: Path) -> Path:
    # terminals.parquet is shared across configs in data/simulated/, but each
    # dev_fixture config dir doesn't have its own -- it's one level up.
    local = data_root / "terminals.parquet"
    if local.exists():
        return local
    shared = data_root.parent / "terminals.parquet"
    if shared.exists():
        return shared
    raise FileNotFoundError(f"terminals.parquet not found near {data_root}")


def load_raw_tables(config: str) -> dict[str, pd.DataFrame]:
    data_root = _data_root_for_config(config)
    terminals = pd.read_parquet(_terminals_path_for(data_root))
    accounts = pd.read_parquet(data_root / "accounts.parquet")
    complaints = pd.read_parquet(data_root / "complaints.parquet")

    txn_dir = data_root / "transactions"
    parts = sorted(txn_dir.glob("date=*/part.parquet"))
    if not parts:
        # single-file fallback
        parts = sorted(txn_dir.glob("*.parquet"))
    txns = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    txns = txns.sort_values("timestamp").reset_index(drop=True)

    return {"terminals": terminals, "accounts": accounts, "complaints": complaints,
            "transactions": txns}


def _mule_linked_accounts_as_of(complaints: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
    """Accounts that are complaint-linked using only complaints filed strictly
    before `as_of` -- this is what a bank would actually know at that time."""
    if len(complaints) == 0:
        return set()
    known = complaints[complaints["filed_at"] < as_of]
    return set(known["account_id"].unique())


def build_windows(terminals: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    """Cartesian terminal x 2h-window grid spanning the observed transaction range."""
    t_min = txns["timestamp"].min().floor("2h")
    t_max = txns["timestamp"].max().ceil("2h")
    window_starts = pd.date_range(t_min, t_max, freq=WINDOW, inclusive="left")
    grid = pd.MultiIndex.from_product(
        [terminals["terminal_id"], window_starts], names=["terminal_id", "window_start"]
    ).to_frame(index=False)
    return grid


def _h3_kring_distance_lookup(terminals: pd.DataFrame) -> dict[str, set[str]]:
    """terminal_id -> set of h3 cells within H3_KRING of its own cell (for spatial density)."""
    out = {}
    for _, row in terminals.iterrows():
        out[row["terminal_id"]] = set(h3.grid_disk(row["h3_cell"], H3_KRING))
    return out


def compute_labels(windows: pd.DataFrame, txns: pd.DataFrame, complaints: pd.DataFrame) -> pd.DataFrame:
    """label=1 if a complaint-linked account (as known via complaints filed at
    ANY time -- ground truth for evaluation, this is the target, not a
    feature) withdraws at that (terminal, window)."""
    if len(complaints) == 0:
        windows = windows.copy()
        windows["label"] = 0
        return windows

    mule_accounts_ever = set(complaints["account_id"].unique())
    mule_txns = txns[txns["account_id"].isin(mule_accounts_ever)].copy()
    mule_txns["window_start"] = mule_txns["timestamp"].dt.floor("2h")
    hits = mule_txns[["terminal_id", "window_start"]].drop_duplicates()
    hits["label"] = 1

    out = windows.merge(hits, on=["terminal_id", "window_start"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    return out


def compute_rolling_txn_features(windows: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    """As-of rolling txn count/amount per terminal over LOOKBACKS_H, using only
    transactions strictly before window_start (no same-window leakage)."""
    txns_sorted = txns.sort_values("timestamp")
    out = windows.copy()

    for lb in LOOKBACKS_H:
        cnt_col = f"txn_count_{lb}h"
        amt_col = f"txn_amount_sum_{lb}h"
        out[cnt_col] = 0
        out[amt_col] = 0.0

    # group txns by terminal for efficient searchsorted lookups
    grouped = {tid: g.reset_index(drop=True) for tid, g in txns_sorted.groupby("terminal_id")}

    results = {f"txn_count_{lb}h": [] for lb in LOOKBACKS_H}
    results.update({f"txn_amount_sum_{lb}h": [] for lb in LOOKBACKS_H})

    # process per-terminal in window order for speed
    out = out.sort_values(["terminal_id", "window_start"]).reset_index(drop=True)
    for tid, sub in out.groupby("terminal_id", sort=False):
        g = grouped.get(tid)
        idx = sub.index
        ws = sub["window_start"].values
        if g is None or len(g) == 0:
            for lb in LOOKBACKS_H:
                results[f"txn_count_{lb}h"].extend([0] * len(ws))
                results[f"txn_amount_sum_{lb}h"].extend([0.0] * len(ws))
            continue
        ts = g["timestamp"].values.astype("datetime64[ns]")
        amt = g["amount"].values
        cumsum_amt = np.concatenate([[0.0], np.cumsum(amt)])
        end_pos = np.searchsorted(ts, ws, side="left")  # txns strictly before window_start
        for lb in LOOKBACKS_H:
            start_ts = ws - np.timedelta64(lb, "h")
            start_pos = np.searchsorted(ts, start_ts, side="left")
            counts = end_pos - start_pos
            amounts = cumsum_amt[end_pos] - cumsum_amt[start_pos]
            results[f"txn_count_{lb}h"].extend(counts.tolist())
            results[f"txn_amount_sum_{lb}h"].extend(amounts.tolist())

    for col, vals in results.items():
        out[col] = vals
    return out


def compute_past_hotspot_freq(windows: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """As-of historical label rate at that terminal: fraction of PAST windows
    (strictly before this window_start) at this terminal that were label=1.
    This is both a feature and the baseline itself (baselines.py reuses this
    same as-of computation)."""
    df = labels.sort_values(["terminal_id", "window_start"]).copy()
    df["past_hotspot_freq"] = (
        df.groupby("terminal_id")["label"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["past_hotspot_freq"] = df["past_hotspot_freq"].fillna(0.0)
    return df.set_index(["terminal_id", "window_start"])["past_hotspot_freq"]


def compute_complaint_density(windows: pd.DataFrame, terminals: pd.DataFrame,
                               complaints: pd.DataFrame, txns: pd.DataFrame) -> pd.Series:
    """Spatial complaint density near a terminal: count of DISTINCT complaints
    filed strictly before window_start whose account had a transaction at any
    terminal within H3_KRING of this terminal's h3 cell, at any time before
    window_start (as-of). This proxies "complaints near here recently".

    Vectorized: for each terminal, find the earliest time each complainant
    account was ever seen transacting within its H3 k-ring (a single scalar
    per (terminal, account) pair), then for each window_start just count how
    many complaints (filed before ws) have that earliest-seen time < ws. This
    avoids per-window Python-level loops entirely -- it's an as-of join via
    searchsorted on a small per-terminal sorted array.
    """
    unique_windows = windows[["terminal_id", "window_start"]].drop_duplicates()
    idx_all = pd.MultiIndex.from_frame(unique_windows)

    if len(complaints) == 0:
        return pd.Series(0.0, index=idx_all)

    kring = _h3_kring_distance_lookup(terminals)
    term_cell = terminals.set_index("terminal_id")["h3_cell"].to_dict()

    acc_txn = txns[["account_id", "terminal_id", "timestamp"]].copy()
    acc_txn["h3_cell"] = acc_txn["terminal_id"].map(term_cell)

    # Only complainant accounts matter -- restrict early to shrink the join.
    complainant_accounts = set(complaints["account_id"].unique())
    acc_txn = acc_txn[acc_txn["account_id"].isin(complainant_accounts)]

    comp_sorted = complaints.sort_values("filed_at")
    comp_time = comp_sorted["filed_at"].values.astype("datetime64[ns]")
    comp_acc = comp_sorted["account_id"].values

    out_vals = {}
    for ti, (tid, sub) in enumerate(unique_windows.groupby("terminal_id")):
        if ti % 20 == 0:
            print(f"[features]   complaint_density progress: {ti} terminals")
        cell_set = kring.get(tid, set())
        nearby = acc_txn[acc_txn["h3_cell"].isin(cell_set)]
        ws_sorted = np.sort(sub["window_start"].values)

        if len(nearby) == 0:
            for ws in ws_sorted:
                out_vals[(tid, pd.Timestamp(ws))] = 0.0
            continue

        # earliest-seen-near-this-terminal time per complainant account
        first_seen = nearby.groupby("account_id")["timestamp"].min()
        # align to complaint order; accounts never seen near here -> NaT (never counts)
        first_seen_per_complaint = first_seen.reindex(comp_acc).values.astype("datetime64[ns]")

        # for each window_start: count complaints with filed_at < ws AND first_seen < ws.
        # Since both conditions are "< ws", the binding constraint is min(filed_at, first_seen) < ws.
        eligible_time = np.where(
            np.isnat(first_seen_per_complaint), np.datetime64("NaT"),
            np.maximum(comp_time, first_seen_per_complaint)
        )
        valid = ~np.isnat(eligible_time)
        eligible_sorted = np.sort(eligible_time[valid])
        counts = np.searchsorted(eligible_sorted, ws_sorted, side="left")
        for ws, c in zip(ws_sorted, counts):
            out_vals[(tid, pd.Timestamp(ws))] = float(c)

    idx = pd.MultiIndex.from_tuples(out_vals.keys(), names=["terminal_id", "window_start"])
    return pd.Series(out_vals.values(), index=idx)


def compute_graph_distance_feature(windows: pd.DataFrame, terminals: pd.DataFrame,
                                    txns: pd.DataFrame, complaints: pd.DataFrame) -> pd.Series:
    """As-of shortest-path distance (in the account-terminal bipartite graph)
    from any account recently active at this terminal to the nearest
    complaint-linked account known as-of that time. Smaller = more
    graph-proximate to known mule activity. Capped at 6; -1 means
    unreachable/no info.

    PERFORMANCE / CORRECTNESS NOTE (fixed from a prior version):
    The original implementation rebuilt a fresh NetworkX graph from a 7-day
    lookback slice of transactions AND ran a full BFS for EVERY distinct
    window_start (~250 windows for config A). Since consecutive 2h windows'
    7-day lookback slices overlap by ~84 windows, this did ~250x redundant
    graph construction + BFS work and did not finish in a reasonable time
    (observed to hang for 1000+ CPU-seconds without completing).

    Fix (option 1 from the task brief): sweep window_starts in time order and
    maintain ONE cumulative graph, adding only the NEW edges for the
    just-completed window at each step (edges from transactions with
    window_start <= t < next window_start), then run a single multi-source
    BFS (via a virtual super-source connected to all known-mule nodes) from
    the graph's CURRENT state -- i.e. only edges strictly before this
    window's start are ever present when we query it. This is also more
    correct than the old 7-day-lookback-cap version: it's a true as-of
    cumulative graph (no artificial recency window), closing a leakage gap
    risk from a fixed lookback silently including/excluding edges
    inconsistently near boundaries. Complexity: O(num_windows) incremental
    edge additions (total edges added across all windows = n_txns, not
    n_windows * edges_per_lookback) + O(num_windows) BFS calls on the
    graph-so-far, each bounded by cutoff=MAX_DIST+1 hops -- tractable at this
    scale (1150 terminals, ~194k txns, ~252 windows).
    """
    MAX_DIST = 6

    txns_sorted = txns.sort_values("timestamp").reset_index(drop=True)

    unique_ws = sorted(windows["window_start"].unique())
    out_vals = {}

    # pre-group windows by window_start once (avoid re-filtering the full 289k-row frame per window)
    terms_by_ws = {ws: sub.values for ws, sub in windows.groupby("window_start")["terminal_id"]}

    # complaints sorted once, for fast as-of "known mules" slicing per window
    complaints_sorted = complaints.sort_values("filed_at") if len(complaints) else complaints
    complaint_filed = complaints_sorted["filed_at"].values.astype("datetime64[ns]") if len(complaints) else np.array([])
    complaint_acc = complaints_sorted["account_id"].values if len(complaints) else np.array([])

    # last-active counterparty account per terminal, tracked incrementally as we sweep forward
    last_acc_by_terminal: dict[str, str] = {}

    SUPER_SOURCE = "__mule_source__"
    G = nx.Graph()

    ts_arr = txns_sorted["timestamp"].values.astype("datetime64[ns]")
    acc_arr = txns_sorted["account_id"].values
    term_arr = txns_sorted["terminal_id"].values

    edge_cursor = 0  # index into txns_sorted of the next edge not yet added to G
    n_txns = len(txns_sorted)

    for i, ws in enumerate(unique_ws):
        if i % 50 == 0:
            print(f"[features]   graph_dist progress: {i}/{len(unique_ws)} windows")
        ws_ts = np.datetime64(ws)

        # advance the cursor: add every transaction strictly before this window_start
        # that we haven't added yet (as-of correctness: this window never sees itself
        # or future transactions).
        end = np.searchsorted(ts_arr, ws_ts, side="left")
        if end > edge_cursor:
            new_accs = acc_arr[edge_cursor:end]
            new_terms = term_arr[edge_cursor:end]
            edge_pairs = list(zip(
                (f"acc::{a}" for a in new_accs),
                (f"term::{t}" for t in new_terms),
            ))
            G.add_edges_from(edge_pairs)
            for a, t in zip(new_accs, new_terms):
                last_acc_by_terminal[t] = a
            edge_cursor = end

        if G.number_of_edges() == 0:
            for tid in terms_by_ws.get(ws, []):
                out_vals[(tid, ws)] = -1
            continue

        # known mules as-of this window (complaints filed strictly before ws)
        n_known = np.searchsorted(complaint_filed, ws_ts, side="left") if len(complaint_filed) else 0
        known_mules = set(complaint_acc[:n_known]) if n_known else set()
        mule_nodes = [f"acc::{a}" for a in known_mules if f"acc::{a}" in G]

        if not mule_nodes:
            dist_from_mules = {}
        else:
            # single BFS from a virtual super-source connected to all known-mule nodes,
            # instead of looping single_source_shortest_path_length per mule node.
            G.add_node(SUPER_SOURCE)
            G.add_edges_from((SUPER_SOURCE, m) for m in mule_nodes)
            lengths = nx.single_source_shortest_path_length(G, SUPER_SOURCE, cutoff=MAX_DIST + 1)
            # subtract 1 to undo the extra super-source hop; mule nodes themselves get distance 0
            dist_from_mules = {node: d - 1 for node, d in lengths.items() if node != SUPER_SOURCE}
            G.remove_node(SUPER_SOURCE)

        for tid in terms_by_ws.get(ws, []):
            last_acc = last_acc_by_terminal.get(tid)
            if last_acc is None:
                out_vals[(tid, ws)] = -1
                continue
            node = f"acc::{last_acc}"
            out_vals[(tid, ws)] = dist_from_mules.get(node, -1) if dist_from_mules else -1

    idx = pd.MultiIndex.from_tuples(out_vals.keys(), names=["terminal_id", "window_start"])
    return pd.Series([out_vals[k] for k in out_vals], index=idx)


def make_rolling_origin_splits(windows: pd.DataFrame) -> pd.Series:
    """Rolling-origin split by calendar time with a 7-day embargo between
    train and test (and between train/val and test), per PROJECT_SPEC.md.

    Timeline layout (fractions of the full observed window_start range):
      [-------- train 60% --------][--embargo 7d--][val 15%][--embargo 7d--][test up to 25%]
    If the data range is too short to fit both embargoes + all three splits,
    embargo is still enforced exactly (7 days) and split sizes shrink instead
    -- correctness of the embargo is never compromised for coverage.
    """
    ws = np.sort(windows["window_start"].unique())
    t0, t1 = pd.Timestamp(ws[0]), pd.Timestamp(ws[-1])
    total = t1 - t0

    train_end = t0 + total * 0.55
    val_start = train_end + EMBARGO
    val_end = val_start + total * 0.15
    test_start = val_end + EMBARGO

    tags = pd.Series(index=windows.index, dtype=object)
    w = windows["window_start"]
    tags[w <= train_end] = "train"
    tags[(w > train_end) & (w < val_start)] = "embargo"
    tags[(w >= val_start) & (w <= val_end)] = "val"
    tags[(w > val_end) & (w < test_start)] = "embargo"
    tags[w >= test_start] = "test"
    return tags


def build_features(config: str) -> pd.DataFrame:
    print(f"[features] loading raw tables for config {config}")
    raw = load_raw_tables(config)
    terminals, accounts, complaints, txns = (
        raw["terminals"], raw["accounts"], raw["complaints"], raw["transactions"]
    )

    print(f"[features] {len(terminals)} terminals, {len(txns)} txns, {len(complaints)} complaints")
    windows = build_windows(terminals, txns)
    print(f"[features] {len(windows)} terminal x window rows")

    labels = compute_labels(windows, txns, complaints)
    print(f"[features] label positive rate: {labels['label'].mean():.4f}")

    feat = compute_rolling_txn_features(labels, txns)

    phf = compute_past_hotspot_freq(feat[["terminal_id", "window_start", "label"]], feat)
    feat = feat.set_index(["terminal_id", "window_start"])
    feat["past_hotspot_freq"] = phf
    feat = feat.reset_index()

    print("[features] computing complaint density (spatial, as-of)...")
    dens = compute_complaint_density(feat, terminals, complaints, txns)
    feat = feat.set_index(["terminal_id", "window_start"])
    feat["complaint_density_kring"] = dens
    feat["complaint_density_kring"] = feat["complaint_density_kring"].fillna(0.0)
    feat = feat.reset_index()

    print("[features] computing as-of graph distance feature...")
    gdist = compute_graph_distance_feature(feat, terminals, txns, complaints)
    feat = feat.set_index(["terminal_id", "window_start"])
    feat["graph_dist_to_known_mule"] = gdist
    feat["graph_dist_to_known_mule"] = feat["graph_dist_to_known_mule"].fillna(-1)
    feat = feat.reset_index()

    feat["split_tag"] = make_rolling_origin_splits(feat)
    feat["sim_config"] = config

    term_meta = terminals[["terminal_id", "lat", "lon", "type", "bank"]]
    feat = feat.merge(term_meta, on="terminal_id", how="left")

    ordered_cols = [
        "terminal_id", "window_start", "label", "split_tag", "sim_config",
        "lat", "lon", "type", "bank",
    ] + [c for c in feat.columns if c.startswith("txn_")] + [
        "past_hotspot_freq", "complaint_density_kring", "graph_dist_to_known_mule",
    ]
    feat = feat[ordered_cols]
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="A, B, C (real) or fx_A/fx_B/fx_C style letter e.g. 'A'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feat = build_features(args.config)
    out_path = Path(args.out) if args.out else REPO_ROOT / "data" / "processed" / f"features_config_{args.config}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    print(f"[features] wrote {len(feat)} rows -> {out_path}")
    print(feat["split_tag"].value_counts())


if __name__ == "__main__":
    main()
