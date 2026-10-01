"""Core seeded generator: builds accounts, transactions, complaints for one SimConfig.

Label mechanism (kept deliberately simple for this thin slice):
  label(terminal, 2h window) = 1  iff  a complaint-linked mule account made a cash-out
  withdrawal (is_mule_hop=True, final hop of a ring) at that terminal within that window,
  AND that ring's complaint was (eventually) filed -- i.e. not dropped by the
  false_negative_rate noise. Label noise/delay is applied on top of this ground truth by
  the feature-table stage (model/features.py) using the complaint filed_at + label_delay; we
  hand off ground truth (is_mule_hop, complaint linkage) and let the delay be expressed via
  complaints.filed_at, not by mutating transactions.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from .config import SimConfig
from .dist import sample_one

INDIAN_STATES = [
    "Karnataka", "Maharashtra", "Delhi", "Tamil Nadu", "Telangana",
    "Uttar Pradesh", "West Bengal", "Gujarat", "Rajasthan", "Kerala",
    "Punjab", "Haryana", "Bihar", "Madhya Pradesh", "Andhra Pradesh",
]


def _daterange(start: date, end: date):
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)


@dataclass
class GenResult:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    complaints: pd.DataFrame
    stats: dict


def generate(cfg: SimConfig, terminals: pd.DataFrame) -> GenResult:
    seed = cfg.seed
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    faker = Faker()
    Faker.seed(seed)

    start = datetime.strptime(cfg.time_range[0], "%Y-%m-%d").date()
    end = datetime.strptime(cfg.time_range[1], "%Y-%m-%d").date()
    days = list(_daterange(start, end))

    term_ids = terminals["terminal_id"].tolist()
    term_by_id = terminals.set_index("terminal_id")
    n_terminals = len(term_ids)

    # ---------------- accounts ----------------
    accounts_rows = []
    n_benign = int(cfg.scale.get("num_benign_accounts", 500))

    def new_account_id(idx: int, prefix: str) -> str:
        return f"{prefix}{idx:06d}"

    benign_open_start = start - timedelta(days=720)
    for i in range(n_benign):
        opened = benign_open_start + timedelta(
            days=int(rng.integers(0, (end - benign_open_start).days))
        )
        accounts_rows.append(
            {"account_id": new_account_id(i, "ACB"), "is_mule": False, "opened_date": opened}
        )
    benign_account_ids = [r["account_id"] for r in accounts_rows]

    # ---------------- mule rings ----------------
    mule_net = cfg.mule_network
    noise = cfg.label_noise
    n_complaints = int(cfg.scale.get("num_complaints", 100))

    complaints_rows = []
    mule_txn_rows = []
    mule_account_counter = 0
    victim_account_counter = 0

    def new_mule_id() -> str:
        nonlocal mule_account_counter
        mule_account_counter += 1
        return f"ACM{mule_account_counter:06d}"

    def new_victim_id() -> str:
        nonlocal victim_account_counter
        victim_account_counter += 1
        return f"ACV{victim_account_counter:06d}"

    def pick_terminal(strategy: str, last_terminal_id: str | None, recent_hotspots: set) -> str:
        if strategy == "random_in_city":
            return py_rng.choice(term_ids)
        if strategy == "nearest_to_last_hop":
            if last_terminal_id is None:
                return py_rng.choice(term_ids)
            lat0 = term_by_id.loc[last_terminal_id, "lat"]
            lon0 = term_by_id.loc[last_terminal_id, "lon"]
            # sample a small candidate pool, pick nearest (keeps O(n) cheap at this scale)
            cand = py_rng.sample(term_ids, k=min(15, n_terminals))
            best = min(
                cand,
                key=lambda t: (term_by_id.loc[t, "lat"] - lat0) ** 2
                + (term_by_id.loc[t, "lon"] - lon0) ** 2,
            )
            return best
        if strategy == "hotspot_avoidant":
            cand = [t for t in term_ids if t not in recent_hotspots]
            if not cand:
                cand = term_ids
            return py_rng.choice(cand)
        raise ValueError(f"Unknown terminal_choice: {strategy}")

    txn_counter = 0

    def new_txn_id() -> str:
        nonlocal txn_counter
        txn_counter += 1
        return f"TXN{txn_counter:08d}"

    # track terminals used heavily by mule activity so far, for hotspot_avoidant heuristic
    hotspot_terminal_counts: dict[str, int] = {}
    hotspot_threshold = 3

    label_windows_pos: set[tuple[str, pd.Timestamp]] = set()

    for c_idx in range(n_complaints):
        victim_id = new_victim_id()
        accounts_rows.append(
            {
                "account_id": victim_id,
                "is_mule": False,
                "opened_date": benign_open_start
                + timedelta(days=int(rng.integers(0, (end - benign_open_start).days))),
            }
        )

        # Incident start time: random moment within the FIRST PORTION of the config's time
        # range, leaving headroom for the mule chain (hop_count x delay_between_hops) to
        # unfold before `end` without drifting the generated transactions far outside the
        # documented time_range. Configs with longer/slower chains (e.g. B) effectively get
        # less headroom fraction used per complaint, which is fine -- we clip below anyway.
        headroom_days = max(1, int(len(days) * 0.7))
        incident_day = py_rng.choice(days[:headroom_days]) if headroom_days > 1 else days[0]
        incident_dt = datetime(
            incident_day.year, incident_day.month, incident_day.day,
            py_rng.randint(0, 23), py_rng.randint(0, 59),
        )
        # Hard cap: no transaction in this ring may land more than 10 days after the
        # configured end date. If a sampled hop delay would exceed that, clip the hop's
        # timestamp to the cap (keeps ordering monotonic, just compresses the tail) rather
        # than silently letting rare heavy-tailed lognormal draws blow the window out by
        # weeks (observed in testing with config B's wide delay distribution).
        max_dt = datetime(end.year, end.month, end.day) + timedelta(days=10)

        fan_out = int(sample_one(mule_net["fan_out_accounts_per_complaint"], rng))
        fan_out = max(1, fan_out)
        ring_terminals_hit: list[tuple[str, pd.Timestamp]] = []
        ring_dropped_by_fn = py_rng.random() < noise["false_negative_rate"]

        for branch in range(fan_out):
            hop_count = int(sample_one(mule_net["hop_count_distribution"], rng))
            hop_count = max(1, hop_count)
            t = incident_dt
            last_terminal = None
            chain_account = victim_id  # funds conceptually originate at victim
            for hop in range(hop_count):
                delay_h = float(sample_one(mule_net["delay_between_hops_hours"], rng))
                t = min(t + timedelta(hours=delay_h), max_dt)
                mule_id = new_mule_id()
                accounts_rows.append(
                    {
                        "account_id": mule_id,
                        "is_mule": True,
                        "opened_date": t.date() - timedelta(days=int(rng.integers(1, 60))),
                    }
                )
                is_final_hop = hop == hop_count - 1
                if is_final_hop:
                    terminal_id = pick_terminal(
                        mule_net["terminal_choice"], last_terminal,
                        {tid for tid, cnt in hotspot_terminal_counts.items() if cnt >= hotspot_threshold},
                    )
                    hotspot_terminal_counts[terminal_id] = hotspot_terminal_counts.get(terminal_id, 0) + 1

                    if mule_net["cash_out_pattern"] == "single_large_withdrawal":
                        amount = float(rng.lognormal(9.5, 0.6))  # large single pull
                        txn_ts = pd.Timestamp(t)
                        mule_txn_rows.append(
                            {
                                "txn_id": new_txn_id(),
                                "terminal_id": terminal_id,
                                "account_id": mule_id,
                                "amount": round(amount, 2),
                                "timestamp": txn_ts,
                                "is_mule_hop": True,
                            }
                        )
                        ring_terminals_hit.append((terminal_id, txn_ts))
                    else:  # structured_below_threshold
                        n_sub = py_rng.randint(2, 4)
                        for s in range(n_sub):
                            sub_t = t + timedelta(minutes=py_rng.randint(0, 90) * s)
                            amount = float(rng.uniform(4000, 9800))  # kept under common 10k reporting threshold
                            txn_ts = pd.Timestamp(sub_t)
                            mule_txn_rows.append(
                                {
                                    "txn_id": new_txn_id(),
                                    "terminal_id": terminal_id,
                                    "account_id": mule_id,
                                    "amount": round(amount, 2),
                                    "timestamp": txn_ts,
                                    "is_mule_hop": True,
                                }
                            )
                            ring_terminals_hit.append((terminal_id, txn_ts))
                    last_terminal = terminal_id
                else:
                    # intermediate hop: an inter-account transfer, modeled as a transaction at
                    # a terminal near the previous hop (POS/ATM transfer-adjacent activity)
                    terminal_id = pick_terminal(
                        mule_net["terminal_choice"], last_terminal,
                        {tid for tid, cnt in hotspot_terminal_counts.items() if cnt >= hotspot_threshold},
                    )
                    amount = float(rng.lognormal(8.5, 0.7))
                    mule_txn_rows.append(
                        {
                            "txn_id": new_txn_id(),
                            "terminal_id": terminal_id,
                            "account_id": mule_id,
                            "amount": round(amount, 2),
                            "timestamp": pd.Timestamp(t),
                            "is_mule_hop": True,
                        }
                    )
                    last_terminal = terminal_id
                chain_account = mule_id

        if not ring_dropped_by_fn:
            last_hit_time = max(ts for _, ts in ring_terminals_hit) if ring_terminals_hit else pd.Timestamp(incident_dt)
            delay_days = float(sample_one(noise["label_delay_days"], rng))
            filed_at = last_hit_time + timedelta(days=delay_days)
            complaints_rows.append(
                {
                    "complaint_id": f"CMP{c_idx+1:06d}",
                    "account_id": victim_id,
                    "filed_at": filed_at,
                    "state": py_rng.choice(INDIAN_STATES),
                }
            )
            for tid, ts in ring_terminals_hit:
                window_start = ts.floor("2h")
                label_windows_pos.add((tid, window_start))
        # if dropped by false negative: transactions still exist (is_mule_hop=True ground
        # truth preserved) but no complaint row is filed, so the feature/label join in
        # model/features.py yields 0 for that ring's windows unless another ring also hits
        # the same window -> correctly simulates a missed real-world detection.

    # ---------------- false positive complaints (innocent accounts misattributed) ----------------
    n_fp = int(round(n_complaints * noise["false_positive_rate"]))
    for i in range(n_fp):
        innocent_id = py_rng.choice(benign_account_ids)
        fp_day = py_rng.choice(days[:-1]) if len(days) > 1 else days[0]
        filed_at = datetime(fp_day.year, fp_day.month, fp_day.day, py_rng.randint(0, 23), py_rng.randint(0, 59))
        complaints_rows.append(
            {
                "complaint_id": f"CMPFP{i+1:06d}",
                "account_id": innocent_id,
                "filed_at": pd.Timestamp(filed_at),
                "state": py_rng.choice(INDIAN_STATES),
            }
        )
        # Note: FP complaints deliberately do NOT create any (terminal,window) label=1 entries
        # here, since the innocent account never made a mule-hop withdrawal -- consumers of
        # ground truth should join complaints -> is_mule_hop transactions only, so this FP
        # complaint correctly contributes no positive label but does add label *noise* at the
        # complaint level (an accusation with no matching transaction), which the
        # feature/label build (model/features.py) should account for when deriving
        # account-level suspicion features.

    # ---------------- benign background transactions ----------------
    benign_rows = []
    lam = cfg.benign_background["daily_txn_volume_per_terminal"]
    amt_spec = cfg.benign_background["amount_distribution"]
    for d in days:
        for tid in term_ids:
            n_txn = int(sample_one(lam, rng))
            if n_txn <= 0:
                continue
            seconds = rng.integers(0, 24 * 3600, size=n_txn)
            amounts = rng.lognormal(amt_spec["mu"], amt_spec["sigma"], size=n_txn)
            accts = py_rng.choices(benign_account_ids, k=n_txn)
            for i in range(n_txn):
                ts = datetime(d.year, d.month, d.day) + timedelta(seconds=int(seconds[i]))
                benign_rows.append(
                    {
                        "txn_id": new_txn_id(),
                        "terminal_id": tid,
                        "account_id": accts[i],
                        "amount": round(float(amounts[i]), 2),
                        "timestamp": pd.Timestamp(ts),
                        "is_mule_hop": False,
                    }
                )

    transactions = pd.DataFrame(benign_rows + mule_txn_rows)
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    accounts = pd.DataFrame(accounts_rows)
    accounts["opened_date"] = pd.to_datetime(accounts["opened_date"])

    complaints = pd.DataFrame(complaints_rows)
    if not complaints.empty:
        complaints["filed_at"] = pd.to_datetime(complaints["filed_at"])
        complaints = complaints.sort_values("filed_at").reset_index(drop=True)

    stats = {
        "n_accounts": len(accounts),
        "n_mule_accounts": int(accounts["is_mule"].sum()),
        "n_transactions": len(transactions),
        "n_mule_hop_transactions": int(transactions["is_mule_hop"].sum()),
        "n_complaints": len(complaints),
        "n_positive_terminal_windows": len(label_windows_pos),
    }

    return GenResult(accounts=accounts, transactions=transactions, complaints=complaints, stats=stats)
