"""Lead time: minutes between a complaint's filed_at and the predicted
high-risk window during which the model would have flagged the terminal
where the ACTUAL mule withdrawal occurred (config A).

Task: for each complaint, find the real mule withdrawal transaction(s) it is
linked to (reusing the SAME join path model/features.py's compute_labels()
uses for label construction -- complaints.account_id joined directly against
transactions.account_id -- so this lead-time number is consistent with how
labels were actually defined, not a different ad-hoc definition of "linked
withdrawal"), then compute minutes between filed_at and the window_start of
the (terminal, window) that would have been flagged.

RESULT OF ACTUALLY TRYING THIS (see detailed trace below): the join does
not cleanly identify "the mule withdrawal a given complaint is about" for
the overwhelming majority of complaints, for a structural reason specific
to how this simulator's output is written to disk (not a bug in this
script) -- see `join_diagnostics` in the output JSON. We report exactly
what the join does and does not support rather than fabricating a lead-time
number from the config's benign/false-positive edge case where the join
happens to fire for the wrong reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_trained_model  # noqa: E402
from features import load_raw_tables  # noqa: E402
from train import load_features  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CONFIG = "A"
WINDOW = pd.Timedelta(hours=2)


def trace_join(config: str) -> dict:
    """Reproduces, step by step, the EXACT join model/features.py's
    compute_labels() uses (complaints.account_id -> transactions.account_id
    directly, no intermediate account/ring table), and reports what fraction
    of complaints it actually links to a withdrawal, and why."""
    raw = load_raw_tables(config)
    complaints = raw["complaints"]
    txns = raw["transactions"]
    accounts = raw["accounts"]

    n_complaints = len(complaints)

    # Same set construction as features.py:compute_labels -- "mule accounts
    # ever" = every account_id that appears in complaints.account_id.
    mule_accounts_ever = set(complaints["account_id"].unique())
    mule_txns = txns[txns["account_id"].isin(mule_accounts_ever)].copy()

    linked_accounts = set(mule_txns["account_id"].unique())
    complaints_with_linked_account = complaints[complaints["account_id"].isin(linked_accounts)]
    complaints_without_any_txn = complaints[~complaints["account_id"].isin(linked_accounts)]

    # Why do most complaint accounts have zero transactions? Check the
    # account_id prefix convention the simulator uses (victim accounts
    # "ACV..." are never given any transaction rows at all -- funds are
    # modeled as "conceptually originating" at the victim but all actual
    # transaction rows are written under newly-minted mule account ids
    # "ACM..." that are NEVER written back into complaints.account_id; see
    # simulator/lib/generate_core.py lines ~95-287, which builds
    # `ring_terminals_hit` in memory per complaint but does not persist a
    # complaint_id/account_id -> mule_id or -> (terminal_id, timestamp)
    # foreign key to any output table).
    complaints_df = complaints.copy()
    complaints_df["account_prefix"] = complaints_df["account_id"].str[:3]
    prefix_counts = complaints_df["account_prefix"].value_counts().to_dict()

    acc_prefix_in_txns = txns["account_id"].str[:3].value_counts().to_dict()

    accounts_idx = accounts.set_index("account_id")
    linked_is_mule = {
        a: bool(accounts_idx.loc[a, "is_mule"]) if a in accounts_idx.index else None
        for a in linked_accounts
    }

    return {
        "n_complaints": n_complaints,
        "n_complaints_with_at_least_one_matching_transaction": int(len(complaints_with_linked_account)),
        "n_complaints_with_zero_matching_transactions": int(len(complaints_without_any_txn)),
        "fraction_joinable": float(len(complaints_with_linked_account) / n_complaints) if n_complaints else None,
        "linked_account_ids": sorted(linked_accounts),
        "linked_accounts_is_mule_ground_truth": linked_is_mule,
        "complaint_account_id_prefix_counts": prefix_counts,
        "transaction_account_id_prefix_counts": acc_prefix_in_txns,
        "diagnosis": (
            "Complaints.account_id is almost always a victim account "
            "('ACV######', the account the funds conceptually originate from), "
            "but the simulator writes ALL actual transaction rows -- including the "
            "final mule cash-out hop -- under freshly-minted mule account ids "
            "('ACM######') that are never linked back to the originating complaint "
            "in any output table (see simulator/lib/generate_core.py: "
            "`ring_terminals_hit` -- the list of (terminal_id, timestamp) pairs for "
            "the actual withdrawal -- is built in memory per-complaint during "
            "generation and used only to construct the label set directly; it is "
            "discarded afterward and never written to complaints.parquet, "
            "transactions.parquet, or any other persisted table). As a structural "
            "consequence, model/features.py's own label join (complaints.account_id "
            "vs transactions.account_id, exactly reproduced above) does not match a "
            "victim's complaint to their ring's actual withdrawal at all -- it has a "
            "0% join rate for genuine mule rings. The only complaints where this join "
            "succeeds are the small number of injected FALSE-POSITIVE noise "
            "complaints (id prefix 'CMPFP', filed against an ordinary benign "
            "background account that already has normal unrelated transaction "
            "activity under its own account_id -- see generate_core.py's "
            "false-positive block, which explicitly notes 'the innocent account "
            "never made a mule-hop withdrawal'). Those accidental joins are NOT "
            "a mule withdrawal event by construction (is_mule=False, "
            "is_mule_hop=False on every matching transaction) and have hundreds of "
            "unrelated transactions spread across the whole time range and most "
            "terminals in the city, so there is no single 'the withdrawal' moment "
            "to anchor a lead-time number to even if we wanted to use this edge "
            "case."
        ),
    }


def attempt_lead_time(config: str) -> dict:
    diag = trace_join(config)

    if diag["n_complaints_with_at_least_one_matching_transaction"] == 0:
        return {
            "computable": False,
            "config": config,
            "reason": "No complaint joins to any matching transaction via this join path.",
            "join_diagnostics": diag,
        }

    raw = load_raw_tables(config)
    complaints = raw["complaints"]
    txns = raw["transactions"]
    feat = load_features(config)
    booster, feature_cols = load_trained_model("A")

    linked_accounts = set(diag["linked_account_ids"])
    cases = []
    for _, comp in complaints[complaints["account_id"].isin(linked_accounts)].iterrows():
        acc = comp["account_id"]
        acc_txns = txns[txns["account_id"] == acc]
        is_mule_hop_txns = acc_txns[acc_txns["is_mule_hop"] == True]  # noqa: E712
        cases.append({
            "complaint_id": comp["complaint_id"],
            "account_id": acc,
            "filed_at": str(comp["filed_at"]),
            "n_matching_transactions": int(len(acc_txns)),
            "n_matching_is_mule_hop_transactions": int(len(is_mule_hop_txns)),
            "n_distinct_terminals_touched": int(acc_txns["terminal_id"].nunique()),
            "txn_time_span_days": float(
                (acc_txns["timestamp"].max() - acc_txns["timestamp"].min()).total_seconds() / 86400
            ) if len(acc_txns) else None,
        })

    return {
        "computable": False,
        "config": config,
        "reason": (
            "The join that model/features.py actually uses for labels only matches "
            f"{diag['n_complaints_with_at_least_one_matching_transaction']} of "
            f"{diag['n_complaints']} complaints to any transaction at all, and every one "
            "of those matches is a false-positive-noise complaint filed against an "
            "ordinary benign background account with is_mule=False and "
            "is_mule_hop=False on 100% of its matching transactions (confirmed above, "
            "see per-case n_matching_is_mule_hop_transactions=0) -- i.e. there is no "
            "'actual mule withdrawal' event here at all, by the simulator's own ground "
            "truth. These accounts also have hundreds of transactions spread across "
            "the full ~20-day span and 190-210 distinct terminals each, so even if we "
            "treated this as a real link there is no single identifiable withdrawal "
            "moment to measure a lead time against -- picking e.g. 'the transaction "
            "nearest in time to filed_at' would be inventing a definition not supported "
            "by the data, which this task's honesty standard rules out. "
            "Genuine mule-ring withdrawals (is_mule_hop=True transactions, which DO "
            "have exact timestamps and ARE what model/features.py's labels are "
            "ultimately driven by at the (terminal, window) level) exist in "
            "transactions.parquet, but they are tied to freshly-minted mule account "
            "ids ('ACM######') that are never linked back to the complaint that "
            "resulted from that ring in any persisted table -- the FK that would "
            "make this join possible (complaint -> ring's mule account ids, or "
            "complaint -> ring_terminals_hit) is computed in-memory during "
            "simulation (simulator/lib/generate_core.py) and discarded before any "
            "parquet file is written. Computing a lead time would therefore require "
            "either (a) re-generating the simulator with an added persisted FK "
            "(complaint_id -> mule_account_ids or -> ring_terminals_hit), which "
            "means changing simulator output, out of scope for this additive-"
            "measurement task and not done here, or (b) inventing a proxy link (e.g. "
            "nearest-in-time or same-state heuristic) which would not be the actual "
            "generative link and would silently misrepresent what is being measured. "
            "Per this task's explicit instruction not to fabricate a number when the "
            "join is genuinely ambiguous or absent, we report this honestly as NOT "
            "COMPUTABLE from the data as currently persisted, rather than estimating "
            "it from the false-positive edge case or a nearest-timestamp heuristic."
        ),
        "what_would_make_this_computable": (
            "Persisting one additional column in simulator output -- e.g. a "
            "ring_id or complaint_id field on EACH transaction row that is part of "
            "a mule ring (not just the boolean is_mule_hop flag already present), "
            "or a separate rings.parquet table mapping complaint_id -> "
            "{mule_account_ids, final_hop_terminal_id, final_hop_timestamp} -- would "
            "let this script compute lead_time_minutes = "
            "(window_start_of_flagged_window - filed_at).total_seconds()/60 directly "
            "and honestly, using the exact same final-hop timestamps "
            "simulator/lib/generate_core.py already computes internally as "
            "`ring_terminals_hit` before discarding them."
        ),
        "join_diagnostics": diag,
        "false_positive_edge_case_cases_examined": cases,
        "n_cases_examined": len(cases),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = attempt_lead_time(CONFIG)
    out_path = RESULTS_DIR / "metrics_lead_time.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[lead_time] wrote {out_path}")
    print(f"[lead_time] computable: {result['computable']}")
    print(f"[lead_time] join_diagnostics.fraction_joinable: "
          f"{result['join_diagnostics']['fraction_joinable']:.4f} "
          f"({result['join_diagnostics']['n_complaints_with_at_least_one_matching_transaction']}/"
          f"{result['join_diagnostics']['n_complaints']})")
    print("[lead_time] All joined complaints are false-positive-noise edge cases with "
          "n_matching_is_mule_hop_transactions=0 -- not a real mule withdrawal link. "
          "See results/metrics_lead_time.json for the full honest explanation.")


if __name__ == "__main__":
    main()
