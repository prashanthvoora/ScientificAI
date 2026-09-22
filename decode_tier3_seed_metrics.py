#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 30

def first_existing(cols, names):
    for name in names:
        if name in cols:
            return name
    return None

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) == 0:
        raise ValueError("No valid target/prediction pairs found.")
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mad = float(np.mean(np.abs(yt - np.mean(yt))))
    mad_mae = float(mad / mae) if mae > 0 else np.inf
    return len(yt), mae, rmse, mad, mad_mae, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True,
                    help="Path to tier3_best_checkpoint_test_prediction_audit.csv")
    ap.add_argument("--seed", required=True, type=int,
                    help="Seed number, e.g. 124 or 777")
    ap.add_argument("--summary", default="tier3_seed_metrics_summary.csv",
                    help="Summary CSV to create/update")
    ap.add_argument("--expected-n", type=int, default=EXPECTED_N)
    args = ap.parse_args()

    path = Path(args.audit)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)
    print("="*72)
    print("Tier-3 best-checkpoint TEST metric decoder")
    print("="*72)
    print(f"Seed      : {args.seed}")
    print(f"Audit file: {path}")
    print(f"CSV rows  : {len(df)}")

    true_col = first_existing(df.columns,
        ["k_true","k_measured","k_total_true","target","y_true"])
    pred_col = first_existing(df.columns,
        ["k_pred","k_total_pred","prediction","y_pred"])

    if true_col is None or pred_col is None:
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit("ERROR: could not find linear-space target/prediction columns.")

    true_log_col = first_existing(df.columns,
        ["k_true_log","k_total_log_true","k_measured_log_true","y_true_log"])
    pred_log_col = first_existing(df.columns,
        ["k_pred_log","k_total_log_pred","k_measured_log_pred","y_pred_log"])

    k_true = pd.to_numeric(df[true_col], errors="coerce").to_numpy(float)
    k_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(float)

    if true_log_col and pred_log_col:
        ln_true = pd.to_numeric(df[true_log_col], errors="coerce").to_numpy(float)
        ln_pred = pd.to_numeric(df[pred_log_col], errors="coerce").to_numpy(float)
        log_source = f"stored columns: {true_log_col}, {pred_log_col}"
    else:
        if np.any((k_true <= 0) & np.isfinite(k_true)) or np.any((k_pred <= 0) & np.isfinite(k_pred)):
            sys.exit("ERROR: log columns absent and non-positive k values prevent natural-log calculation.")
        ln_true = np.log(k_true)
        ln_pred = np.log(k_pred)
        log_source = f"natural log computed from {true_col}, {pred_col}"

    n_ln, ln_mae, ln_rmse, ln_mad, ln_mad_mae, vln = metrics(ln_true, ln_pred)
    n_k,  k_mae,  k_rmse,  k_mad,  k_mad_mae,  vk  = metrics(k_true, k_pred)

    print("\nDECODED METRICS")
    print("-"*72)
    print(f"N valid      = {min(n_ln,n_k)}")
    print(f"log source   = {log_source}")
    print(f"ln-MAE       = {ln_mae:.6f}")
    print(f"ln-RMSE      = {ln_rmse:.6f}")
    print(f"ln-MAD       = {ln_mad:.6f}")
    print(f"ln MAD:MAE   = {ln_mad_mae:.6f}")
    print(f"k-MAE        = {k_mae:.6f}")
    print(f"k-RMSE       = {k_rmse:.6f}")
    print(f"k-MAD        = {k_mad:.6f}")
    print(f"k MAD:MAE    = {k_mad_mae:.6f}")

    n_valid = min(n_ln, n_k)
    if n_valid == args.expected_n:
        print(f"\nPASS: locked TEST N = {args.expected_n}")
    else:
        print(f"\nWARNING: expected N={args.expected_n}, decoded N={n_valid}")
        print("Do not use this run in RQ5 until the row-count mismatch is understood.")

    id_col = first_existing(df.columns,
        ["dataset_row_id","row_id","paper_id","id","index"])
    unique_ids = ""
    duplicate_ids = ""
    if id_col:
        ids = df.loc[vk, id_col]
        unique_ids = int(ids.nunique(dropna=False))
        duplicate_ids = int(ids.duplicated().sum())
        print(f"ID column    = {id_col}")
        print(f"Unique IDs   = {unique_ids}")
        print(f"Duplicate IDs= {duplicate_ids}")

    # per-row decoded errors
    row_out = df.copy()
    row_out["decoded_ln_abs_error"] = np.abs(ln_true - ln_pred)
    row_out["decoded_k_abs_error"] = np.abs(k_true - k_pred)
    row_path = path.with_name(f"tier3_seed_{args.seed}_decoded_row_metrics.csv")
    row_out.to_csv(row_path, index=False)

    # summary CSV: replace same-seed row if rerun
    summary_path = Path(args.summary)
    new = pd.DataFrame([{
        "seed": args.seed,
        "audit_file": str(path),
        "n_valid": n_valid,
        "ln_mae": ln_mae,
        "ln_rmse": ln_rmse,
        "ln_mad": ln_mad,
        "ln_mad_mae": ln_mad_mae,
        "k_mae": k_mae,
        "k_rmse": k_rmse,
        "k_mad": k_mad,
        "k_mad_mae": k_mad_mae,
        "id_column": id_col or "",
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "expected_n": args.expected_n,
        "n_check_pass": n_valid == args.expected_n,
        "log_source": log_source,
    }])

    if summary_path.exists():
        old = pd.read_csv(summary_path)
        if "seed" in old.columns:
            old = old[old["seed"] != args.seed]
        summary = pd.concat([old, new], ignore_index=True)
    else:
        summary = new

    summary = summary.sort_values("seed").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    print(f"\nPer-row output : {row_path}")
    print(f"Summary output : {summary_path}")

    print("\nCOPY THIS BLOCK BACK TO CHATGPT")
    print("-"*72)
    print(f"SEED = {args.seed}")
    print(f"N valid = {n_valid}")
    print(f"ln-MAE = {ln_mae:.6f}")
    print(f"ln-RMSE = {ln_rmse:.6f}")
    print(f"ln MAD:MAE = {ln_mad_mae:.6f}")
    print(f"k-MAE = {k_mae:.6f}")
    print(f"k-RMSE = {k_rmse:.6f}")
    print(f"k MAD:MAE = {k_mad_mae:.6f}")

if __name__ == "__main__":
    main()
