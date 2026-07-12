#!/usr/bin/env python3
"""Tier-3 frozen-base integrity summary generator (v4.44).

Consumes artifacts from the v4.41 training script and refuses to treat normal
versus ablation metrics as comparable when row counts or row identities differ.
It also verifies that the structural/base prediction is invariant across
checkpoints and that final_prediction = base_prediction + applied_delta row by
row. These checks directly diagnose residual-only training integrity.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: unable to read {path}: {exc}")
        return None


def _stage_from_name(path: Path) -> str:
    m = re.match(r"tier3_(.+?)_(test|val)_prediction_audit\.csv$", path.name)
    return m.group(1) if m else path.stem


def _identity_series(df: pd.DataFrame) -> pd.Series:
    if "split_identity" in df.columns:
        return df["split_identity"].fillna("").astype(str)
    # Backward-compatible fallback for legacy files.
    cols = [c for c in ["row_id", "paper_id", "doi", "material", "formula", "k_true"] if c in df.columns]
    if not cols:
        return pd.Series([str(i) for i in range(len(df))], index=df.index)
    return df[cols].fillna("").astype(str).agg("|".join, axis=1)


def _metrics(df: pd.DataFrame, subgroup: str, mask: pd.Series) -> dict:
    g = df.loc[mask].copy()
    if g.empty:
        return {"subgroup": subgroup, "n": 0}
    true_log = pd.to_numeric(g.get("k_true_log"), errors="coerce")
    pred_log = pd.to_numeric(g.get("k_pred_log"), errors="coerce")
    true_k = pd.to_numeric(g.get("k_true"), errors="coerce")
    pred_k = pd.to_numeric(g.get("k_pred"), errors="coerce")
    e_log = (pred_log - true_log).abs().dropna()
    e_k = (pred_k - true_k).dropna()
    mae_k = float(e_k.abs().mean()) if len(e_k) else float("nan")
    rmse_k = float(np.sqrt(np.mean(np.square(e_k)))) if len(e_k) else float("nan")
    valid_true = true_k.dropna()
    mad = float(np.mean(np.abs(valid_true - valid_true.mean()))) if len(valid_true) else float("nan")
    raw = pd.to_numeric(g.get("raw_process_delta", 0), errors="coerce")
    bounded = pd.to_numeric(g.get("bounded_process_delta", 0), errors="coerce")
    return {
        "subgroup": subgroup,
        "n": int(len(g)),
        "unique_rows": int(_identity_series(g).nunique()),
        "log_mae": float(e_log.mean()) if len(e_log) else float("nan"),
        "linear_mae": mae_k,
        "linear_rmse": rmse_k,
        "mad": mad,
        "mad_mae": mad / mae_k if np.isfinite(mad) and mae_k > 0 else float("nan"),
        "mean_raw_delta": float(raw.mean()),
        "mean_abs_bounded_delta": float(bounded.abs().mean()),
        "max_abs_bounded_delta": float(bounded.abs().max()),
        "mean_feature_count": float(pd.to_numeric(g.get("process_feature_count", 0), errors="coerce").mean()),
        "proc_available_pct": float(100.0 * pd.to_numeric(g.get("proc_avail", 0), errors="coerce").fillna(0).gt(0).mean()),
    }


def summarize_prediction_file(path: Path, run_type: str) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()
    is_exp = pd.to_numeric(df.get("is_experimental", 0), errors="coerce").fillna(0).astype(bool)
    is_imp = df.get("imputed_structure", False)
    if not isinstance(is_imp, pd.Series):
        is_imp = pd.Series(False, index=df.index)
    is_imp = is_imp.astype(str).str.lower().isin(["true", "1", "yes"])
    proc = pd.to_numeric(df.get("proc_avail", 0), errors="coerce").fillna(0).gt(0)
    groups = {
        "all": pd.Series(True, index=df.index),
        "experimental": is_exp,
        "native_dft": ~is_exp,
        "imputed_experimental": is_exp & is_imp,
        "proc_available": proc,
        "proc_unavailable": ~proc,
        "experimental_proc_available": is_exp & proc,
    }
    stage = _stage_from_name(path)
    split = "test" if "_test_" in path.name else "val"
    rows = []
    for name, mask in groups.items():
        row = _metrics(df, name, mask)
        row.update({"run_type": run_type, "stage": stage, "split": split, "source_file": str(path)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_history(project_root: Path, run_type: str) -> pd.DataFrame:
    path = project_root / "reports" / f"tier3_training_history_{run_type}.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.DataFrame(json.loads(path.read_text()))
        df.insert(0, "run_type", run_type)
        return df
    except Exception as exc:
        print(f"WARNING: unable to parse {path}: {exc}")
        return pd.DataFrame()


def optimizer_summary(audit_dir: Path, run_type: str) -> pd.DataFrame:
    path = audit_dir / "tier3_optimizer_parameter_manifest.csv"
    df = _safe_read_csv(path) if path.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    residual = df[df["parameter"].astype(str).str.startswith("process_delta_head")]
    return pd.DataFrame([{
        "run_type": run_type,
        "residual_parameter_count": int(residual["numel"].sum()),
        "all_residual_requires_grad": bool(residual["requires_grad"].astype(bool).all()) if len(residual) else False,
        "all_residual_in_optimizer": bool(residual["in_optimizer"].astype(bool).all()) if len(residual) else False,
        "residual_lr_values": ",".join(sorted(set(residual["optimizer_lr"].astype(str)))) if len(residual) else "",
    }])


def composition_summary(audit_dir: Path, run_type: str) -> pd.DataFrame:
    path = audit_dir / "tier3_dataset_composition.csv"
    df = _safe_read_csv(path) if path.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for split, g in df.groupby("split", dropna=False):
        rows.append({
            "run_type": run_type,
            "split": split,
            "n": int(len(g)),
            "experimental": int(pd.to_numeric(g["is_experimental"], errors="coerce").fillna(0).sum()),
            "imputed": int(g["imputed_structure"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
            "proc_available": int(pd.to_numeric(g["proc_avail"], errors="coerce").fillna(0).gt(0).sum()),
            "target_valid": int(g["target_valid"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
        })
    return pd.DataFrame(rows)


def frozen_base_integrity(base: Path) -> pd.DataFrame:
    """Compare every checkpoint audit to epoch0 on identical row identities."""
    rows = []
    tol = 1e-6
    for run_type in ("normal", "ablation"):
        run_dir = base / run_type
        for split in ("test", "val"):
            epoch0_path = run_dir / f"tier3_epoch0_pretrain_{split}_prediction_audit.csv"
            epoch0 = _safe_read_csv(epoch0_path) if epoch0_path.exists() else None
            if epoch0 is None or epoch0.empty:
                continue
            epoch0 = epoch0.copy()
            epoch0["_identity"] = _identity_series(epoch0)
            if epoch0["_identity"].duplicated().any():
                continue
            epoch0 = epoch0.set_index("_identity")
            base0 = pd.to_numeric(epoch0.get("k_dft_log"), errors="coerce")
            pred0 = pd.to_numeric(epoch0.get("k_pred_log"), errors="coerce")
            for path in sorted(run_dir.glob(f"tier3_*_{split}_prediction_audit.csv")):
                stage = _stage_from_name(path)
                cur = _safe_read_csv(path)
                if cur is None or cur.empty:
                    continue
                cur = cur.copy()
                cur["_identity"] = _identity_series(cur)
                cur = cur.set_index("_identity")
                common = epoch0.index.intersection(cur.index)
                if len(common) == 0:
                    continue
                cur_base = pd.to_numeric(cur.loc[common].get("k_dft_log"), errors="coerce")
                cur_pred = pd.to_numeric(cur.loc[common].get("k_pred_log"), errors="coerce")
                cur_delta = pd.to_numeric(cur.loc[common].get("delta_log_pred", 0.0), errors="coerce").fillna(0.0)
                base_ref = base0.loc[common]
                pred_ref = pred0.loc[common]
                base_change = cur_base - base_ref
                pred_change = cur_pred - pred_ref
                residual_movement_error = pred_change - cur_delta
                if "prediction_decomposition_error" in cur.columns:
                    decomposition = pd.to_numeric(cur.loc[common, "prediction_decomposition_error"], errors="coerce")
                else:
                    decomposition = cur_pred - (cur_base + cur_delta)
                rows.append({
                    "run_type": run_type,
                    "stage": stage,
                    "split": split,
                    "n_common": int(len(common)),
                    "max_abs_base_drift": float(base_change.abs().max()),
                    "mean_abs_base_drift": float(base_change.abs().mean()),
                    "base_rows_over_tolerance": int(base_change.abs().gt(tol).sum()),
                    "max_abs_prediction_decomposition_error": float(decomposition.abs().max()),
                    "mean_abs_prediction_decomposition_error": float(decomposition.abs().mean()),
                    "max_abs_prediction_movement_minus_delta": float(residual_movement_error.abs().max()),
                    "mean_abs_prediction_movement_minus_delta": float(residual_movement_error.abs().mean()),
                    "base_frozen_pass": bool(base_change.abs().max() <= tol),
                    "decomposition_pass": bool(decomposition.abs().max() <= tol),
                    "movement_explained_by_delta_pass": bool(residual_movement_error.abs().max() <= tol),
                    "source_file": str(path),
                })
    return pd.DataFrame(rows)

def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    keys = ["stage", "split", "subgroup"]
    cols = ["n", "unique_rows", "log_mae", "linear_mae", "linear_rmse", "mad_mae", "mean_abs_bounded_delta"]
    pivot = metrics.pivot_table(index=keys, columns="run_type", values=cols, aggfunc="first")
    pivot.columns = [f"{metric}_{run}" for metric, run in pivot.columns]
    pivot = pivot.reset_index()
    if "linear_mae_normal" in pivot and "linear_mae_ablation" in pivot:
        pivot["mae_improvement_normal_vs_ablation"] = pivot["linear_mae_ablation"] - pivot["linear_mae_normal"]
        pivot["mae_improvement_pct"] = 100.0 * pivot["mae_improvement_normal_vs_ablation"] / pivot["linear_mae_ablation"].replace(0, np.nan)
    if "log_mae_normal" in pivot and "log_mae_ablation" in pivot:
        pivot["log_mae_improvement"] = pivot["log_mae_ablation"] - pivot["log_mae_normal"]
    return pivot


def integrity_checks(base: Path, metrics: pd.DataFrame, history: pd.DataFrame, freeze_integrity: pd.DataFrame) -> pd.DataFrame:
    checks = []
    # Every VAL/TEST prediction file within each run must have one common count.
    for run_type in ("normal", "ablation"):
        run_dir = base / run_type
        files = sorted(run_dir.glob("tier3_*_test_prediction_audit.csv")) + sorted(run_dir.glob("tier3_*_val_prediction_audit.csv"))
        counts = {}
        identities = {}
        for p in files:
            df = _safe_read_csv(p)
            if df is None:
                continue
            counts[p.name] = len(df)
            identities[p.name] = set(_identity_series(df).tolist())
            checks.append({"check": f"unique_rows:{run_type}:{p.name}", "passed": len(df) == len(identities[p.name]), "detail": f"rows={len(df)} unique={len(identities[p.name])}"})
        if counts:
            common = len(set(counts.values())) == 1
            checks.append({"check": f"constant_stage_count:{run_type}", "passed": common, "detail": str(counts)})
            id_sets = list(identities.values())
            same_ids = all(x == id_sets[0] for x in id_sets[1:]) if id_sets else True
            checks.append({"check": f"same_rows_all_stages:{run_type}", "passed": same_ids, "detail": f"files={len(id_sets)}"})

    # Normal and ablation must use the same row identities at matching stages.
    normal_dir, abl_dir = base / "normal", base / "ablation"
    for npth in sorted(normal_dir.glob("tier3_*_test_prediction_audit.csv")):
        apth = abl_dir / npth.name
        if not apth.exists():
            continue
        nd, ad = _safe_read_csv(npth), _safe_read_csv(apth)
        if nd is None or ad is None:
            continue
        same = set(_identity_series(nd)) == set(_identity_series(ad))
        checks.append({"check": f"normal_vs_ablation_rows:{npth.name}", "passed": same, "detail": f"normal={len(nd)} ablation={len(ad)}"})

    # v4.44: frozen base and prediction decomposition must hold at every stage.
    if not freeze_integrity.empty:
        for _, r in freeze_integrity.iterrows():
            tag = f"{r['run_type']}:{r['stage']}:{r['split']}"
            checks.append({
                "check": f"frozen_base_prediction:{tag}",
                "passed": bool(r["base_frozen_pass"]),
                "detail": f"max_abs_drift={float(r['max_abs_base_drift']):.3e} rows_over_tol={int(r['base_rows_over_tolerance'])}",
            })
            checks.append({
                "check": f"prediction_decomposition:{tag}",
                "passed": bool(r["decomposition_pass"]),
                "detail": f"max_abs_error={float(r['max_abs_prediction_decomposition_error']):.3e}",
            })
            checks.append({
                "check": f"prediction_movement_equals_delta:{tag}",
                "passed": bool(r["movement_explained_by_delta_pass"]),
                "detail": f"max_abs_error={float(r['max_abs_prediction_movement_minus_delta']):.3e}",
            })

    # Frozen ablation validation should be constant across epochs.
    ah = history[history.get("run_type", pd.Series(dtype=str)).eq("ablation")] if not history.empty and "run_type" in history else pd.DataFrame()
    if not ah.empty and "val_mae" in ah:
        vals = pd.to_numeric(ah["val_mae"], errors="coerce").dropna()
        span = float(vals.max() - vals.min()) if len(vals) else float("nan")
        checks.append({"check": "frozen_ablation_val_deterministic", "passed": bool(len(vals) == 0 or span <= 1e-7), "detail": f"val_mae_span={span:.3e}"})
    return pd.DataFrame(checks)


def write_markdown(path: Path, integrity: pd.DataFrame, metrics: pd.DataFrame,
                   comparison: pd.DataFrame, history: pd.DataFrame,
                   optimizer: pd.DataFrame, composition: pd.DataFrame,
                   freeze_integrity: pd.DataFrame) -> None:
    lines: List[str] = ["# Tier-3 Frozen-Base Process-Residual Audit Summary v4.44", ""]
    if not integrity.empty:
        lines += ["## Audit integrity checks", "", integrity.to_markdown(index=False), ""]
        if not bool(integrity["passed"].all()):
            lines += ["**STOP:** One or more integrity checks failed. Do not interpret normal-versus-ablation performance until all checks pass.", ""]
    if not optimizer.empty:
        lines += ["## Optimizer wiring", "", optimizer.to_markdown(index=False), ""]
    if not composition.empty:
        lines += ["## Dataset composition", "", composition.to_markdown(index=False), ""]
    if not freeze_integrity.empty:
        keep_f = [c for c in [
            "run_type", "stage", "split", "n_common",
            "max_abs_base_drift", "base_rows_over_tolerance",
            "max_abs_prediction_decomposition_error",
            "max_abs_prediction_movement_minus_delta",
            "base_frozen_pass", "decomposition_pass",
            "movement_explained_by_delta_pass",
        ] if c in freeze_integrity.columns]
        lines += ["## Frozen-base and prediction-decomposition integrity", "",
                  freeze_integrity[keep_f].to_markdown(index=False), ""]
    if not comparison.empty:
        focus = comparison[(comparison["split"] == "test") & comparison["subgroup"].isin(["all", "experimental_proc_available", "imputed_experimental"])]
        lines += ["## Normal vs ablation", "", focus.to_markdown(index=False), ""]
    if not history.empty:
        keep = [c for c in ["run_type", "epoch", "val_mae", "val_rmse", "process_delta_grad_norm", "process_delta_parameter_norm", "process_delta_parameter_change_from_init", "process_delta_mean_abs", "process_delta_max_abs", "frozen_base_max_abs_change", "frozen_base_l2_change", "frozen_base_changed_tensors"] if c in history.columns]
        lines += ["## Training trajectory", "", history[keep].to_markdown(index=False), ""]
    lines += ["## Interpretation gate", "",
              "Only interpret process-context value when every audit integrity check above passes.",
              "The decisive subgroup is `experimental_proc_available` on the fixed TEST rows.", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("highk_project"))
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    root = args.project_root
    base = root / "reports" / "tier3_prediction_audit"
    out = args.output_dir or (base / "summary")
    out.mkdir(parents=True, exist_ok=True)

    metric_frames, history_frames, opt_frames, comp_frames = [], [], [], []
    for run_type in ("normal", "ablation"):
        run_dir = base / run_type
        files = sorted(run_dir.glob("tier3_*_test_prediction_audit.csv")) + sorted(run_dir.glob("tier3_*_val_prediction_audit.csv"))
        for path in files:
            frame = summarize_prediction_file(path, run_type)
            if not frame.empty:
                metric_frames.append(frame)
        for frame, bucket in [(load_history(root, run_type), history_frames),
                              (optimizer_summary(run_dir, run_type), opt_frames),
                              (composition_summary(run_dir, run_type), comp_frames)]:
            if not frame.empty:
                bucket.append(frame)

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    optimizer = pd.concat(opt_frames, ignore_index=True) if opt_frames else pd.DataFrame()
    composition = pd.concat(comp_frames, ignore_index=True) if comp_frames else pd.DataFrame()
    comparison = build_comparison(metrics)
    freeze_integrity = frozen_base_integrity(base)
    integrity = integrity_checks(base, metrics, history, freeze_integrity)

    metrics.to_csv(out / "tier3_all_stage_subgroup_metrics.csv", index=False)
    comparison.to_csv(out / "tier3_normal_vs_ablation_comparison.csv", index=False)
    history.to_csv(out / "tier3_training_trajectory.csv", index=False)
    optimizer.to_csv(out / "tier3_optimizer_summary.csv", index=False)
    composition.to_csv(out / "tier3_dataset_composition_summary.csv", index=False)
    integrity.to_csv(out / "tier3_audit_integrity_checks.csv", index=False)
    freeze_integrity.to_csv(out / "tier3_frozen_base_integrity.csv", index=False)
    write_markdown(out / "tier3_audit_summary.md", integrity, metrics, comparison, history, optimizer, composition, freeze_integrity)

    print(f"Audit summary written to: {out}")
    if metrics.empty:
        print("WARNING: no prediction-audit CSV files found.")
        return 2
    if not integrity.empty and not bool(integrity["passed"].all()):
        print("ERROR: audit integrity checks failed; inspect tier3_audit_integrity_checks.csv")
        return 3
    print("All audit integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
