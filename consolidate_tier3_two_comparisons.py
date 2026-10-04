#!/usr/bin/env python3
"""Extract the two independent Tier-3 comparisons across seeds 42, 124, 777.

Requires pandas and numpy, not PyTorch/DGL/ALIGNN. No training or refitting.
Usage:
  python consolidate_tier3_two_comparisons.py --run_roots \
    tier3_baselines_seed42 tier3_baselines_seed124 tier3_baselines_seed777 \
    --output_dir tier3_paper_consolidated

Each root must contain completed reports/tier3_baselines outputs from v4.60.5
or v4.60.6. A reports/tier3_baselines directory can also be passed directly.
The donor lookup and within-checkpoint ablations are omitted from this extract.
Original evidence is read only and preserved. Seed 42 remains the paper's
primary paired analysis; all seeds contribute descriptive sensitivity results.
Three seeds evaluate the SAME 30 TEST records: they are not 90 independent
observations. Mean +/- sample SD describes initialization spread. Confidence
intervals are retained per seed, never averaged or treated as a pooled interval.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "Independently trained adapter-only": "independent_adapter_ln_prediction",
    "Independent adapter + TRAIN-fitted scalar": "adapter_scalar_ln_prediction",
    "Full process-conditioned Tier 3": "full_ln_prediction",
}
METRICS = ["ln_MAE", "ln_RMSE", "ln_MAD_to_MAE", "linear_MAE", "linear_RMSE", "linear_MAD_to_MAE"]
PAIRED = ["delta_MAE", "relative_reduction_pct", "rows_improved", "rows_worse", "rows_tied",
          "row_CI95_low", "row_CI95_high", "group_CI95_low", "group_CI95_high",
          "row_P_delta_positive", "group_P_delta_positive"]


def directory(path):
    path = Path(path)
    return path if (path / "baseline_summary.json").is_file() else path / "reports/tier3_baselines"


def load_run(root):
    folder = directory(root)
    summary = json.loads((folder / "baseline_summary.json").read_text())
    if summary.get("schema") != "v4.60.5" or summary.get("selection_partition") != "VAL" or summary.get("scalar_and_donor_fit_partition") != "experimental TRAIN":
        raise ValueError(f"Unexpected fitting/selection protocol: {folder}")
    seed = int(summary["seed"])
    metrics = pd.read_csv(folder / "table_XII_metrics.csv")
    effects = pd.read_csv(folder / "table_XIII_paired_effects.csv")
    test = pd.read_csv(folder / "test_predictions.csv", dtype={"record_id": str}).sort_values("record_id").reset_index(drop=True)
    if len(test) != 30 or test.record_id.duplicated().any() or test.condition_group_id.nunique() != 25:
        raise ValueError(f"Expected 30 unique TEST records / 25 conditions: {folder}")
    if not test.is_experimental.eq(1).all() or not test.split.eq("TEST").all():
        raise ValueError("Expected exclusively experimental TEST observations")
    metrics = metrics.loc[metrics.configuration.isin(METHODS)].copy()
    labels = [label + " -> full" for label in list(METHODS)[:2]]
    effects = effects.loc[effects.comparison.isin(labels)].copy()
    if len(metrics) != 3 or metrics.configuration.duplicated().any():
        raise ValueError("Missing/duplicate required metric rows")
    if len(effects) != 4 or effects.duplicated(["comparison", "space"]).any():
        raise ValueError("Need two comparisons, each in ln-k and k")
    if not set(effects.space).issubset({"ln-k", "k"}) or set(effects.space) != {"ln-k", "k"}:
        raise ValueError("Unexpected reporting spaces")
    if not metrics.seed.eq(seed).all() or not effects.seed.eq(seed).all():
        raise ValueError("Inconsistent seed fields")
    if not metrics.N.eq(30).all() or not effects.N.eq(30).all() or not effects.condition_groups.eq(25).all():
        raise ValueError("Inconsistent evaluation populations")
    if not effects.resamples.eq(100000).all() or not effects.bootstrap_seed.eq(2026).all():
        raise ValueError("Unexpected bootstrap protocol")
    y = test.measured_ln_k.to_numpy()
    for label, column in METHODS.items():
        prediction = test[column].to_numpy()
        if not np.isfinite(prediction).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite targets/predictions")
        log_error = prediction-y
        linear_error = np.exp(prediction)-np.exp(y)
        expected = [np.abs(log_error).mean(), np.sqrt(np.square(log_error).mean()),
                    np.abs(y-y.mean()).mean()/np.abs(log_error).mean() if np.abs(log_error).mean() else np.nan,
                    np.abs(linear_error).mean(), np.sqrt(np.square(linear_error).mean()),
                    np.abs(np.exp(y)-np.exp(y).mean()).mean()/np.abs(linear_error).mean() if np.abs(linear_error).mean() else np.nan]
        actual = metrics.loc[metrics.configuration.eq(label), METRICS].iloc[0].to_numpy(dtype=float)
        if not np.allclose(actual, expected, atol=2e-6, rtol=1e-7, equal_nan=True):
            raise ValueError(f"Metrics do not reproduce predictions: seed {seed}, {label}")
    full = test.full_ln_prediction.to_numpy()
    for label, column in list(METHODS.items())[:2]:
        for space in ("ln-k", "k"):
            target, base, prediction = y, test[column].to_numpy(), full
            if space == "k":
                target, base, prediction = np.exp(target), np.exp(base), np.exp(prediction)
            difference = np.abs(base-target)-np.abs(prediction-target)
            row = effects.loc[effects.comparison.eq(label+" -> full") & effects.space.eq(space)].iloc[0]
            if not np.isclose(row.delta_MAE, difference.mean(), atol=2e-6, rtol=1e-7):
                raise ValueError("Paired delta does not reproduce predictions")
            if row.rows_improved + row.rows_worse + row.rows_tied != 30:
                raise ValueError("Incorrect improved/worse/tied counts")
    selection = {"seed": seed, "selected_adapter_epoch": summary["selected_adapter_epoch"],
        "selected_adapter_VAL_ln_MAE": summary["selected_adapter_VAL_ln_MAE"],
        "selected_full_epoch": summary["selected_full_epoch"],
        "selected_full_VAL_ln_MAE": summary["selected_full_VAL_ln_MAE"],
        "stopping_epoch": summary["stopping_epoch"], "scalar_ln_offset": summary["scalar_ln_offset"],
        "frozen_non_adapter_max_change": summary["frozen_non_adapter_max_change"],
        "source_directory": str(folder.resolve())}
    return seed, metrics, effects, test, selection


def markdown(frame):
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    rows = ["| " + " | ".join(map(cell, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def consolidate(roots, output_dir):
    runs = sorted([load_run(root) for root in roots], key=lambda run: run[0])
    if [r[0] for r in runs] != [42, 124, 777]:
        raise ValueError("Exactly one completed run each for seeds 42, 124 and 777 is required")
    reference = runs[0][3]
    fixed_fields = ["record_id", "donor_key", "atoms_sha256", "condition_group_id"]
    for _, _, _, test, _ in runs[1:]:
        if not reference[fixed_fields].equals(test[fixed_fields]):
            raise ValueError("TEST identities/donors/structures/conditions differ across seeds")
        for column in ("measured_ln_k", "prior_ln_prediction"):
            if not np.allclose(reference[column], test[column], rtol=0, atol=2e-6):
                raise ValueError("Targets or frozen prior differ across seeds")
    output = Path(output_dir)
    input_dirs = [directory(root).resolve() for root in roots]
    if any(output.resolve() == root or root in output.resolve().parents for root in input_dirs):
        raise ValueError("Consolidation output must be separate from source evidence directories")
    metrics = pd.concat([r[1] for r in runs], ignore_index=True)
    effects = pd.concat([r[2] for r in runs], ignore_index=True)
    mean_rows = []
    for label in METHODS:
        subset = metrics.loc[metrics.configuration.eq(label)]
        row = {"configuration": label, "N_TEST":30, "seeds":3}
        for metric in METRICS:
            row[metric+"_mean"] = float(subset[metric].mean())
            row[metric+"_sample_SD"] = float(subset[metric].std(ddof=1))
        mean_rows.append(row)
    averages = pd.DataFrame(mean_rows)
    seed_summary = []
    for (label, space), subset in effects.groupby(["comparison", "space"], sort=False):
        seed_summary.append({"comparison":label, "space":space,
            "delta_MAE_mean":subset.delta_MAE.mean(), "delta_MAE_sample_SD":subset.delta_MAE.std(ddof=1),
            "delta_MAE_min":subset.delta_MAE.min(), "delta_MAE_max":subset.delta_MAE.max(),
            "seeds_positive_delta":int(subset.delta_MAE.gt(0).sum()),
            "seeds_positive_row_interval":int(subset.row_CI95_low.gt(0).sum()),
            "seeds_positive_group_interval":int(subset.group_CI95_low.gt(0).sum())})
    sensitivity = pd.DataFrame(seed_summary)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "table_XII_three_methods_by_seed.csv", index=False)
    averages.to_csv(output / "table_XII_three_seed_mean_SD.csv", index=False)
    effects.to_csv(output / "table_XIII_two_comparisons_by_seed.csv", index=False)
    sensitivity.to_csv(output / "RQ5_comparison_seed_sensitivity.csv", index=False)
    pd.DataFrame([r[4] for r in runs]).to_csv(output / "selection_and_scalar_audit.csv", index=False)
    display = pd.DataFrame({"Configuration":averages.configuration})
    for metric, label in [("ln_MAE","ln-MAE"),("ln_RMSE","ln-RMSE"),("linear_MAE","k-MAE"),("linear_RMSE","k-RMSE")]:
        display[label] = [f"{m:.4f} ± {sd:.4f}" for m, sd in zip(averages[metric+"_mean"], averages[metric+"_sample_SD"])]
    primary = effects.loc[effects.seed.eq(42) & effects.space.eq("ln-k")].copy()
    primary_display = pd.DataFrame({"Comparison":primary.comparison,
        "Delta ln-MAE":primary.delta_MAE.map(lambda v:f"{v:+.6f}"),
        "Reduction":primary.relative_reduction_pct.map(lambda v:f"{v:.2f}%"),
        "Row 95% CI":[f"[{a:+.6f}, {b:+.6f}]" for a,b in zip(primary.row_CI95_low,primary.row_CI95_high)],
        "Condition-group 95% CI":[f"[{a:+.6f}, {b:+.6f}]" for a,b in zip(primary.group_CI95_low,primary.group_CI95_high)]})
    full_row = averages.loc[averages.configuration.eq("Full process-conditioned Tier 3")].iloc[0]
    extract = "# Tier 3 results for manuscript updates\n\n"
    extract += "Three methods on the same 30 TEST records. Values below are mean ± sample standard deviation across seeds 42, 124 and 777; this spread is not a population confidence interval.\n\n"
    extract += markdown(display) + "\n\nSeed 42 remains the primary paired analysis. Positive Delta MAE favors the full model. The two comparisons are independently trained adapter-only versus full and adapter plus a scalar fitted on experimental TRAIN versus full.\n\n"
    extract += markdown(primary_display) + "\n\n"
    extract += f"Abstract numerical extract: full-model ln-MAE = {full_row.ln_MAE_mean:.4f} ± {full_row.ln_MAE_sample_SD:.4f} (mean ± sample s.d.; three seeds; N = 30). Report superiority only where the corresponding retained paired interval supports it.\n\n"
    extract += "Update Table XII with the three-method metrics; Table XIII with seed-42 paired effects; RQ5 with per-seed effects and initialization spread; Methods with independent training and TRAIN-only scalar fitting. Update the Abstract, RQ3, RQ4 and Conclusions to match the actual effects. The scalar is a median residual in natural-log space, applied to the independently trained adapter. No separate neural training is needed for it.\n\n"
    extract += "Donor lookup is outside this focused extract. These comparisons do not exclude donor/provenance confounding or establish causality. Preserve all original audit and prediction files. Do not pool the three seeds into 90 independent TEST records, average CI endpoints, or select the best TEST seed.\n"
    (output / "paper_extract.md").write_text(extract, encoding="utf-8")
    print("Consolidation verified: 3 seeds, identical 30 TEST records, 25 condition groups.")
    print("Required manuscript values:", (output / "paper_extract.md").resolve())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_roots", nargs=3, required=True)
    parser.add_argument("--output_dir", default="tier3_paper_consolidated")
    args = parser.parse_args()
    consolidate(args.run_roots, args.output_dir)


if __name__ == "__main__":
    main()
