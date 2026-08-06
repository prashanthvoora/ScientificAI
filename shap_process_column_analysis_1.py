#!/usr/bin/env python3
"""
shap_process_column_analysis.py
================================
Step 2 of the SHAP process-column analysis.

Goal: decide which columns from the FULL process DB (process_db_clean.csv)
should be wired into ProcessParamsEncoder / PROCESS_PARAMS_FEATURES, using a
methodology defensible enough to actually change that schema on. This means:

  1. TRAIN-ONLY ROWS. Only tier3_split=="train" rows (tagged by
     generate_backbone_predictions.py from the LOCKED disjoint split files)
     are used. VAL/TEST rows never touch feature discovery -- letting them in
     would be the same class of leakage as the VAL==TEST bug already fixed
     for MAD:MAE reporting, one step earlier in the pipeline.

  2. OUT-OF-BAG SHAP. Every SHAP value is computed on rows the surrogate did
     NOT see when it was fit. Bootstrap resampling happens at the GROUP level
     (material_family by default) so the same material can't sit on both the
     in-bag and "held-out" side -- the same disjointness concern as your
     donor_material group-mode issue.

  3. REVIEWED ALLOWLIST. Candidate columns come ONLY from a CSV you provide
     and have reviewed (--allowlist column,type). There is no automatic
     blacklist-based inclusion. --suggest_allowlist will print a starting
     point, clearly labeled as suggestions to review, never auto-applied.

  4. STABILITY, NOT CI>0. Column selection uses a Boruta-style shadow-feature
     null test (does the real column beat a freshly-reshuffled noise twin of
     itself more often than chance, binomial-tested?) plus rank stability
     (median OOB rank + top-K selection frequency across bootstrap
     iterations), plus a leave-one-out grouped-CV ablation check on whatever
     the null test confirms.

CAVEAT: Boruta-on-boosted-trees is weaker when the candidate set is small
(<~6 columns) -- a single shadow column can go entirely unused by a greedy
booster, making shadow_max an artificially low bar. This script mitigates it
with multiple shadow copies per column (--n_shadow_copies) and lower
colsample_bytree/bynode to force feature rotation, plus a practical hit-rate
floor (--min_hit_rate) on top of the binomial p-value -- but with a very
short allowlist, treat "confirmed" verdicts with extra scrutiny and lean
harder on the ablation check.

SCOPE: this script's output is a recommended COLUMN SET for
PROCESS_PARAMS_FEATURES only. It does not and should not inform embedding
dimension, hidden width, learning rate, or process_delta_bound -- those stay
validation-stage hyperparameters, tuned separately after the schema is fixed.

USAGE:
    # one-time: build a starting point for your allowlist, then review/edit it
    python shap_process_column_analysis.py \\
        --full_db process_db_clean.csv --backbone_preds backbone_predictions_full_db.csv \\
        --suggest_allowlist --out_dir shap_results

    # main run
    python shap_process_column_analysis.py \\
        --full_db process_db_clean.csv \\
        --backbone_preds backbone_predictions_full_db.csv \\
        --allowlist reviewed_process_column_allowlist.csv \\
        --training_script_dir /path/to/training/script/dir \\
        --out_dir shap_results

Requires: pip install xgboost shap scikit-learn scipy matplotlib --break-system-packages
(xgboost >= 1.6 for native pandas-categorical support)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Columns that can NEVER be a feature regardless of what the allowlist says --
# targets, target aliases, and the join key. This is a safety net UNDER the
# allowlist, not a substitute for it: it only ever narrows, never expands,
# the candidate set.
HARD_LEAK_GUARD = {
    "row_id", "paper_id", "doi",
    "k_dielectric_constant", "dielectric_constant_k", "dielectric_k",
    "k_total", "k_measured", "k_total_log", "k_measured_log", "epsilon_total",
    "band_gap_eV", "band_gap", "Eg_eV",
    "J_g_A_cm2_at_1V", "J_g_A_cm2_at_1v", "leakage_J_A_cm2_at_field",
    "leakage_current_J_A_cm2", "J_g", "J_g_A_cm2", "J_g_log",
    "breakdown_field_MV_cm", "E_breakdown_MV_cm", "E_BD", "E_BD_MV_cm",
    "k_total_log_true", "k_dft_log_backbone", "residual_log", "tier3_split",
}


# --------------------------------------------------------------------------
# Loading / merging / split filtering
# --------------------------------------------------------------------------

def load_full_db(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded full process DB: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_backbone_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded backbone predictions: {df.shape[0]} rows")
    if "tier3_split" not in df.columns:
        raise KeyError(
            "backbone_predictions_full_db.csv has no 'tier3_split' column -- "
            "regenerate it with the current generate_backbone_predictions.py "
            "(it tags every row train/val/test from the locked split files)."
        )
    return df


def merge_on_row_id(df_full: pd.DataFrame, df_pred: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    if id_column:
        key = id_column
    elif "row_id" in df_full.columns:
        key = "row_id"
    elif "paper_id" in df_full.columns:
        key = "paper_id"
    else:
        raise KeyError(
            "Could not find a 'row_id' or 'paper_id' column in the full DB to "
            "join against backbone_predictions_full_db.csv. Pass --id_column."
        )
    df_full = df_full.copy()
    df_full["_join_key"] = df_full[key].astype(str).str.strip()
    df_pred = df_pred.copy()
    df_pred["_join_key"] = df_pred["paper_id"].astype(str).str.strip()

    merged = df_full.merge(
        df_pred[["_join_key", "k_total_log_true", "k_dft_log_backbone",
                 "residual_log", "tier3_split", "material_family"]],
        on="_join_key", how="inner", suffixes=("", "_pred"),
    )
    print(f"Merged on '{key}': {len(merged)}/{len(df_full)} full-DB rows matched "
          f"to a backbone prediction")

    if len(merged) < 0.5 * len(df_full):
        full_keys = set(df_full["_join_key"])
        pred_keys = set(df_pred["_join_key"])
        overlap = full_keys & pred_keys
        print("WARNING: fewer than half the full-DB rows matched. Check that the "
              f"join key values line up between process_db_clean.csv's '{key}' "
              "column and backbone_predictions_full_db.csv's 'paper_id' column.")
        print(f"  full_db unique keys: {len(full_keys)}   "
              f"predictions unique keys: {len(pred_keys)}   overlap: {len(overlap)}")
        print(f"  sample full_db keys (first 5):     {list(df_full['_join_key'].head(5))}")
        print(f"  sample predictions keys (first 5): {list(df_pred['_join_key'].head(5))}")
        if overlap:
            print(f"  {len(overlap)} keys DO match exactly -- if that's far fewer than "
                  "expected, some rows likely differ only in formatting (e.g. "
                  "'1024' vs '1024.0' from pandas int/float coercion, leading zeros, "
                  "case, or extra whitespace) rather than being genuinely different IDs.")
        else:
            print("  ZERO keys match at all. Look closely at the two sample lists above "
                  "-- if one side has a trailing '.0' (e.g. '1024.0' vs '1024'), a decimal "
                  "reformat, leading zeros, or case differences, it's a formatting "
                  "mismatch (very common when an ID column round-trips through pandas as "
                  "float). If the samples look like genuinely different ID schemes, "
                  "double-check what load_experimental_process_db() renamed 'row_id' "
                  "to/from in your training script version, and whether "
                  "process_db_clean.csv's id column really is 'row_id' -- if it's a "
                  "different name, pass --id_column explicitly.")
    if len(merged) == 0:
        raise RuntimeError(
            "Join produced 0 rows -- refusing to continue (every downstream step, "
            "including --suggest_allowlist, would fail or silently produce nothing "
            "useful). Fix the join key per the diagnostics above and rerun."
        )
    return merged


def filter_to_train_split(merged: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """
    tier3_split values now include "train_excluded_leak_adjacent" (rows that
    share paper/material/donor identity with a locked VAL/TEST row -- see
    generate_backbone_predictions.py's exclude_leak_adjacent_train_rows()).
    Filtering on the exact string "train" naturally excludes those rows too.
    """
    counts = merged["tier3_split"].value_counts().to_dict()
    print(f"tier3_split composition in merged data: {counts}")
    out = merged[merged["tier3_split"] == split].copy()
    print(f"Restricting SHAP feature discovery to tier3_split=='{split}': "
          f"{len(out)}/{len(merged)} rows kept")
    if len(out) < 20:
        print(f"WARNING: only {len(out)} train rows available for SHAP -- "
              "bootstrap/OOB estimates below will be noisy regardless of the "
              "stability tests. Treat results as a weak prior.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Allowlist (replaces the old auto-blacklist)
# --------------------------------------------------------------------------

def load_allowlist(path: Path) -> tuple[list, list]:
    df = pd.read_csv(path)
    required = {"column", "type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Allowlist file must have columns {required}, got {list(df.columns)}")
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    bad = set(df["type"]) - {"numeric", "categorical"}
    if bad:
        raise ValueError(f"Allowlist 'type' values must be 'numeric' or 'categorical', got: {bad}")
    leaked = (set(df["column"]) & HARD_LEAK_GUARD)
    if leaked:
        raise ValueError(
            f"Allowlist contains target/id/join columns that can never be "
            f"features: {sorted(leaked)}. Remove them from the allowlist."
        )
    numeric_cols = df.loc[df["type"] == "numeric", "column"].tolist()
    categorical_cols = df.loc[df["type"] == "categorical", "column"].tolist()
    print(f"Allowlist loaded: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
    return numeric_cols, categorical_cols


def suggest_allowlist(df: pd.DataFrame, out_dir: Path):
    """
    Heuristic candidate scan, written to a template CSV for manual review.
    NEVER used automatically to select features -- it only helps you write
    the allowlist faster.
    """
    rows = []
    for col in df.columns:
        if col in HARD_LEAK_GUARD or col == "_join_key":
            continue
        s = df[col]
        if s.isna().all():
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1:
                continue
            rows.append({"column": col, "type": "numeric",
                         "non_null": int(s.notna().sum()), "n_unique": int(s.nunique(dropna=True))})
        else:
            n_unique = s.astype(str).nunique(dropna=True)
            guess = "categorical" if n_unique <= max(20, int(0.5 * len(df))) else "SKIP_likely_free_text_or_id"
            rows.append({"column": col, "type": guess,
                         "non_null": int(s.notna().sum()), "n_unique": int(n_unique)})
    template = pd.DataFrame(rows)
    if template.empty:
        raise RuntimeError(
            "No candidate columns found to suggest -- 'merged' had too few usable "
            "rows/columns to scan. This is almost always downstream of the join-key "
            "warning printed above (merge_on_row_id): if the join matched 0 or very "
            "few rows, every column looks empty here. Fix the join key mismatch "
            "first (see the diagnostics printed by merge_on_row_id), then rerun "
            "--suggest_allowlist."
        )
    template = template.sort_values("column")
    out_path = out_dir / "allowlist_template.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_path, index=False)
    print(f"\nWrote SUGGESTED (not applied) allowlist template to {out_path}")
    print("Review it: delete rows that shouldn't be candidates, fix any "
          "'SKIP_likely_free_text_or_id' guesses, then pass the reviewed file "
          "as --allowlist to run the actual analysis.")


# --------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, numeric_cols, categorical_cols, group_col: str):
    missing = [c for c in numeric_cols + categorical_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Allowlisted columns not found in merged data: {missing}")
    if group_col not in df.columns:
        raise KeyError(f"--group_col '{group_col}' not found in merged data.")

    X = df[numeric_cols + categorical_cols].copy()
    for c in categorical_cols:
        X[c] = X[c].astype(str).replace({"nan": np.nan}).astype("category")
    y = df["residual_log"].astype(float)
    groups = df[group_col].astype(str)

    mask = y.notna()
    X, y, groups = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), groups[mask].reset_index(drop=True)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns; "
          f"target = residual_log; groups = {groups.nunique()} unique '{group_col}' values")
    return X, y, groups


# --------------------------------------------------------------------------
# Grouped CV sanity check (surrogate must be able to predict SOMETHING
# before its SHAP explanations mean anything)
# --------------------------------------------------------------------------

def cross_validated_r2(X, y, groups, n_splits=5, seed=42):
    """
    GroupKFold only -- never falls back to plain KFold. If there aren't
    enough distinct groups for the requested n_splits, n_splits is reduced
    to the number of available groups (still strictly grouped); CV is
    refused entirely below 2 groups rather than silently ungrouping, which
    would let the same material sit in both a train and a test fold.
    """
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score

    n_groups = groups.nunique()
    if n_groups < 2:
        raise RuntimeError(
            f"Only {n_groups} distinct group(s) in the grouping column -- "
            "grouped CV requires at least 2. Refusing to fall back to "
            "ordinary KFold, which would let the same material appear in "
            "both a train and a test fold. Use a coarser --group_col or "
            "gather more diverse rows before rerunning."
        )
    effective_splits = min(n_splits, n_groups)
    if effective_splits < n_splits:
        print(f"  Only {n_groups} distinct groups available -- reducing "
              f"GroupKFold splits from {n_splits} to {effective_splits} "
              "(still strictly grouped, never falling back to plain KFold).")

    splitter = GroupKFold(n_splits=effective_splits)
    splits = splitter.split(X, y, groups=groups)

    scores = []
    for fold, (tr_idx, te_idx) in enumerate(splits):
        model = xgb.XGBRegressor(
            max_depth=3, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, enable_categorical=True, tree_method="hist",
            random_state=seed,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[te_idx])
        r2 = r2_score(y.iloc[te_idx], pred)
        scores.append(r2)
        print(f"  fold {fold}: R2 = {r2:.3f}  (n_train={len(tr_idx)} n_test={len(te_idx)})")
    return np.array(scores)


# --------------------------------------------------------------------------
# Core: group-level OOB bootstrap with Boruta-style shadow-feature testing
# --------------------------------------------------------------------------

def _make_shadow_frame(X: pd.DataFrame, rng: np.random.RandomState, n_copies: int = 3) -> pd.DataFrame:
    """
    n_copies independently-permuted twins of every column, carrying each
    column's marginal distribution but no relationship to y or to the other
    columns.

    Using multiple shadow copies (not just one) matters when there are few
    real candidate columns: gradient-boosted trees greedily reuse whichever
    features already explain the residual, so with only 1 shadow per real
    column a lone shadow can go entirely unused across every tree in the
    ensemble -- its SHAP sits near zero not because it's noisier than the
    real columns, but because it was never given a chance to split. Multiple
    shadow copies plus feature subsampling below (colsample_bytree/bynode)
    make the shadow_max a fair ceiling on what pure noise can achieve.
    """
    shadow = pd.DataFrame(index=X.index)
    for c in X.columns:
        for k in range(n_copies):
            perm = rng.permutation(X[c].values)
            shadow[f"shadow{k}__{c}"] = perm
            if str(X[c].dtype) == "category":
                shadow[f"shadow{k}__{c}"] = shadow[f"shadow{k}__{c}"].astype("category")
    return shadow


def group_oob_boruta_bootstrap(X, y, groups, n_boot=200, seed=42, min_oob_rows=8, top_k=5,
                                n_shadow_copies=3, colsample=0.6):
    """
    For n_boot iterations:
      - sample groups WITH replacement (bootstrap at the group level)
      - in-bag rows = all rows whose group was sampled (repeated per draw count)
      - OOB rows = all rows whose group was NEVER sampled this iteration
      - fit XGBoost on in-bag [real | shadow] columns
      - compute SHAP on OOB rows only
      - a real column "wins" the iteration if its OOB |SHAP| beats the max
        OOB |SHAP| across ALL shadow columns this iteration (Boruta hit)
      - record each real column's rank among real columns this iteration

    Returns:
      imp_df        -- one row per iteration, mean |SHAP| per real column (OOB)
      rank_df       -- one row per iteration, rank per real column (1=best, OOB)
      hits_df       -- one row per iteration, 1/0 hit-vs-shadow-max per real column
      n_used, n_boot
    """
    import xgboost as xgb
    import shap

    rng = np.random.RandomState(seed)
    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    imp_records, rank_records, hit_records = [], [], []
    n_used = 0

    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        in_bag_group_set = set(sampled_groups.tolist())
        oob_mask = ~groups.isin(in_bag_group_set)
        if oob_mask.sum() < min_oob_rows:
            continue

        counts = pd.Series(sampled_groups).value_counts()
        idx_list = []
        for g, c in counts.items():
            g_idx = X.index[groups == g].tolist()
            idx_list.extend(g_idx * int(c))

        Xb_real = X.loc[idx_list].reset_index(drop=True)
        yb      = y.loc[idx_list].reset_index(drop=True)
        shadow_rng = np.random.RandomState(seed * 100003 + b)
        Xb_shadow = _make_shadow_frame(Xb_real, shadow_rng, n_copies=n_shadow_copies)
        Xb = pd.concat([Xb_real, Xb_shadow], axis=1)

        Xoob_real = X[oob_mask].reset_index(drop=True)
        # OOB shadow columns are only used to keep the SAME feature schema for
        # prediction/SHAP; their values don't matter for real-column ranking.
        Xoob_shadow = _make_shadow_frame(Xoob_real, shadow_rng, n_copies=n_shadow_copies)
        Xoob = pd.concat([Xoob_real, Xoob_shadow], axis=1)

        # Lower colsample forces trees to rotate through more of the (now much
        # larger, real+shadow) feature set instead of greedily reusing the same
        # few real columns every round -- otherwise shadows can go unused for
        # the entire ensemble and shadow_max is an artificially weak floor.
        model = xgb.XGBRegressor(
            max_depth=3, n_estimators=200, subsample=0.8,
            colsample_bytree=colsample, colsample_bynode=colsample,
            learning_rate=0.05, enable_categorical=True, tree_method="hist",
            random_state=seed + b,
        )
        model.fit(Xb, yb)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xoob)
        mean_abs = np.abs(shap_values).mean(axis=0)
        col_importance = dict(zip(Xb.columns, mean_abs))

        real_cols = list(X.columns)
        shadow_cols = [f"shadow{k}__{c}" for c in real_cols for k in range(n_shadow_copies)]
        shadow_max = max(col_importance[c] for c in shadow_cols)

        real_importance = {c: col_importance[c] for c in real_cols}
        imp_records.append(real_importance)

        order = sorted(real_cols, key=lambda c: -real_importance[c])
        rank_records.append({c: order.index(c) + 1 for c in real_cols})

        hit_records.append({c: int(real_importance[c] > shadow_max) for c in real_cols})

        n_used += 1
        if n_used % 25 == 0:
            print(f"  OOB-Boruta iteration {n_used} (of {n_boot} attempted, "
                  f"{b + 1} drawn so far)")

    if n_used == 0:
        raise RuntimeError(
            f"No bootstrap iteration produced >= {min_oob_rows} OOB rows out of "
            f"{n_groups} groups. Too few groups for group-level OOB bootstrap -- "
            "lower --min_oob_rows, use a coarser --group_col, or gather more rows."
        )
    print(f"OOB-Boruta bootstrap: {n_used}/{n_boot} iterations usable "
          f"(rest skipped: OOB group set too small)")

    return (pd.DataFrame(imp_records), pd.DataFrame(rank_records),
            pd.DataFrame(hit_records), n_used, n_boot)


def summarize_boruta(imp_df, rank_df, hits_df, top_k=5, alpha=0.05, min_hit_rate=0.6):
    """
    Confirmation requires BOTH statistical and practical significance:
      - p_confirm < alpha (binomial test vs 50/50 -- statistical)
      - hit_rate >= min_hit_rate (practical margin over chance)
    With n_boot in the hundreds, a binomial test alone flags hit_rate=0.55 as
    "significant" even though that's barely better than a coin flip -- the
    hit_rate floor keeps "confirmed" meaning "clearly and consistently beats
    noise," not just "beats noise more often than not, with enough samples
    to prove it statistically."
    """
    from scipy.stats import binomtest

    n_iter = len(hits_df)
    rows = []
    for col in imp_df.columns:
        hits = int(hits_df[col].sum())
        hit_rate = hits / n_iter
        p_confirm = binomtest(hits, n_iter, p=0.5, alternative="greater").pvalue
        p_reject  = binomtest(hits, n_iter, p=0.5, alternative="less").pvalue
        if p_confirm < alpha and hit_rate >= min_hit_rate:
            verdict = "confirmed"
        elif p_reject < alpha:
            verdict = "rejected"
        else:
            verdict = "tentative"

        ranks = rank_df[col]
        topk_freq = float((ranks <= top_k).mean())

        rows.append({
            "column":            col,
            "mean_oob_abs_shap": imp_df[col].mean(),
            "std_oob_abs_shap":  imp_df[col].std(),
            "median_rank":       ranks.median(),
            "rank_iqr":          ranks.quantile(0.75) - ranks.quantile(0.25),
            f"top{top_k}_freq":  topk_freq,
            "boruta_hit_rate":   hit_rate,
            "boruta_p_confirm":  p_confirm,
            "boruta_verdict":    verdict,
        })
    out = pd.DataFrame(rows).sort_values("mean_oob_abs_shap", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Ablation stability check (corroborates Boruta-confirmed columns only --
# expensive to run on every candidate, cheap enough to run on the shortlist)
# --------------------------------------------------------------------------

def ablation_stability(X, y, groups, candidate_cols, n_splits=5, n_repeats=5, seed=42):
    """Leave-one-column-out grouped-CV. GroupKFold only, same no-fallback
    rule as cross_validated_r2 -- see that function's docstring."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score

    n_groups = groups.nunique()
    if n_groups < 2:
        raise RuntimeError(
            f"Only {n_groups} distinct group(s) -- ablation stability requires "
            "grouped CV with at least 2 groups. Refusing to fall back to "
            "ordinary KFold."
        )
    effective_splits = min(n_splits, n_groups)

    def _cv_r2(X_sub, seed_):
        splitter = GroupKFold(n_splits=effective_splits)
        splits = list(splitter.split(X_sub, y, groups=groups))
        scores = []
        for tr_idx, te_idx in splits:
            model = xgb.XGBRegressor(
                max_depth=3, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, enable_categorical=True, tree_method="hist",
                random_state=seed_,
            )
            model.fit(X_sub.iloc[tr_idx], y.iloc[tr_idx])
            scores.append(r2_score(y.iloc[te_idx], model.predict(X_sub.iloc[te_idx])))
        return np.mean(scores)

    rows = []
    for col in candidate_cols:
        full_scores, dropped_scores = [], []
        for r in range(n_repeats):
            s = seed + r
            full_scores.append(_cv_r2(X, s))
            dropped_scores.append(_cv_r2(X.drop(columns=[col]), s))
        full_scores, dropped_scores = np.array(full_scores), np.array(dropped_scores)
        delta = full_scores - dropped_scores
        rows.append({
            "column": col,
            "mean_delta_r2_with_minus_without": delta.mean(),
            "std_delta_r2": delta.std(),
            "ablation_supports_inclusion": bool(delta.mean() > 0 and (delta.mean() - delta.std()) > -1e-3),
        })
    return pd.DataFrame(rows).sort_values("mean_delta_r2_with_minus_without", ascending=False)


def flag_multicollinearity(X, numeric_cols, threshold=0.8):
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = X[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr.loc[c1, c2]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"col_1": c1, "col_2": c2, "correlation": r})
    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)


def compare_to_current_compact_schema(confirmed_cols, training_script_dir):
    if training_script_dir:
        sys.path.insert(0, str(training_script_dir))
    try:
        import highk_alignn_train_v4_60_3_locked_disjoint_baseline as T
        current = set(T.PROCESS_PARAMS_FEATURES["numerical"]) | set(T.PROCESS_PARAMS_FEATURES["categorical"].keys())
    except Exception as e:
        print(f"  (could not import training script to compare against current compact schema: {e})")
        return

    rec = set(confirmed_cols)
    print("\nCurrent compact PROCESS_PARAMS_FEATURES columns:")
    print(" ", sorted(current))
    print("\nBoruta-CONFIRMED columns not currently in PROCESS_PARAMS_FEATURES (candidates to ADD):")
    print(" ", sorted(rec - current) or "(none)")
    print("\nCurrent compact columns Boruta did NOT confirm (candidates to review for removal):")
    print(" ", sorted(current - rec) or "(none)")
    print("\nReminder: this is a schema recommendation only. Embedding dim, hidden "
          "width, learning rate, and process_delta_bound remain validation-stage "
          "hyperparameters, tuned separately after the column set is fixed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_db", required=True)
    ap.add_argument("--backbone_preds", required=True)
    ap.add_argument("--allowlist", default=None,
                     help="REQUIRED for a real run: reviewed CSV with columns "
                          "'column,type' (type in {numeric,categorical}).")
    ap.add_argument("--id_column", default=None)
    ap.add_argument("--tier3_split", default="train", choices=["train", "val", "test"],
                     help="Which locked split to use for feature discovery. "
                          "Leave at 'train' -- val/test exist only for later "
                          "checkpoint selection and reporting.")
    ap.add_argument("--group_col", default="material_family",
                     help="Column to group-bootstrap on for OOB SHAP and CV, "
                          "to keep the same material off both sides of a fold.")
    ap.add_argument("--training_script_dir", default=None)
    ap.add_argument("--out_dir", default="shap_results")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--min_oob_rows", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05, help="Boruta binomial-test significance level")
    ap.add_argument("--min_hit_rate", type=float, default=0.6,
                     help="Practical-significance floor: fraction of iterations a real "
                          "column must beat the max shadow to count as 'confirmed', "
                          "on top of the statistical p<alpha test.")
    ap.add_argument("--n_shadow_copies", type=int, default=3,
                     help="Independent shuffled shadow copies per real column per "
                          "iteration. Raise this if you have very few candidate "
                          "columns (<6) -- see comments in _make_shadow_frame.")
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--dump_columns", action="store_true")
    ap.add_argument("--suggest_allowlist", action="store_true",
                     help="Write a candidate allowlist template for manual review, then exit.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_full = load_full_db(Path(args.full_db))
    if args.dump_columns:
        print("\nAll columns in full DB:")
        for c in df_full.columns:
            print(f"  {c}  (dtype={df_full[c].dtype}, non-null={df_full[c].notna().sum()}/{len(df_full)})")
        return

    df_pred = load_backbone_preds(Path(args.backbone_preds))
    merged = merge_on_row_id(df_full, df_pred, id_column=args.id_column)

    if args.suggest_allowlist:
        suggest_allowlist(merged, out_dir)
        return

    if not args.allowlist:
        raise SystemExit(
            "--allowlist is required. Run once with --suggest_allowlist to get "
            "a starting template, review/edit it, then pass it back in."
        )

    numeric_cols, categorical_cols = load_allowlist(Path(args.allowlist))

    train_df = filter_to_train_split(merged, split=args.tier3_split)

    X, y, groups = build_feature_matrix(train_df, numeric_cols, categorical_cols, args.group_col)

    print("\n--- Surrogate sanity check (grouped CV R^2 on TRAIN rows) ---")
    scores = cross_validated_r2(X, y, groups)
    print(f"Mean R2 = {scores.mean():.3f}  (std={scores.std():.3f})")
    if scores.mean() < 0.05:
        print("WARNING: surrogate R^2 is near zero on TRAIN rows alone. The stability "
              "tests below are still valid (they compare real vs shadow within the "
              "same weak fits), but treat any 'confirmed' columns as a weak signal, "
              "not strong evidence, until N grows.")

    print(f"\n--- Group-level OOB bootstrap with Boruta shadow-feature test ({args.n_boot} iterations) ---")
    imp_df, rank_df, hits_df, n_used, n_boot = group_oob_boruta_bootstrap(
        X, y, groups, n_boot=args.n_boot, min_oob_rows=args.min_oob_rows, top_k=args.top_k,
        n_shadow_copies=args.n_shadow_copies,
    )
    summary = summarize_boruta(imp_df, rank_df, hits_df, top_k=args.top_k,
                                alpha=args.alpha, min_hit_rate=args.min_hit_rate)
    summary.to_csv(out_dir / "shap_boruta_summary.csv", index=False)
    print("\nColumn stability summary:")
    print(summary.to_string(index=False))

    confirmed = summary.loc[summary["boruta_verdict"] == "confirmed", "column"].tolist()
    tentative = summary.loc[summary["boruta_verdict"] == "tentative", "column"].tolist()
    rejected  = summary.loc[summary["boruta_verdict"] == "rejected", "column"].tolist()
    print(f"\nConfirmed ({len(confirmed)}): {confirmed}")
    print(f"Tentative ({len(tentative)}): {tentative}")
    print(f"Rejected  ({len(rejected)}): {rejected}")

    print("\n--- Multicollinearity check (numeric columns, |r| >= 0.8) ---")
    collinear = flag_multicollinearity(X, numeric_cols)
    if len(collinear):
        print(collinear.to_string(index=False))
        collinear.to_csv(out_dir / "multicollinear_pairs.csv", index=False)
        print("NOTE: SHAP splits credit between correlated columns above -- judge "
              "their COMBINED confirmation, not each one's rank alone.")
    else:
        print("(none found)")

    if not args.skip_ablation and confirmed:
        print(f"\n--- Ablation stability check on {len(confirmed)} Boruta-confirmed column(s) ---")
        ablation_df = ablation_stability(X, y, groups, confirmed)
        ablation_df.to_csv(out_dir / "ablation_stability.csv", index=False)
        print(ablation_df.to_string(index=False))
        disagreeing = ablation_df.loc[~ablation_df["ablation_supports_inclusion"], "column"].tolist()
        if disagreeing:
            print(f"\nNOTE: Boruta confirmed these but ablation did NOT clearly support "
                  f"them ({disagreeing}) -- review manually before adding to "
                  "PROCESS_PARAMS_FEATURES; small-N ablation is itself noisy.")

    compare_to_current_compact_schema(confirmed, args.training_script_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(summary))))
        plot_df = summary.sort_values("mean_oob_abs_shap")
        colors = plot_df["boruta_verdict"].map(
            {"confirmed": "#4C9A2A", "tentative": "#CCA300", "rejected": "#B3413E"}
        )
        ax.barh(plot_df["column"], plot_df["mean_oob_abs_shap"],
                xerr=plot_df["std_oob_abs_shap"], color=colors)
        ax.set_xlabel("mean OOB |SHAP value| (bootstrap mean +/- std)")
        ax.set_title("Process column stability for residual_log (TRAIN rows only)\n"
                      "green=confirmed  yellow=tentative  red=rejected (Boruta shadow test)")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_boruta_bar.png", dpi=150)
        print(f"\nSaved bar chart: {out_dir / 'shap_boruta_bar.png'}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
