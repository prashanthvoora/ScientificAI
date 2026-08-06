#!/usr/bin/env python3
"""
generate_backbone_predictions.py
=================================
Step 1 of the SHAP process-column analysis.

Produces one row per experimental sample in the FULL process DB
(data/processed/process_db_clean.csv) with:
    - k_total_log_true      (measured, log space)
    - k_dft_log_backbone    (frozen Tier-2 DFT backbone prediction, log space,
                              computed BEFORE any process_delta_head correction
                              -- i.e. what the crystal-graph-only model would
                              say with zero knowledge of ALD/PDA conditions)
    - residual_log           = k_total_log_true - k_dft_log_backbone
    - tier3_split             = "train" / "val" / "test" /
                                 "train_excluded_leak_adjacent", read from the
                                 LOCKED disjoint split files your training
                                 script already persists (TIER3_TEST_SPLIT_PATH,
                                 TIER3_VAL_SPLIT_PATH), with STRICT-100 recovery
                                 (not the training script's 95% floor) and a
                                 paper/material/donor leakage-adjacency guard
                                 on top. SHAP feature discovery in step 2 must
                                 only use tier3_split=="train" rows -- see
                                 tag_tier3_splits() / exclude_leak_adjacent_train_rows().

`residual_log` is the correct SHAP target: it isolates exactly what the
process encoder / process_delta_head is being asked to explain, rather than
mixing in whatever the DFT backbone already gets right from crystal structure
alone.

This script is a thin wrapper around functions ALREADY PROVEN in
highk_alignn_train_v4_60_3_locked_disjoint_baseline.py:
    - DatasetExtractor.load_experimental_process_db()
    - TierDatasetBuilder.build_tier3()
    - _impute_structures()          (same donor-pool imputation used in
                                      tier3_finetune / tier3_evaluate --rebuild_tier3)
    - HighKALIGNN / ALIGNNTrainer / _load_state_dict_shape_compatible

It does NOT retrain anything and does NOT touch the locked val/test split --
it just runs inference over every experimental row that has (or can be
donor-imputed with) a valid crystal structure, using the Tier-2 checkpoint
as the frozen DFT-only backbone.

USAGE (run in the SAME directory as the training script, or put it on
PYTHONPATH -- see the run instructions in chat):

    python generate_backbone_predictions.py \\
        --tier2_checkpoint checkpoints/tier2_best.pt \\
        --out backbone_predictions_full_db.csv

Requires: Tier 1 and Tier 2 HDF5 caches to already exist (i.e. you have run
tier1_pretrain and tier2_finetune at least once). It does NOT require the
Tier 3 HDF5 cache -- it rebuilds Tier 3 in-memory with force_rebuild=True so
it always reflects the CURRENT process_db_clean.csv on disk.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# IMPORTANT: this must be importable -- either run this script from the same
# folder as the training file, or add that folder to PYTHONPATH first, e.g.:
#   export PYTHONPATH=/path/to/training/script/folder:$PYTHONPATH
import highk_alignn_train_v4_60_3_locked_disjoint_baseline as T


def build_full_experimental_dataset(builder, extractor, process_db_path: Path) -> pd.DataFrame:
    """
    Mirrors the --rebuild_tier3 block inside main()'s tier3_evaluate path,
    but keeps EVERY experimental row (not just the locked val/test split).
    """
    t1_hdf5 = builder.TIER_PATHS[1]
    t2_hdf5 = builder.TIER_PATHS[2]
    if not t1_hdf5.exists() or not t2_hdf5.exists():
        raise FileNotFoundError(
            f"Tier1/Tier2 HDF5 caches not found at {t1_hdf5} / {t2_hdf5}. "
            "Run --mode tier1_pretrain and --mode tier2_finetune at least "
            "once before this script."
        )

    df_tier2 = pd.read_hdf(t2_hdf5, key="data")

    df_exp = extractor.load_experimental_process_db(path=process_db_path)
    if len(df_exp) == 0:
        raise RuntimeError(
            f"process_db_clean.csv not found or empty at {process_db_path}"
        )
    resolved = process_db_path.resolve()
    mtime = pd.Timestamp(resolved.stat().st_mtime, unit="s") if resolved.exists() else "N/A"
    print(f"Loaded full experimental process DB: {len(df_exp)} rows")
    print(f"  path (resolved): {resolved}")
    print(f"  last modified:   {mtime}")
    print("  ^ compare this path/mtime/row-count against whatever you pass as "
          "--full_db to shap_process_column_analysis.py -- they must be the same file.")

    df_tier3 = builder.build_tier3(df_tier2, df_exp, force_rebuild=True)

    # -- donor-pool structure imputation (identical to tier3_evaluate --rebuild_tier3) --
    df_structural    = df_tier3[df_tier3["atoms_dict"].notna()].copy()
    df_process_only  = df_tier3[df_tier3["atoms_dict"].isna()].copy()
    if len(df_process_only) > 0 and len(df_structural) > 0:
        df_donor_pool_wide = df_tier2[
            df_tier2["formula"].apply(
                lambda f: isinstance(f, str) and "O" in f
                and any(el in f for el in T.HIGH_K_DONOR_ELEMENTS)
            )
        ].copy()
        print(f"Donor pool (Tier2 high-k oxides): {len(df_donor_pool_wide)} rows")
        df_imputed, df_unmatched = T._impute_structures(df_process_only, df_donor_pool_wide)
        print(f"Structure-imputed: {len(df_imputed)}  unmatched: {len(df_unmatched)}")
        if len(df_imputed) > 0:
            df_tier3 = pd.concat(
                [df_structural, df_imputed, df_unmatched], ignore_index=True, sort=False
            )

    cfg = T.TIER3_TRAIN_CONFIG
    if cfg.get("log_transform", False):
        src_col, log_col = cfg["log_original_col"], cfg["target"]
        k_num    = pd.to_numeric(df_tier3[src_col], errors="coerce")
        valid_k  = k_num.notna()
        df_tier3[log_col] = np.nan
        df_tier3.loc[valid_k, log_col] = np.log(k_num[valid_k].clip(lower=0.1))

    df_kvalid = df_tier3[df_tier3[cfg["target"]].notna()].copy()
    df_final  = df_kvalid[df_kvalid["atoms_dict"].notna()].copy()

    # Keep only actual experimental rows (excludes any Tier2 DFT donor rows
    # that build_tier3 may also fold in for the HfO2-family DFT subset).
    df_final = df_final[df_final["source"] == "Experimental"].reset_index(drop=True)
    return df_final


def tag_tier3_splits(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every row with its locked Tier-3 split membership (train/val/test),
    using the SAME persisted split files and key-matching logic that
    tier3_finetune / tier3_evaluate already trust -- never re-derive a split.

    SHAP feature discovery must only see 'train' rows. VAL is used for
    checkpoint selection and TEST is the locked holdout; letting either
    influence which columns get wired into ProcessParamsEncoder is the same
    class of leakage as the VAL==TEST bug already fixed for MAD:MAE
    reporting -- just one step earlier in the pipeline.

    STRICT-100 (tightened beyond the training script's own tolerance):
    load_tier3_test_positions_from_split() internally accepts >=95% key
    recovery -- a reasonable tolerance for keeping the main MAD:MAE pipeline
    running through minor drift. This script requires 100% recovery of both
    the locked TEST and VAL split files. A feature-discovery run deciding the
    process-encoder SCHEMA itself should never operate on an ambiguous or
    partially-reconstructed notion of "which rows are held out."

    Fails hard (no silent fallback) if either split file is missing, if
    STRICT-SPLIT-LOAD can't match >=95% (its own floor), or if it matches
    less than 100%.
    """
    if not T.TIER3_TEST_SPLIT_PATH.exists():
        raise RuntimeError(
            f"No locked TEST split found at {T.TIER3_TEST_SPLIT_PATH}. "
            "SHAP feature discovery must run on Tier-3 TRAIN rows only, which "
            "requires the locked disjoint split to already exist. Run "
            "tier3_finetune (v4.59.1+ disjoint protocol) at least once first."
        )
    if not T.TIER3_VAL_SPLIT_PATH.exists():
        raise RuntimeError(
            f"No locked VAL split found at {T.TIER3_VAL_SPLIT_PATH}. "
            "TEST alone is not enough to isolate TRAIN rows safely under the "
            "DISJOINT-VAL protocol -- both split files are required."
        )

    n_saved_test = len(pd.read_csv(T.TIER3_TEST_SPLIT_PATH))
    n_saved_val  = len(pd.read_csv(T.TIER3_VAL_SPLIT_PATH))

    test_pos = T.load_tier3_test_positions_from_split(df_final, T.TIER3_TEST_SPLIT_PATH)
    val_pos  = T.load_tier3_test_positions_from_split(df_final, T.TIER3_VAL_SPLIT_PATH)

    if test_pos is None:
        raise RuntimeError(
            "Locked TEST split file exists but STRICT-SPLIT-LOAD could not match "
            ">=95% of its rows against the freshly rebuilt Tier-3 dataset (see "
            "warnings above). Refusing to guess which rows are train-only -- "
            "regenerate the split via tier3_finetune before rerunning this script."
        )
    if val_pos is None:
        raise RuntimeError(
            "Locked VAL split file exists but STRICT-SPLIT-LOAD could not match "
            ">=95% of its rows -- same guard as TEST, applied to VAL."
        )

    if len(test_pos) != n_saved_test:
        raise RuntimeError(
            f"STRICT-100 recovery failed for TEST: matched {len(test_pos)}/{n_saved_test} "
            f"({100*len(test_pos)/n_saved_test:.1f}%) saved rows. The training script's "
            "own 95% floor would accept this, but SHAP feature discovery requires exact "
            "(100%) recovery. Investigate why rows are missing (donor-imputation drift, "
            "process_db_clean.csv edits since the split was saved, hashing changes) "
            f"before rerunning -- see {T.TIER3_TEST_SPLIT_PATH}."
        )
    if len(val_pos) != n_saved_val:
        raise RuntimeError(
            f"STRICT-100 recovery failed for VAL: matched {len(val_pos)}/{n_saved_val} "
            f"({100*len(val_pos)/n_saved_val:.1f}%) saved rows. Same STRICT-100 "
            f"requirement as TEST -- see {T.TIER3_VAL_SPLIT_PATH}."
        )

    tags = np.full(len(df_final), "train", dtype=object)
    tags[test_pos] = "test"
    tags[val_pos]  = "val"
    overlap = set(test_pos.tolist()) & set(val_pos.tolist())
    if overlap:
        raise RuntimeError(
            f"DISJOINT-VAL leakage detected while tagging splits for SHAP: "
            f"{len(overlap)} row(s) matched BOTH the TEST and VAL split files. "
            "This should be impossible under the locked protocol -- stop and "
            "investigate before trusting any downstream feature selection."
        )

    df_final = df_final.copy()
    df_final["tier3_split"] = tags
    n_train = int((tags == "train").sum())
    print(f"Tier-3 split tagging (STRICT-100 recovery): train={n_train}  "
          f"val={len(val_pos)}/{n_saved_val}  test={len(test_pos)}/{n_saved_test}  "
          f"(total experimental rows={len(df_final)})")
    return df_final


def exclude_leak_adjacent_train_rows(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    tier3_split=="train" only guarantees a row wasn't ITSELF in the locked
    VAL/TEST split. It says nothing about a sibling row that:
      - comes from the same paper/DOI as a VAL/TEST row (shared authors,
        equipment, calibration -- systematic offsets that correlate across
        rows from one publication),
      - comes from the same FIRST AUTHOR as a VAL/TEST row, even under a
        DIFFERENT DOI (same lab, same equipment, same calibration and
        reporting conventions frequently carry across a group's papers --
        this was added after first_author/journal showed up as
        confirmed/tentative in a real SHAP run despite not being process
        parameters, indicating doi-level exclusion alone wasn't catching
        cross-paper, same-lab correlation),
      - reports the same material/formula as a VAL/TEST row (structural
        embedding similarity independent of process signal), or
      - was donor-imputed from the SAME DFT structure as a VAL/TEST row
        (identical crystal embedding from the frozen backbone -- residuals
        would correlate through donor-sharing, not through real process
        signal). This is the exact 'donor_material disjointness' concern
        already flagged as an open item for the main training protocol.

    Any of these would let a nominally-train row leak VAL/TEST-adjacent
    information into SHAP feature discovery. This relabels such rows
    'train_excluded_leak_adjacent' so step 2's default train-only filter
    drops them automatically.
    """
    def _key_set(series: pd.Series) -> set:
        if series is None:
            return set()
        vals = series.dropna().astype(str).str.strip().str.lower()
        return set(vals) - {"", "nan", "none"}

    vt = df_final[df_final["tier3_split"].isin(["val", "test"])]
    leak_paper    = _key_set(vt.get("paper_id")) | _key_set(vt.get("doi"))
    leak_author   = _key_set(vt.get("first_author"))
    leak_material = _key_set(vt.get("material")) | _key_set(vt.get("formula"))
    leak_donor    = _key_set(vt.get("imputed_from"))

    def _reason(row) -> str:
        if row["tier3_split"] != "train":
            return ""
        if leak_paper and (
            str(row.get("paper_id", "")).strip().lower() in leak_paper
            or str(row.get("doi", "")).strip().lower() in leak_paper
        ):
            return "paper"
        if leak_author and str(row.get("first_author", "")).strip().lower() in leak_author:
            return "author"
        if leak_material and (
            str(row.get("material", "")).strip().lower() in leak_material
            or str(row.get("formula", "")).strip().lower() in leak_material
        ):
            return "material"
        if leak_donor and str(row.get("imputed_from", "")).strip().lower() in leak_donor:
            return "donor"
        return ""

    reasons = df_final.apply(_reason, axis=1)
    df_final = df_final.copy()
    df_final["leak_adjacency_reason"] = reasons

    n_excluded = int((reasons != "").sum())
    n_train_before = int((df_final["tier3_split"] == "train").sum())
    if n_excluded:
        by_reason = reasons[reasons != ""].value_counts().to_dict()
        print(f"Leakage-adjacency guard: excluding {n_excluded}/{n_train_before} "
              f"nominally-train rows that share paper/author/material/donor identity "
              f"with a locked VAL/TEST row  (breakdown: {by_reason})")
        df_final.loc[reasons != "", "tier3_split"] = "train_excluded_leak_adjacent"
    else:
        print("Leakage-adjacency guard: no train rows share paper/author/material/donor "
              "identity with VAL/TEST -- nothing excluded.")

    n_train_after = int((df_final["tier3_split"] == "train").sum())
    print(f"tier3_split=='train' available for SHAP: {n_train_after} "
          f"(was {n_train_before} before the leakage-adjacency guard)")
    return df_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_db", default=str(T.DATA_ROOT / "processed" / "process_db_clean.csv"),
                     help="Path to process_db_clean.csv. MUST be the exact same file you "
                          "pass as --full_db to shap_process_column_analysis.py -- this "
                          "used to be silently hardcoded to the training script's default "
                          "path, which caused row-ID mismatches when a different file was "
                          "used downstream. Defaults to that same hardcoded path for "
                          "backward compatibility, but pass it explicitly to be sure.")
    ap.add_argument("--tier2_checkpoint", default=str(T.CKPT_ROOT / "tier2_best.pt"),
                     help="Frozen DFT backbone checkpoint (Tier-2, NOT Tier-3 -- "
                          "we want the pure crystal-graph prediction, before any "
                          "process_delta_head correction).")
    ap.add_argument("--out", default="backbone_predictions_full_db.csv")
    args = ap.parse_args()

    T.init_dist()   # safe no-op single-GPU/CPU init when not launched via torchrun

    extractor = T.DatasetExtractor()
    builder   = T.TierDatasetBuilder()

    df_final = build_full_experimental_dataset(builder, extractor, Path(args.full_db))
    print(f"Experimental rows with valid target + structure: {len(df_final)}")
    if len(df_final) == 0:
        raise RuntimeError("No usable rows -- check process_db_clean.csv / donor imputation logs above.")

    df_final = tag_tier3_splits(df_final)
    df_final = exclude_leak_adjacent_train_rows(df_final)

    target_col = T.TIER2_TRAIN_CONFIG["target"]                        # "k_total_log"
    task_names = [target_col] + T.TIER2_TRAIN_CONFIG["aux_targets"]     # must match tier2_best.pt heads

    model = T.HighKALIGNN(config=T.ALIGNN_BASE_CONFIG, task_names=task_names)
    model.fit_encoder_stats(df_final)

    ckpt = T.safe_load_checkpoint(Path(args.tier2_checkpoint))
    if ckpt is None:
        raise RuntimeError(f"Could not load checkpoint: {args.tier2_checkpoint}")
    missing, unexpected = T._load_state_dict_shape_compatible(
        model, ckpt.get("model_state_dict", ckpt),
        checkpoint_label=args.tier2_checkpoint,
    )
    print(f"Checkpoint loaded (epoch={ckpt.get('epoch','?')}, "
          f"val_mae={ckpt.get('val_mae', float('nan')):.4f}). "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    trainer = T.ALIGNNTrainer(model, T.TIER2_TRAIN_CONFIG,
                               ckpt_prefix="tier2_backbone_dump", ablate_context=False)
    model_core = trainer.model_core
    model_core.eval()

    dataset = T.HighKGraphDataset(df_final, target_col=target_col, aux_cols=[])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=T.TIER2_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=T.HighKGraphDataset.collate_fn,
    )

    # Same reset_index the Dataset does internally -- keeps row_idx alignment exact.
    df_ref = df_final.reset_index(drop=True)
    expected_row_idx = set(dataset.valid_idx)

    rows = []
    seen_row_idx = set()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            graph      = batch["graph"].to(trainer.device)
            line_graph = batch["line_graph"].to(trainer.device)
            target     = batch["target"].detach().cpu().view(-1)
            func_code  = batch.get("functional_code")
            if func_code is not None:
                func_code = func_code.to(trainer.device)

            # Backbone-only forward pass -- identical to the base_log branch
            # used in _write_tier3_prediction_audit, deliberately WITHOUT any
            # process_delta_head contribution.
            embedding = model_core.backbone((graph, line_graph, None))
            embedding = model_core._apply_func_conditioning(embedding, func_code)
            base_log  = model_core.task_heads[target_col](embedding).detach().cpu().view(-1)

            # ONE-TO-ONE ROW IDENTITY: no arange() fallback. If collate_fn
            # ever fails to return row_indices, or drops a row mid-batch for
            # a graph-schema mismatch, silently substituting positional
            # indices would mislabel every subsequent row_idx <-> prediction
            # pair in that batch. Fail loudly instead.
            row_indices = batch.get("row_indices")
            if row_indices is None:
                raise RuntimeError(
                    "collate_fn did not return 'row_indices' for this batch. "
                    "Refusing to fall back to positional arange() indices -- "
                    "that would silently mismatch row identity whenever any "
                    "row in the batch was dropped for a graph-schema mismatch. "
                    "Fix the dataloader/collate_fn before rerunning."
                )
            row_indices = row_indices.detach().cpu().view(-1).tolist()

            for j, row_idx in enumerate(row_indices):
                if row_idx in seen_row_idx:
                    raise RuntimeError(
                        f"Row identity collision: row_idx={row_idx} was returned "
                        "twice during backbone inference. Refusing to write "
                        "ambiguous predictions -- investigate the DataLoader "
                        "(shuffle/sampler settings) before rerunning."
                    )
                if row_idx not in df_ref.index:
                    raise RuntimeError(
                        f"row_idx={row_idx} returned by the dataloader is not in "
                        "df_ref.index. Row-identity mapping between the dataset "
                        "and df_final is broken -- do not trust any output from "
                        "this run."
                    )
                seen_row_idx.add(row_idx)
                meta     = df_ref.loc[row_idx]
                true_log = float(target[j].item())
                base_l   = float(base_log[j].item())
                rows.append({
                    # paper_id is what load_experimental_process_db() renamed
                    # your CSV's "row_id" column to -- use it to join back to
                    # process_db_clean.csv in the next script.
                    "paper_id":           meta.get("paper_id", ""),
                    "material":           meta.get("material", ""),
                    "material_family":    meta.get("material_family", ""),
                    "data_quality":       meta.get("data_quality", ""),
                    "tier3_split":        meta.get("tier3_split", ""),
                    "leak_adjacency_reason": meta.get("leak_adjacency_reason", ""),
                    "k_total_log_true":   true_log,
                    "k_dft_log_backbone": base_l,
                    "residual_log":       true_log - base_l,
                })

    missing = expected_row_idx - seen_row_idx
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"{len(missing)}/{len(expected_row_idx)} structurally-valid experimental "
            "rows never produced a prediction -- most likely dropped by collate_fn's "
            "DGL graph-schema-consistency filter within a batch. One-to-one row "
            "identity is required for SHAP feature discovery; partial coverage is "
            f"not acceptable. Missing row_idx sample: {sample}. Try rerunning with "
            "a smaller batch_size (e.g. override T.TIER2_TRAIN_CONFIG['batch_size']=1) "
            "to isolate which structures produce a non-standard graph schema."
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}  "
          f"(one-to-one verified: {len(seen_row_idx)}/{len(expected_row_idx)} rows covered)")
    print("\nresidual_log summary:")
    print(out_df["residual_log"].describe())


if __name__ == "__main__":
    main()
