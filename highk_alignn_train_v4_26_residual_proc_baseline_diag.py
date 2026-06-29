# -*- coding: utf-8 -*-
"""
==============================================================================
 High-k Dielectric Discovery -- Three-Tier Scalable ALIGNN Training Pipeline
 Version 4.5.22 (v4.25: fix DGLError schema mismatch in collate_fn for imputed rows)
==============================================================================
 v4.5.22 fixes  (v4.25: DGLError schema mismatch in collate_fn)
 ---------------------------------------------------------------------------
 FIX-DGL-SCHEMA  DGLError in DataLoader worker: "Expect all graphs to have
           the same schema on nodes['_N'].data but graph 5 got ..."
           Three causes all fixed:
           1. DGLError inherits from BaseException not Exception in some
              DGL versions — the existing try/except Exception in __getitem__
              didn't catch it. Graph with bad schema reached collate_fn.
              Fix: added explicit BaseException catch in graph build block.
           2. Imputed experimental rows may inherit is_molecule=True from a
              QM9 donor, causing use_lattice_prop=False → different node
              schema than crystal graphs (use_lattice_prop=True).
              Fix: is_mol forced to False for all experimental rows
              (ALD films are always crystalline).
           3. Degenerate graphs (0 nodes, 0 edges, non-92-dim atom_features)
              not caught before collate_fn.
              Fix: explicit schema validation in __getitem__ after graph
              build — returns None for degenerate graphs.
           4. collate_fn had no protection around dgl.batch() calls.
              Fix: _graph_schema_key() pre-filters batch to schema-matching
              graphs only; BaseException catch around dgl.batch() as
              last-resort fallback to skip the entire batch safely.

 v4.5.21 fixes  (v4.24: has_structure=True for imputed rows)
 ---------------------------------------------------------------------------
 FIX-HAS-STRUCT  _impute_structures set atoms_dict but NOT has_structure=True.
           HighKGraphDataset valid filter: has_structure.fillna(False).
           Imputed rows had atoms_dict set but has_structure=NaN → excluded
           from valid_idx → never sampled → proc_avail_pct=0% always.
           Fix: df_imp.at[idx, "has_structure"] = True for matched rows.

 FIX-DONOR-SPLIT  Separate donor pool (broad) from DFT training rows (Hf/Zr):
           v4.23 expanded is_hfo2_family to all high-k elements causing
           df_structural to grow to 1128 rows (diluting proc signal).
           Fix: is_hfzr_core for df_hf training (~28 rows), broad
           df_donor_pool_wide for _impute_structures only.

 v4.5.20 fixes  (v4.23: expand donor pool + HfO2 fallback → 0 unmatched)
 ---------------------------------------------------------------------------
 FIX-DONOR-POOL  is_hfo2_family filter excluded Al2O3, ZrO2, TiO2, Ta2O5,
           SrTiO3, La2O3, Y2O3 from donor_pool (only Hf-containing formulas
           were included). _find_donor alias table mapped these CSV materials
           to HfO2 as a proxy, but the actual DFT structures for Al2O3,
           ZrO2 etc. were available in df_tier2 and should have been used.
           Fix: expanded is_hfo2_family to include all high-k oxide elements
           (Hf, Zr, Al, Ti, Ta, Sr, La, Y, Ba, Nb, Ga, In, Sc, Ce, Pr, Nd).
           Donor pool grows from ~28 to ~200+ rows covering all CSV families.

 FIX-DONOR-FALLBACK  Added HfO2/ZrO2 last-resort fallback in _find_donor.
           Any Hf-containing formula that passes all 5 match strategies
           (exact, pymatgen-reduced, alias, substring, first-token) uses
           HfO2 as structural proxy. Zr-containing → ZrO2. This guarantees
           every experimental row gets a crystal structure and can contribute
           process-param gradient. Logged at DEBUG level for auditability.
           Expected result: 0 unmatched rows, experimental k = ~111.

 v4.5.19 fixes  (v4.22: _find_donor alias table + warning log fix)
 ---------------------------------------------------------------------------
 FIX-DONOR-MATCH  69/120 experimental rows unmatched in _impute_structures:
           _find_donor had 3 match strategies (exact, pymatgen-reduced,
           substring) which only covered HfO2 and ZrO2 exact matches.
           All other formulas failed because:
           1. JARVIS stores 'Hf4O8', 'Hf2Zr2O8' etc. while CSV uses
              'HfO2', 'Hf0.5Zr0.5O2' — reduced formula only matches
              with pymatgen which may not be installed.
           2. Complex notations like 'HfAlO', 'HfLaO', 'HfO2 on MoS2',
              'HfO2/Al2O3_nanolaminate' have no JARVIS equivalent.
           Fix: added FORMULA_ALIASES dict mapping 25 CSV notation
           variants to their best JARVIS donor formulas, plus a 5th
           match strategy (first-token split on '/', '_', ' ').
           Expected unmatched after fix: ~10 rows (BaTiO3, Nb2O5,
           La2O3, LaAlO3, Y2O3 that have no Hf/Zr JARVIS donor at all).
           Expected experimental k rows: ~100 (was 46).

 FIX-DONOR-LOG   Warning showed empty [] for unmatched formulas:
           df_unmatched.get('material') returned default empty Series
           when column was absent — misleading diagnostic.
           Fix: explicit column detection with 'material'/'material_system'
           fallback so actual formula values appear in the warning.

 v4.5.18 fixes  (v4.21: TypeError in fit_stats — non-numeric strings)
 ---------------------------------------------------------------------------
 FIX-FIT-STATS  TypeError: could not convert string to float in fit_stats:
           After FIX-PROC-DROP proc param columns now correctly survive into
           df_structural. But four columns still contain non-numeric strings
           from the CSV that were never coerced to NaN:
             substrate_temp_C:    'RT' (room temp), 'photo_anneal'  (3 rows)
             n_cycles:            '90+10', 'varies'                 (22 rows)
             pressure_mTorr:      'atm', '10Pa'                     (17 rows)
             anneal_duration_min: '30s', '60s', '20 min' etc.       (68 rows)
           These pass _clean_numeric_series (runs before proc_renames) and
           _safe_val (only checks for None/float-NaN, not string-NaN) but
           crash vals.mean() in fit_stats with TypeError.
           Fix 1: pd.to_numeric(errors='coerce') in both fit_stats methods.
           Fix 2: pd.to_numeric(errors='coerce') in _safe_val return path.
           Fix 3: explicit coercion loop FIX-PROC-COERCE added at end of
           load_experimental_process_db() to clean at source with log output
           showing which columns and how many rows were coerced.

 v4.5.17 fixes  (v4.20: FIX-PROC-DROP — root cause of proc_avail=0%)
 ---------------------------------------------------------------------------
 FIX-PROC-DROP  PROCESS PARAM COLUMNS SILENTLY DROPPED IN build_tier3():
           load_experimental_process_db() correctly populates substrate_temp_C,
           anneal_temp_C, growth_rate_A_per_cycle, n_cycles, film_thickness_A,
           pressure_mTorr, anneal_duration_min, anneal_ambient, precursor_type,
           oxidant_type in df_exp (confirmed by FIX-T3-1 log coverage lines).
           But immediately after, build_tier3() does:
               df_tier3 = pd.concat([df_hf, df_exp_aligned[df_hf.columns]])
           df_hf is a DFT-only dataframe with NO process parameter columns.
           df_exp_aligned[df_hf.columns] selects only the columns present in
           df_hf — silently discarding every process param column from df_exp.
           df_tier3 stored in the HDF5 cache has substrate_temp_C = column
           ABSENT (not NaN — the column does not exist at all).
           HighKGraphDataset._has_context = False (none of the proc cols are
           in df.columns) → fast-path zeros for all rows → avail_flag=0 for
           every row → proc_avail_pct=0% throughout all Tier 3 training.
           This bug existed since v4.0 and was masked by all other fixes.
           Fix: concat full df_exp_aligned (not the column-subset). Pandas
           fills DFT rows with NaN for proc columns automatically — correct
           because DFT rows have no ALD process parameters.
           Requires --rebuild_tier3 to regenerate the HDF5 cache.

 v4.5.16 fixes  (v4.19: iloc→loc definitive fix + --rebuild_tier3 mandatory)
 ---------------------------------------------------------------------------
 FIX-ILOC-ROOT  iloc→loc IN __getitem__ (definitive fix):
           v4.18 reset_index on df_phase_b was a workaround, not a root fix.
           The correct fix is iloc→loc in HighKGraphDataset.__getitem__ and
           get_stratified_split so the code is correct regardless of whether
           the caller resets the dataframe index.
           __getitem__:         self.df.iloc[row_idx] → self.df.loc[row_idx]
           get_stratified_split: df.iloc[valid_idx]   → df.loc[valid_idx]

 REBUILD-MANDATORY  --rebuild_tier3 IS THE ACTUAL CAUSE of proc_avail=0:
           proc_avail=0% persisting through v4.18 despite code fixes means
           the HDF5 cache (tier3_project.h5) still has OLD column names from
           before the BUG-2 proc_renames fix. load_experimental_process_db
           only reruns when --rebuild_tier3 is passed. The cache stores
           ald_substrate_temp_C (raw CSV name) not substrate_temp_C (renamed
           schema name) → _extract_proc_feats finds no matching columns →
           avail=0 for all rows → proc_avail_pct=0%.
           THIS IS THE MANDATORY RUN COMMAND:
           torchrun ... v4_19.py --mode tier3_finetune \
               --weights checkpoints/tier2_best.pt --rebuild_tier3

 v4.5.15 fixes  (v4.18: iloc vs loc bug → proc_avail=0% + N_Valid counter)
 ---------------------------------------------------------------------------
 FIX-ILOC  proc_avail_pct=0% DESPITE BUG-2 FIX (substrate_temp_C populated):
           HighKGraphDataset.__getitem__ uses self.df.iloc[row_idx] where
           row_idx values come from self.df[valid].index.tolist() — these are
           INDEX LABELS, not positions. When df_phase_b is a filtered subset
           of df_structural (952 rows → 139 rows) the retained rows have
           non-contiguous labels (e.g. [813, 814, ..., 951]) matching their
           original positions in the 952-row df_structural. But iloc[813] on
           a 139-row df raises IndexError (silently caught in __getitem__ →
           returns None → collate_fn skips → row never enters a batch) or
           reads the wrong row entirely. Result: all 111 experimental rows
           with filled substrate_temp_C were silently skipped. The model
           trained ONLY on the 28 DFT rows (avail_flag=0 for all) → process
           encoder received zero gradient for the entire Tier 3 run despite
           the BUG-2 column-rename fix being correctly applied.
           Fix: .reset_index(drop=True) on df_phase_b makes labels ==
           positions == 0..138 so iloc and loc are equivalent.

 FIX-NVALID  N_Valid=5 ARTIFACT IN evaluate_multitask:
           evaluate_multitask was called with df_full=df_structural (952 rows,
           14.6% k density). It computed N_Valid as test_rows × 14.6% ≈ 3–5
           instead of the true 22 test rows (all k-valid in df_phase_b).
           MAE/RMSE/MAD:MAE were correct (computed from actual batch outputs)
           but the N_Valid column in the results table was misleading.
           Fix: evaluate_multitask now receives df_full=df_phase_b (139 rows,
           100% k density) → N_Valid correctly reports ~22 test rows.

 v4.5.14 fixes  (v4.17: AttributeError trainer.loss_fn in Phase A step loop)
 ---------------------------------------------------------------------------
 PHASE-A-ATTR  Phase A step-budget loop (v4.13) referenced wrong attributes:
               1. trainer.loss_fn → AttributeError (correct: trainer.criterion)
               2. criterion called with wrong signature — raw tensors + wrong
                  kwargs instead of (predictions_dict, targets_dict, functional_weights)
               3. model forward missing task="__all__" → scalar not dict → KeyError
               Fix: replaced 55-line inline loop with _StepBudgetLoader wrapper
               that caps iteration at phase_a_steps batches, then delegates to
               trainer.train_epoch() — reuses all correct logic, removes
               duplicated code, eliminates all three errors.

 v4.5.13 fixes  (v4.16: RuntimeError at load_pretrained_weights — strip embedding size mismatches)
 ---------------------------------------------------------------------------
 CKPT-RT4  SIZE-MISMATCH IN proc_encoder EMBEDDING TABLES (the real cause):
           strict=False handles MISSING keys and UNEXPECTED keys cleanly.
           It does NOT handle TENSOR SIZE MISMATCHES — those raise RuntimeError
           regardless of strict mode. The 3 proc_encoder categorical embedding
           tables exist in BOTH the Tier 2 checkpoint AND the v4.15 model,
           but with different row counts after BUG-5 vocab expansion:
             proc_encoder.embeddings.anneal_ambient.weight  [5,8] → [8,8]
             proc_encoder.embeddings.precursor_type.weight  [6,8] → [12,8]
             proc_encoder.embeddings.oxidant_type.weight    [5,8] → [8,8]
           This caused RuntimeError at load_pretrained_weights even with
           strict=False, which was introduced in v4.15 (CKPT-RT2).
           Fix: _remap_state_dict() now also strips these 3 keys before
           load_state_dict. Stripped keys become MISSING → random init.
           This is safe and correct because Tier 2 trains with avail_flag=0
           for all rows (no ALD process data) so these embedding tables
           received ZERO gradient during Tier 2 — their weights are random
           init values with no learned information to preserve.

 v4.5.12 fixes  (v4.15: RuntimeError loading state_dict for HighKALIGNN)
 ---------------------------------------------------------------------------
 CKPT-RT1  DEFAULT task_names IN HighKALIGNN.__init__ STILL HAD k_measured:
           task_names = ["k_measured", "band_gap", "J_g_log", "E_BD"]
           Any code path calling HighKALIGNN() without explicit task_names
           (evaluate modes, ablation runs) built task_heads.k_measured.*
           in the model, while checkpoints from v4.14 training contain
           task_heads.k_total_log.* → key-name collision at load time.
           Fix: updated default to ["k_total_log","band_gap","J_g_A_cm2","E_BD_MV_cm"]
           matching TIER3_TRAIN_CONFIG.target and aux_targets.

 CKPT-RT2  STRICT=TRUE POST-TRAINING RELOAD (line 5400):
           After Tier 3 training, best checkpoint is reloaded with:
               trainer.model_core.load_state_dict(ckpt["model_state_dict"])
           No strict=False — default strict=True raises RuntimeError when
           tier3_best.pt on disk is from a previous run (v4.12/v4.13) with
           task_heads.k_measured_log.* while current model has k_total_log.*.
           Fix: changed to strict=False and added explicit key-remap warning.

 CKPT-RT3  NO STALE CHECKPOINT GUARD — SAFE_LOAD BYPASSED:
           Line 5400 called torch.load directly, bypassing safe_load_checkpoint
           which has integrity checks and better error messages. Also: no guard
           against a stale tier3_best.pt from a previous run with incompatible
           task head names causing a silent wrong-weights evaluation.
           Fix: replaced with safe_load_checkpoint + strict=False +
           _remap_state_dict() which translates legacy key names to current
           names before loading, so old checkpoints load correctly.

 v4.5.11 fixes  (v4.14: unified k_total / k_total_log across all tiers)
 ---------------------------------------------------------------------------
 UNIFY-K  k_measured / k_measured_log RENAMED to k_total / k_total_log in
          all task-head names, config targets, loss weights, benchmarks, and
          diagnostic logs.  Removes the dual-name confusion from v4.5.3.

          STRATEGY — Option B (safe, no HDF5 invalidation):
          Internal DataFrame storage column "k_measured" is KEPT unchanged.
          Aliasing df["k_total"]=df["k_measured"] already exists in all three
          build_tier*() calls and continues to work.  Only TASK HEAD names
          and CONFIG TARGETS are renamed so all tiers use the same identifiers
          in configs, logs, JSON output, and cross-tier comparison tables.

          Changes (12 locations, no HDF5 cache rebuild required):
          1.  TIER3_TRAIN_CONFIG target:           "k_measured_log" → "k_total_log"
          2.  TIER3_TRAIN_CONFIG log_original_col: "k_measured"     → "k_total"
          3.  TASK_TO_COLUMN:      "k_measured_log" entry           → "k_total_log"
          4+5 MaskedMultiTaskLoss task_weights: "k_measured" and "k_measured_log"
              entries removed (k_total:2.0 and k_total_log:2.0 already present)
          6.  HIGH_K_THRESHOLDS: "k_measured": 35.0 removed (k_total:35.0 exists)
          7.  log_full_dataset_summary: k_col always "k_total" (was tier-conditional)
          8.  PAPER_MAD_MAE: "k_measured":1.63 removed (k_total:1.63 already exists)
          9.  _kmeas_col comment updated (logic unchanged — reads cfg["target"])
          10. tier3_test_results.json primary section key updated
          11. Tier 3 final summary log line updated
          Does NOT change: _parse_*_entry storage, CSV target_renames, HDF5 files.

 v4.5.10 fixes  (v4.13: two training-dynamics bugs)
 ---------------------------------------------------------------------------
 FIX-T3-KSUBSET  PHASE B DATALOADER BUILT ON FULL df_structural (952 rows):
        Only 139/952 rows (14.6%) have k_measured_log. In every batch of 8
        only ~1.17 rows carry k gradient; 85% of Phase B steps optimise
        band_gap/formation_energy and contribute ZERO k_measured_log signal.
        The model effectively ignores k_measured_log during Phase B.
        Fix: build Phase B dataloader on df_phase_b — df_structural filtered
        to k_measured_log non-null rows only (139 rows → 97 train rows).
        Every batch now has 100% k_measured_log signal. Aux targets
        (band_gap, formation_energy) remain available in the 139-row subset
        because DFT rows in df_structural carry all three targets.
        Diagnostic log line added: "Phase B dataloader: N rows (k-valid subset)".

 FIX-T3-PHASEACAP  PHASE A STEP COUNT INVERTED vs PHASE B (14× ratio):
        Phase A on 49K rows × 20 epochs = 116,860 steps.
        Phase B on 952 rows × 100 epochs = 8,300 steps.
        Phase A was 93% of ALL gradient steps — backbone ended up specialised
        for full-oxide band_gap, not HfO2-family k_measured_log.
        Phase B's 8,300 steps could not undo 116,860 steps of band_gap pull,
        explaining why Tier 3 performance was unchanged vs v4.11.
        Fix: replace phase_a_epochs with phase_a_steps=200 in TIER3_TRAIN_CONFIG.
        Phase A runs until 200 steps total (≈0.03 epochs on 49K rows), then exits.
        New Phase A:Phase B ratio = 200:1,200 = 0.17× (was 14.1×).
        Phase A remains a brief backbone anchor, not a dominant training phase.
        After Fix: Phase B dataloader is 139 rows → steps/epoch=12 → 100 epochs
        = 1,200 Phase B steps. Ratio is healthy and Phase B dominates correctly.

 v4.5.9 fixes  (7 bugs — 5 CSV column-mapping + Phase A + exact linear-k)
 ---------------------------------------------------------------------------
 BUG-1  J_g_A_cm2_at_1v CASE TYPO: target_renames had "J_g_A_cm2_at_1v"
        (lowercase v). CSV column is "J_g_A_cm2_at_1V" (uppercase V).
        Rename never fires → J_g_A_cm2 = 0/120 rows.
        Fix: added "J_g_A_cm2_at_1V" alias alongside lowercase fallback.

 BUG-2  ald_substrate_temp_c CASE TYPO: proc_renames had "ald_substrate_temp_c"
        (lowercase c). CSV column is "ald_substrate_temp_C" (uppercase C).
        substrate_temp_C — the primary ALD process variable — was 0/120 rows,
        making the ProcessParamsEncoder blind to deposition temperature.
        Fix: corrected to "ald_substrate_temp_C"; lowercase alias retained.

 BUG-3  chamber_base_pressure WRONG SUFFIX: proc_renames had
        "chamber_base_pressure_mTorr". CSV column is "chamber_base_pressure_Torr".
        pressure_mTorr = 0/120 rows. Also: no Torr→mTorr ×1000 unit conversion.
        Fix: added "chamber_base_pressure_Torr" key + ×1000 conversion block.

 BUG-4  anneal_duration STRING VALUES NOT PARSED: CSV stores "30s", "60 s",
        "20 min" etc. proc_renames fires correctly but pd.to_numeric() returns
        NaN on unit-suffixed strings. anneal_duration_min = NaN for all 68 rows.
        Fix: _parse_duration_to_min() unit-aware parser applied after proc_renames,
        converting s/min/h strings to float minutes.

 BUG-5  PROCESS PARAM VOCAB MISMATCH: PROCESS_PARAMS_FEATURES categorical vocab
        lists were defined for the 33-row seed CSV. The updated 120-row CSV has
        many precursor/oxidant/ambient values not in vocab → all fall to the OOV
        "other" slot → encoder treats all non-vocab precursors identically.
        Most impactful: TDMAH/TEMAH (dominant Hf precursors) and "O2 plasma"
        (space vs underscore variant) both OOV.
        Fix: expanded vocab lists to cover actual CSV values. Old entries kept at
        same index positions for checkpoint compatibility with strict=False load.

 FIX-T3-PHASE-A  CATASTROPHIC band_gap FORGETTING: Tier 3 fine-tunes the full
        backbone on ~50-80 k_measured_log rows with 2× task weight. The backbone
        forgets band_gap representations, driving MAD:MAE from 7.3 → 0.72.
        The degraded crystal embeddings also hurt k_measured prediction.
        Fix: 20-epoch Phase A on full df_tier2 (~5K oxide rows) with band_gap
        as primary target before Phase B k_measured_log fine-tuning. Mirrors
        Tier 2 proven phase_a_epochs=30 strategy. Requires df_tier2 passed to
        run_tier3_finetune(); falls back gracefully with WARNING if None.

 FIX-T3-LINMAE  APPROXIMATE LINEAR-k MAE REPORTED: Tier 3 test block used
        (exp(log_MAE)-1)*100 — correct only when all k values equal k_mean.
        MAD:MAE=0.14 came from log-space MAD≈0 (5 near-identical test samples),
        not a true model quality signal.
        Fix: return_preds=True → exact mean|exp(pred)-exp(true)| with linear-k
        MAD:MAE and pass/fail status vs paper (1.63) and project (2.5) goals.

 v4.5.8 fix  (valid k_measured_log still only 28 after --rebuild_tier3)
 ---------------------------------------------------------------------------
 FIX-T3-11  UNPARSEABLE CSV VALUES: pd.to_numeric(errors='coerce') silently
            converts all non-float strings to NaN.  Audit of the seed CSV
            (ald_process_params_seed.csv, 33 rows) found three patterns
            affecting ALL numeric target and process columns:

            Pattern A — Approximation prefix  (~20, ~22, ≈25, ca.30):
              6 of 19 filled dielectric_constant_k values had '~' prefix.
              Regex strips prefix before float parse: '~20' → 20.0.
              Also handles: ≈, <, >, ca., about, ~<, ~>

            Pattern B — Numeric range  (50-82, 115-186, 17.0-24.5):
              2 k values and 3 breakdown-field values were ranges.
              Strategy: take midpoint.  '50-82' → 66.0.
              Handles both 'lo-hi' and 'lo to hi' notation.
              Exception: single negative numbers (-5.2) are not treated
              as ranges (checked by requiring both parts to parse as
              positive floats or via explicit 'to' separator).

            Pattern C — Value + unit/field suffix  (3.3e-6 at 1MV/cm,
                         17.0-24.5 MV/cm, 1.7e-10 at 2MV/cm):
              breakdown_field_MV_cm column had leakage notation with field
              condition appended.  Strip ' at N*', ' MV/cm', ' Torr', etc.
              then re-apply A+B.  Extracts leading numeric value.

            Pattern D — Qualitative strings  (negligible Jg up to 6V):
              Truly unmappable to a float.  Returns NaN with a DEBUG log.
              Does NOT attempt to infer a value.

            Recovery from seed CSV (33 rows):
              dielectric_constant_k: 11 clean + 6 (~) + 2 (range) = 19 total
              Cleaning unlocks 8 additional rows that were silently NaN.

            Implementation: _clean_numeric_series() applied to ALL numeric
            target AND process columns immediately after target_renames in
            load_experimental_process_db().  Also applied to numeric process
            parameter columns (substrate_temp_C etc.) since those may also
            contain ~200 or 250±10 notation from literature tables.

 v4.5.7 fixes  (k_measured_log task weight, overfitting guard, rebuild reminder)
 ---------------------------------------------------------------------------
 FIX-T3-10  THREE COMPOUNDING FACTORS caused band_gap MAE 0.10→1.16 eV:

            Factor A — k_measured_log missing from task_weights (code bug):
              FIX-T3-9A switched Tier 3 primary to "k_measured_log" but
              task_weights only had "k_measured": 2.0.  "k_measured_log"
              absent → defaulted to 1.0 = same weight as band_gap (1.0).
              Gradient balance shifted from k_measured(2×) vs band_gap(1×)
              to k_measured_log(1×) vs band_gap(1×) — equal weighting with
              only 5 valid samples pushing backbone into a new log-space
              regime and damaging band_gap representations.
              Fix: "k_measured_log": 2.0 added to task_weights.

            Factor B — Stale HDF5 cache (--rebuild_tier3 not run):
              N Valid=5 for BOTH k_measured and band_gap is the proof.
              FIX-T3-8 (dielectric_constant_k → k_measured column rename)
              runs inside build_tier3() which only executes during rebuild.
              Without --rebuild_tier3, old tier3.h5 loaded with k_measured=NaN
              for 115/120 experimental rows → only 33 JARVIS DFPT entries
              had valid k_measured → train=23, val=5, test=5.
              Fix: operational — must run --rebuild_tier3.
              A WARNING is now emitted if valid k_measured_log count < 50.

            Factor C — min_epochs=60 overfitting on 23 training rows:
              23 train rows ÷ batch_size=8 = 3 batches/epoch.
              60 min_epochs × 3 batches = 180 steps, each sample seen ~62×
              total.  With 10M parameters and 23 examples, backbone overfits
              k_measured_log and catastrophically forgets band_gap.
              Fix: adaptive min_epochs guard in run_tier3_finetune.
              If valid_k_count < 100, min_epochs is capped at
              max(20, valid_k_count // 2) to prevent overfitting on tiny
              datasets while still allowing full min_epochs=60 when the
              cache is rebuilt and ~245 valid k_measured_log rows exist.

 v4.5.6 fixes  (Tier 3 training quality: log-space, upweighting, hyperparam)
 ---------------------------------------------------------------------------
 FIX-T3-9A  LOG-SPACE MISMATCH: Tier 2 backbone learned log(k) representations
            (k_total_log target, CosineAnnealingLR over log-space loss).
            Tier 3 was supervising in LINEAR k_measured space: the backbone
            outputs were calibrated for log-space gradients but received
            linear-space supervision → the head couldn't meaningfully correct
            the backbone's output.  Also: k_measured spans 3.9–386+ (same
            100× dynamic range as k_total), so high-k samples dominate the
            MSE loss in linear space, exactly the problem that required
            log-transform in Tier 2.
            Fix: target switched to "k_measured_log" = log(k_measured).
            run_tier3_finetune mirrors Tier 2 log-transform preprocessing.
            Evaluation reports both log-space MAE (training metric) and
            exp(MAE) × (linear-k MAE, publication metric).

 FIX-T3-9B  EXPERIMENTAL GRADIENT DROWNED OUT: df_structural after imputation
            has ~165 DFT k_measured rows (k_total aliased) vs ~80 experimental.
            DFT rows predict well from Tier 2 transfer (near-zero loss per
            sample).  Experimental rows have larger residuals but 2:1 minority.
            Net: gradient dominated by DFT signal; process-encoder correction
            (the entire point of Tier 3) gets weak gradient.
            Fix: per-sample loss weight of 3.0× for experimental rows.
            Implementation: is_experimental tensor added to __getitem__ /
            collate_fn; train_epoch combines it with functional_weights into
            a single sample_weights vector passed to MaskedMultiTaskLoss.
            Config: exp_sample_weight=3.0 in TIER3_TRAIN_CONFIG.

 FIX-T3-9C  HYPERPARAMETER TUNING for small Tier 3 dataset:
            batch_size   16 → 8    : doubles batches/epoch (~10 → ~20)
                                     experimental rows appear more frequently
            min_epochs   40 → 60   : backbone + process encoder need more
                                     time to jointly converge after log-space
                                     alignment and upweighting
            patience     30 → 40   : val_MAE on small dataset is noisy;
                                     more tolerance prevents false early stops

 v4.5.5 fix  (N Valid=5; k_measured / J_g / E_BD all zero from CSV)
 ---------------------------------------------------------------------------
 FIX-T3-8  TARGET COLUMN NAME MISMATCH — root cause of N Valid=5 and
           identical Tier1/Tier2 k_measured performance.

           The seed CSV schema (ald_process_params_seed.csv) uses column
           names that do NOT match the keys the target_renames dict searched
           for.  target_renames only held identity entries ("k_measured" →
           "k_measured") which never fired because the CSV never had a
           column literally named "k_measured".  All 120 experimental rows
           arrived with k_measured=NaN, J_g_A_cm2=NaN, E_BD_MV_cm=NaN.

           Observed impact:
             k_measured  N Valid = 5  (5 JARVIS DFT DFPT entries only)
             J_g_A_cm2   N Valid = 0  (blank column in results table)
             E_BD_MV_cm  N Valid = 0  (blank column in results table)
             Tier1 ≈ Tier2 performance on k_measured (same 5 data points)
             MAD:MAE = 0.71 (below 1.0 → model worse than mean predictor)

           CSV column        →  Internal schema name
           ─────────────────────────────────────────
           dielectric_constant_k → k_measured          ← was missing
           leakage_J_A_cm2_at_field → J_g_A_cm2        ← was missing
           breakdown_field_MV_cm    → E_BD_MV_cm        ← was missing
           band_gap_eV / band_gap   → band_gap          ← added aliases
           interface_state_density_eV_cm2 → (logged, not a training target)

           Fix A — Expanded target_renames in load_experimental_process_db:
             All seed CSV output column names added as explicit rename keys.
             After fix: all 120 experimental rows will contribute valid
             k_measured, J_g_A_cm2, E_BD_MV_cm targets where filled.

           Fix B — k_total → k_measured propagation in build_tier3:
             HfO2-family DFT rows in df_hf that have k_total computed (DFPT)
             but no k_measured now get k_measured = k_total explicitly.
             Adds ~125-166 more valid k_measured training rows from DFT.

           Fix C — N-valid diagnostic log before build_dataloader:
             run_tier3_finetune logs exact valid-row counts for each target
             column broken down by DFT vs experimental source before the
             dataloader is built.  Surfaces any future column mismatches
             immediately without needing to wait for the test results.

 v4.5.4 fix  (120 process-only rows excluded; process encoder never activated)
 ---------------------------------------------------------------------------
 FIX-T3-7  PROCESS ENCODER NEVER ACTIVATES: All 120 experimental rows were
           landing in df_process_only (no atoms_dict) and being discarded
           before ALIGNN training.  Consequence: avail_flag=0 for every row
           in df_structural (all DFT HfO2-family rows have no process params)
           → ProcessParamsEncoder and StackContextEncoder receive zero gradient
           throughout all 40 epochs → alpha/beta stay at 1e-3 initialisation
           → val_MAE never improves beyond Tier 2 baseline → early stopping
           fires at exactly min_epochs=40 every run.

           Fix: Option A structure imputation (TODO(v4) path A).
           _impute_structures() is called in run_tier3_finetune() before
           build_dataloader().  For each process-only row it finds the best
           JARVIS/MP donor structure in df_structural by:
             1. Exact formula match  (material col → formula col)
             2. Pymatgen reduced-formula fallback for stoichiometric variants
                e.g. "Hf0.5Zr0.5O2" → reduced → tries "HfO2" if no match
             3. Among matches: lowest |formation_energy_peratom| (most stable)
             4. Tie-break: JARVIS preferred over MP (OptB88vdW more consistent)

           After imputation:
             - Matched rows join df_structural → atoms_dict filled
             - imputed_structure=True flag set for tracking
             - imputed_from=donor_jid/mp_id for audit trail
             - functional_code set to OptB88vdW (0) for all JARVIS donors
             - process params (substrate_temp_C etc.) already filled from CSV
               → avail_flag=1 for all imputed rows
             - ProcessParamsEncoder and StackContextEncoder NOW receive real
               gradient signal → alpha/beta grow → process-aware learning

           WARNING is updated to only fire for truly unmatched formulas
           (not all 120 rows as before).  HfO2/Al2O3/ZrO2/La2O3 family
           all have JARVIS donors → expected match rate ~95%+ of rows.

 v4.5.3 fix  (Tier 3 early stopping at epoch 31 — three contributing factors)
 ---------------------------------------------------------------------------
 FIX-T3-6  PREMATURE EARLY STOPPING: Tier 3 stopped at epoch 31 of 100
           planned epochs due to three factors that compounded:

           Factor 1 — LR overshoot (root cause of best_epoch=1):
             learning_rate=5e-5 with a Tier 2 backbone already near-optimal
             for HfO2-DFT k prediction.  CosineAnnealingLR starts at the
             full LR immediately (no warmup).  First gradient step overshoots
             the Tier 2 optimum → val_MAE rises from epoch 1 onward.
             patience_ctr increments every subsequent epoch.
             Fix: 5× LR reduction to 1e-5. Gentler adaptation preserves
             the Tier 2 representation while allowing fine-tuning signal
             from the HfO2-family k_measured distribution to propagate.

           Factor 2 — min_epochs defaulted to 0:
             TIER3_TRAIN_CONFIG had no "min_epochs" key.
             ALIGNNTrainer reads min_ep = cfg.get("min_epochs", 0).
             With min_ep=0 there is no floor — early stopping can fire
             the moment patience_ctr >= patience, even at epoch 2.
             Fix: min_epochs=40. Tier 3 must complete at least 40 epochs
             regardless of patience counter.

           Factor 3 — patience defaulted to 30 (implicit):
             TIER3_TRAIN_CONFIG had no "patience" key.
             ALIGNNTrainer reads patience = cfg.get("patience", 30).
             With best_epoch=1 and patience=30: stopped at 1+30=31.
             Fix: patience=30 made explicit so the intent is documented
             and the value is not silently inherited from a default.

           Combined effect without fix:
             best_epoch=1 (LR overshoot)
             + patience=30 (implicit default)
             + min_epochs=0 (implicit default)
             = stopped at epoch 31

           Combined effect with fix:
             LR=1e-5 → less overshoot; best_epoch now likely 10-30
             + patience=30 (explicit)
             + min_epochs=40 (floor)
             → guaranteed to train through epoch 40; continues until
                val_MAE has not improved for 30 epochs after epoch 40.

 v4.5.2 fix  (silent deduplication loss of experimental rows)
 ---------------------------------------------------------------------------
 FIX-T3-5  SILENT DATA LOSS: row_hash for experimental entries used only
           3 fields (doi + material + substrate_temp_C).  When doi is absent
           (NaN → '?') and multiple rows share the same material + temperature
           (e.g. many HfO2 entries at 250°C), all those rows hash identically
           and drop_duplicates silently discards all but one.

           Observed impact: 120 rows in process_db.csv → 58 survived into
           df_process_only (62 rows lost = 52% of the experimental corpus).

           Fix A -- 7-field hash key (load_experimental_process_db):
             paper_id | material | substrate_temp_C | precursor_type
             | oxidant_type | n_cycles | row_index
             paper_id is always unique in the CSV schema (P001, P002 ...).
             row_index is included as an absolute tie-breaker, guaranteeing
             uniqueness even when all named fields happen to match.

           Fix B -- Pre/post dedup diagnostic log in build_tier3():
             Per-source row counts are now logged both before and after
             drop_duplicates so any future collision losses are immediately
             visible.  A WARNING is emitted if n_exp_lost > 0 directing the
             user to check for true duplicate rows in process_db.csv.

 v4.5.1 fix  (runtime: Tier 1 dedup unnecessarily triggered on tier3_finetune)
 ---------------------------------------------------------------------------
 FIX-T3-4  RUNTIME: --force_rebuild is a global flag that also forced
           build_tier1() to re-run including the full cross-source structural
           deduplication step (~30 min, deduplicate_cross_source MP vs JARVIS).
           When only process_db.csv has changed, Tier 1 and Tier 2 are frozen
           and their HDF5 caches are valid -- rebuilding them wastes 30+ min.

           Fix A -- New flag --rebuild_tier3:
             Dedicated flag that forces ONLY build_tier3() to rebuild.
             build_tier1() and build_tier2() always load from their existing
             HDF5 caches when this flag is used.  Use this instead of
             --force_rebuild whenever only process_db.csv has changed.

           Fix B -- Mode-aware rebuild routing in main():
             When --mode is tier3_finetune, --force_rebuild is scoped to
             Tier 3 only (build_tier1 and build_tier2 are protected).
             When --mode is tier2_finetune, --force_rebuild is scoped to
             Tier 2 and Tier 3 only (build_tier1 is protected).
             Only full_pipeline, extract_only, and tier1_pretrain propagate
             --force_rebuild to all three tiers.
             This makes --force_rebuild safe to use in any mode without
             triggering an unnecessary 30-min Tier 1 cross-dedup.

           Correct command for first process_db.csv integration:
             torchrun --nproc_per_node=2 highk_alignn_train_v4_5.py \\
                 --mode tier3_finetune \\
                 --weights checkpoints/tier2_best.pt \\
                 --rebuild_tier3

 v4.5 fixes  (pre-structure-imputation audit, applied before tier3_finetune)
 ---------------------------------------------------------------------------
 FIX-T3-1  CRITICAL: Process parameter column names in process_db.csv never
           matched PROCESS_PARAMS_FEATURES keys, so avail_flag was permanently
           0.0 for every experimental entry even after structure imputation.
           load_experimental_process_db() now applies proc_renames to align
           all ten column names, plus unit-scale conversions:
             gpc_nm_per_cycle   → growth_rate_A_per_cycle  (×10, nm→Å)
             film_thickness_nm  → film_thickness_A          (×10, nm→Å)
           Full rename map (CSV name → PROCESS_PARAMS_FEATURES name):
             deposition_temp_C      → substrate_temp_C
             post_anneal_temp_C     → anneal_temp_C
             gpc_nm_per_cycle       → growth_rate_A_per_cycle  (×10)
             num_cycles_or_thickness→ n_cycles
             film_thickness_nm      → film_thickness_A          (×10)
             chamber_pressure       → pressure_mTorr
             post_anneal_atm        → anneal_ambient
             precursor_metal        → precursor_type
             precursor_oxidant      → oxidant_type

 FIX-T3-2  FUNCTIONAL: tier3_evaluate mode was missing from the CLI and from
           run_tier_evaluate().  After Tier 3 training completed there was no
           way to re-run the test evaluation on tier3_best.pt without a full
           retrain.  Fix:
             a) "tier3_evaluate" added to argparse choices list.
             b) run_tier_evaluate() guard extended from tier in (1,2) to
                tier in (1,2,3); Tier 3 branch mirrors Tier 2 evaluate path
                using k_measured (linear, no log-transform) as primary target.
             c) tier3_evaluate dispatch block added in main() -- loads Tier 3
                HDF5, splits df_structural, and calls run_tier_evaluate(3,...).

 FIX-T3-3  MINOR: J_g task weight applied at 1.0 instead of the intended 1.5.
           TIER3_TRAIN_CONFIG names the aux head "J_g_A_cm2" (the dataframe
           column name).  MaskedMultiTaskLoss.task_weights only had the key
           "J_g_log" (1.5), which never matched -- the head silently received
           weight 1.0 (default fallback).  Fix: added "J_g_A_cm2": 1.5 and
           "E_BD_MV_cm": 1.0 to task_weights alongside the existing aliases.
           Both the old alias keys (J_g_log, E_BD) and the new canonical keys
           are present so any tier config using either naming convention works.

==============================================================================
 MP API hotfixes  (applied post Tier 1 epoch-15 review)
 -------------------------------------------------------
 FIX-MP1  Critical: k_total silently None for all MP dielectric entries.
          mp-api >= 0.39 returns DielectricDoc with e_total as a direct
          top-level attribute.  There is NO nested .dielectric sub-object.
          Old code: getattr(doc, "dielectric", None) → always None → k_total
          always None → ~4-6K MP k values silently lost every training run.
          Fix: access e_total directly: getattr(doc, "e_total", None).
          Added try/except around the np.diag() call for malformed tensors.

 FIX-MP2  Moderate: every MP entry was getting functional_code = PBE
          regardless of actual functional.
          run_type and is_hubbard are both invalid fields on the current
          mp-api summary endpoint (both raise MPRestError at runtime).
          Neither is requested in the field lists.  The getattr fallback in
          _parse_mp_entry returns False for is_hubbard, so all MP entries
          receive functional_code = PBE.  Impact is low: GGA+U affects only
          transition-metal oxides (Fe, Co, Ni, Mn) which are a minority of
          the high-k targets; functional_code is metadata only and not a
          model input in the current architecture.  Accurate detection deferred
          to v4 via TaskDoc query after residual analysis.

 FIX-MP3  Minor: MP_ELECTRIC_FIELDS contained stale pre-0.39 field names
          ("total","ionic","electronic","n") alongside the current names.
          The API ignores unknown fields silently but the list was misleading.
          Fix: removed the four stale names; kept e_total/e_ionic/e_electronic.

 FIX-MP4  Minor: progress log on line 592 printed batch start index i
          instead of the count of entries fetched so far.
          Fix: log.info("Fetched %d/%d", min(i+BATCH_SIZE, total), total).

 FIX-DIELECTRIC-SCALAR  (root cause of persistent 755-row Tier 2 dataset)
          DielectricDoc field naming in mp-api >= 0.39:
            "total","ionic","electronic"  →  3×3 TENSOR tuples
            "e_total","e_ionic","e_electronic"  →  SCALAR float (trace avg)
          FIX-MP3 removed "total/ionic/electronic" as "stale pre-0.39 names"
          but they are the current 3×3 tensor fields, not stale.
          What remained in MP_ELECTRIC_FIELDS were the SCALAR float fields.
          _compute_k_from_tensor receives a float, np.array(float).ndim=0,
          tensor dimension check (ndim==2) fails, returns None for every MP
          entry regardless of raw_cache state, API version, or rebuild.
          Fix: _parse_mp_entry now uses _safe_float(doc.e_total) directly.
          e_total is already the isotropic trace-average -- no tensor math
          needed.  Same for k_ionic (_safe_float(e_ionic)) and k_elec.

 Known Limitations (to address before v4 claims)
 ------------------------------------------------
 LIMIT-1  Process-only experimental rows excluded from Tier 3 training.
          run_tier3_finetune() splits df_tier3 into df_structural (has
          atoms_dict) and df_process_only (no crystal structure).  Only
          df_structural is trained on.  df_process_only rows -- experimental
          entries with ALD process params + measured k but no matched DFT
          structure -- are counted, logged, and discarded.

          Impact: the ProcessParamsEncoder and StackContextEncoder branches
          receive gradient signal only from the subset of structural rows that
          also have process annotations, not from the richer process-only set.
          Any claim of "full process-aware Tier 3 learning" is premature until
          this is resolved.

          See TODO(v4) block inside run_tier3_finetune() for three concrete
          implementation paths (structure imputation, composition-only graph,
          separate MLP ensemble) and the prerequisite analysis steps.


 ------------------------------------------------------------------
 PERF-1  __getitem__ overhead eliminated for Tier 1/2.
         Every sample was calling _extract_proc_feats + _extract_stack_feats
         (16 pandas _safe_val lookups per sample) even when no context columns
         exist in the Tier 1/2 dataframe.  At ~250K samples this amounts to
         ~4M wasted pandas row-lookups per epoch.

         Fix: HighKGraphDataset.__init__ checks once whether any context
         column exists in self.df.columns and stores _has_context (bool).
         __getitem__ returns pre-built zero lists directly when False --
         zero pandas overhead, zero per-sample allocation.
         Pre-computed _zero_proc_num/cat and _zero_stack_num/cat lists are
         built once at init and reused for every row.

 PERF-2  GPU encoder + projection forward eliminated for Tier 1/2.
         ProcessParamsEncoder, StackContextEncoder and context_proj all ran
         on every batch even with avail_flag=0 throughout Tier 1/2.
         The ReZero guarantee meant delta=0 anyway but GPU kernel launches
         still occurred, adding latency per batch.

         Fix: _fuse() checks proc_context["avail"].any() and
         stack_context["avail"].any() before dispatching to encoders.
         When both are all-zero (every Tier 1/2 batch) it returns
         crystal_emb directly -- no encoder, no context_proj, no extra memory.
         Numerical result is identical (delta was 0 before; now skipped).
         Tier 3 behaviour is completely unchanged since avail_flag > 0 rows
         cause proc_active or stack_active to be True.

==============================================================================
 Architecture
 -------------
 TIER 1  Foundation pretrain   Full JARVIS-DFT (~55K) + full MP (~69K) + QM9 (~130K)
 TIER 2  Domain fine-tune      All oxide dielectrics k > 10, Eg > 1 eV (~10-15K)
 TIER 3  Project fine-tune     HfO2-family + experimental process data (~1,580)

 Post-review corrections to v2.2 (three observations)
 -------------------------------------------------------
 COR-1  Fusion was NOT pure-ALIGNN when context absent.
        Old fusion_mlp took cat([crystal_emb, alpha*proc, beta*stack]).
        crystal_emb was still transformed by a randomly-initialised MLP
        even when proc=stack=0, breaking Tier 1/2 ≈ pure ALIGNN guarantee.

        Fix -- replace fusion_mlp with residual context_proj:
          ctx   = cat([alpha*proc_emb, beta*stack_emb])   # 128-dim only
          delta = context_proj(ctx)                        # zero-init final layer
          fused = crystal_emb + delta                      # residual

        ReZero guarantee: context_proj[-1] weight and bias both zero-init.
          ctx=0  →  hidden=SiLU(W₁·0+b₁)  →  delta=W₂·hidden=0·…=0
                 →  fused = crystal_emb + 0 = crystal_emb  (exact)
        Tier 1/2 training is now mathematically identical to pure ALIGNN.

 COR-2  Added per-epoch logging of alpha, beta, proc_avail_%, stack_avail_%.
        train_epoch() now returns a dict instead of a float; train() logs
        these fields in the epoch line and stores them in history JSON.
        Example epoch line:
          Epoch  42/300  loss=0.0312  val_MAE=0.0418  ...
                         alpha=0.0017 beta=0.0011  proc=0.0% stack=0.0%

 COR-3  Added --ablate_context CLI flag.
        Forces proc_context=stack_context=None in train_epoch, evaluate,
        and evaluate_multitask regardless of what the dataframe contains.
        Enables controlled ablation: same Tier 3 data, zero context signal.
        Use:
          python highk_alignn_train_v2_2.py \\
              --mode tier3_finetune \\
              --weights checkpoints/tier2_best.pt \\
              --ablate_context

 Changes from v2.1 -> v2.2  (process/stack context branches)
 ---------------------------------------------------------------
 FIX-V22  Process-conditioned multi-head architecture.
           Two new input branches added alongside the ALIGNN crystal graph path:

           ProcessParamsEncoder  -- ALD/deposition process parameters
             Numerical: substrate_temp_C, anneal_temp_C, anneal_duration_min
                        (log1p), growth_rate_A_per_cycle, n_cycles,
                        film_thickness_A, pressure_mTorr (log1p)
             Categorical: anneal_ambient, precursor_type, oxidant_type
                          (learned nn.Embedding tables, dim=8 each)
             Output: 2-layer MLP → proc_emb [64]

           StackContextEncoder  -- device stack / interface context
             Numerical: IL_thickness_A, CET_target_A
             Categorical: substrate, IL_type, top_electrode, stack_config
                          (learned nn.Embedding tables, dim=8 each)
             Output: 2-layer MLP → stack_emb [64]

           Fusion (gated residual):
             fused = FusionMLP( cat([crystal_emb,
                                     alpha * proc_emb,
                                     beta  * stack_emb]) )
             FusionMLP: Linear(384→256) + SiLU + Dropout
             Task heads unchanged: still Linear(256→1) × N tasks.

           Non-degradation guarantees:
           a) alpha, beta initialised to 1e-3 (near-zero). At the start of
              Tier 1/2 training the model is numerically ≈ pure ALIGNN.
           b) gate = avail_flag (1.0 if any proc/stack feature present, 0.0
              if all absent). For all Tier 1/2 rows (JARVIS, MP, QM9) every
              context column is absent → avail_flag=0 → proc_emb = stack_emb
              = 0 every batch → zero contribution to fused embedding.
           c) Branches compiled into ALL tiers (not Tier 3-only) for clean
              Tier 1→2→3 weight transfer. strict=False load carries backbone,
              task head, and encoder weights cleanly across tiers.
           d) fit_encoder_stats() called AFTER weight load in every tier so
              Tier 3 training data overwrites uninformative Tier 1/2 zeros.
           e) functional_code remains a separate scalar field in the dataframe
              (FIX5, v2.1); it is NOT routed through ProcessParamsEncoder.

 Changes from v2.0 -> v2.1  (code-review observations)
 -------------------------------------------------------
 FIX-OBS1  Multi-task loss was only training the primary head.
           train_epoch() built targets_dict = {target_col: target} and passed
           it straight to MaskedMultiTaskLoss.  Since the criterion skips any
           head whose key is absent from targets, all auxiliary heads received
           zero gradient despite FIX2's NaN-tensor collation being correct.
           Fix: train_epoch() now iterates preds.keys() after building
           targets_dict, looks up each aux head in batch["aux_targets"] via
           direct name match then TASK_TO_COLUMN fallback, and adds the tensor
           to targets_dict.  All heads now receive gradient on every batch
           where their target is non-NaN.

 FIX-OBS2  k_measured / k_total column mismatch caused silent dataset failure.
           Both _parse_jarvis_entry and _parse_mp_entry stored the computed
           DFT total (ionic + elec) under "k_measured" only.  TIER1 aux_targets
           and TIER2 target both reference "k_total", which never existed in
           the dataframe, causing HighKGraphDataset to see an all-NaN target
           column (or KeyError depending on pandas version).
           Fix (six locations):
           a) _parse_jarvis_entry: adds "k_total": k_total alongside k_measured
           b) _parse_mp_entry:     adds "k_total": k_total alongside k_measured
           c) _parse_qm9_entry:    adds "k_total": None  (alpha proxy ≠ crystal
                                   dielectric; must not pollute k_total target)
           d) load_experimental_process_db: adds df["k_total"] = df["k_measured"]
              (measured k IS total k for experimental entries)
           e) run_tier2_finetune: pre-filter changed from k_measured.notna()
              to k_total.notna() for column consistency
           f) TASK_TO_COLUMN: added "k_total": "k_total" so evaluate_multitask
              routes the k_total head to the correct batch key

 FIX-OBS2  Collateral -- task head names now explicitly derived from tier config
 collateral in run_tier2_finetune and run_tier3_finetune.
           Previously both used HighKALIGNN(n_output_tasks=4) which silently
           fell back to default heads ["k_measured","band_gap","J_g_log","E_BD"]
           -- inconsistent with tier configs and causing train_epoch's new
           FIX-OBS1 routing to misfire.  Both functions now use:
               task_names = [cfg["target"]] + cfg["aux_targets"]
               HighKALIGNN(..., task_names=task_names)

 FIX-OBS2  Collateral -- MaskedMultiTaskLoss.task_weights extended to cover
 collateral  "k_total" (2.0), "formation_energy_per_atom" (1.0), "e_above_hull"
           (0.5).  High-k upweighting check extended from k_measured-only to
           also cover "k_total" head (Tier 1/2 dielectric prediction).

 Validation of v2 fixes against base file
 -----------------------------------------
 FIX1  output_features=256            ALREADY PRESENT in base -- no change needed.
       (ALIGNN_BASE_CONFIG + ALIGNNConfig both correct in base file)

 FIX2  Multi-task test evaluation     APPLIED -- three changes:
       a) __getitem__ aux_targets: None -> NaN tensor (enables collate stacking)
       b) collate_fn: now stacks aux_targets per-task key in returned batch dict
       c) evaluate_multitask() + print_multitask_results() added to ALIGNNTrainer
       Test output now shows: MAE | RMSE | MAD:MAE | Paper ref | N | Coverage
       for every task head across all tiers.

 FIX3  Graph param names              ALREADY PRESENT in base -- no change needed.
       (use_lattice_prop and compute_line_graph already correct in base file)

 FIX4  Cross-source structural dedup  APPLIED -- new method + call in build_tier1.
       deduplicate_cross_source() removes MP entries whose crystal structure
       already exists in JARVIS-DFT (identified via pymatgen StructureMatcher).
       Prevents same crystal appearing twice with contradictory DFT targets.
       --skip_cross_dedup flag available to bypass when speed is priority.

 FIX5  Functional-aware labeling      APPLIED -- three locations:
       _parse_jarvis_entry: adds dft_functional=OptB88vdW, functional_code=0,
                            band_gap_optb88vdw=band_gap, band_gap_pbe=NaN
       _parse_mp_entry:     detects GGA+U via is_hubbard; adds
                            functional_code, band_gap_pbe (is_hubbard=False→PBE)
       build_tier1:         ensures all three sources have functional columns
                            before concatenation

 MAD:MAE  All three tier test outputs now print full table including
          MAD:MAE ratio and paper benchmark reference (v/^ flags).
          Full-dataset MAD used when df_full is passed (matches paper reporting).

 Usage
 -----
 python highk_alignn_train_v2.py --mode tier1_pretrain
 python highk_alignn_train_v2.py --mode tier1_pretrain --skip_cross_dedup
 python highk_alignn_train_v2.py --mode tier2_finetune --weights checkpoints/tier1_best.pt
 python highk_alignn_train_v2.py --mode tier3_finetune --weights checkpoints/tier2_best.pt
 python highk_alignn_train_v2.py --mode full_pipeline
 python highk_alignn_train_v2.py --mode extract_only
 python highk_alignn_train_v2.py --mode dataset_stats

Requirements
 -------------
 pip install alignn jarvis-tools mp-api pymatgen torch dgl \
             scikit-learn pandas numpy tqdm h5py python-dotenv
==============================================================================
"""

import os
import sys
import json
import math
import time
import hashlib
import logging
import argparse
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# -- ALIGNN imports ------------------------------------------------------------
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph, StructureDataset
from alignn.train import train_dgl

# -- JARVIS/Materials Science imports -----------------------------------------
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms as JAtoms
from pymatgen.io.jarvis import JarvisAtomsAdaptor

# -- Materials Project API -----------------------------------------------------
from mp_api.client import MPRester

warnings.filterwarnings("ignore")

# ==============================================================================
# ==============================================================================
# DISTRIBUTED TRAINING (DDP)
# ==============================================================================
# Launch with:  torchrun --nproc_per_node=2 highk_alignn_train_v3.py --mode tier1_pretrain
#
# torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE env vars.
# Single-GPU runs (python script.py ...) work unchanged — _DIST["active"]=False.
#
# Why DDP and not DataParallel:
#   DGL graphs (created per-sample in __getitem__) cannot be split across GPUs
#   by DataParallel's scatter(). DDP avoids this: each process builds its own
#   DGLGraph from its data shard, runs a full forward pass, and AllReduce
#   synchronises only the float32 gradients — no graph objects cross the wire.

_DIST: Dict[str, object] = {"rank": 0, "world": 1, "active": False, "device": "cpu"}


def init_dist() -> bool:
    """
    Initialise NCCL process group when launched under torchrun.
    Returns True if distributed mode is active.
    Called once at the top of main() before anything else.
    """
    if "RANK" not in os.environ:
        _DIST["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        return False
    dist.init_process_group(backend="nccl")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    _DIST.update({
        "rank":   rank,
        "world":  world,
        "active": True,
        "device": f"cuda:{rank}",
    })
    log.info("[rank %d/%d] DDP active on cuda:%d", rank, world, rank)
    return True


def is_rank0() -> bool:
    """True for the primary process (rank 0), or in single-GPU mode."""
    return _DIST["rank"] == 0


def dist_barrier():
    """All-ranks barrier — no-op in single-GPU mode."""
    if _DIST["active"]:
        dist.barrier()


def shutdown_dist():
    """
    Destroy the NCCL process group.
    Must be called before process exit to avoid the warning:
      "destroy_process_group() was not called before exit which can leak resources"
    Called in the finally block of main() so it fires on every exit path
    including exceptions, early returns, and normal completion.
    No-op in single-GPU mode.
    """
    if _DIST["active"] and dist.is_initialized():
        dist.destroy_process_group()
        log.info("NCCL process group destroyed cleanly.")


# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

ROOT          = Path("highk_project")
DATA_ROOT     = ROOT / "data"
CKPT_ROOT     = ROOT / "checkpoints"
LOG_ROOT      = ROOT / "logs"
REPORT_ROOT   = ROOT / "reports"

for d in [DATA_ROOT, CKPT_ROOT, LOG_ROOT, REPORT_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

MP_API_KEY = os.environ.get("MP_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_ROOT / f"pipeline_{datetime.date.today()}.log"),
    ],
)
log = logging.getLogger(__name__)

# -- Target element groups -----------------------------------------------------
TIER3_ELEMENTS  = {"Hf", "Zr"}          # primary HfO2 family

# High-k oxide elements used as structural donor pool for experimental rows
HIGH_K_DONOR_ELEMENTS = {"Hf","Zr","Al","Ti","Ta","Sr","La","Y",
                         "Ba","Nb","Ga","In","Sc","Ce","Pr","Nd"}

TIER2_CATIONS   = {                       # reference only -- no longer used as a filter
    "Hf", "Zr", "Ti", "La", "Ce", "Pr", "Nd", "Gd", "Dy", "Y", "Lu",
    "Al", "Ga", "In", "Si", "Ge", "Sn", "Nb", "Ta", "W", "Mo",
    "Ba", "Sr", "Ca", "Mg",
}

# Minimum dielectric constant for Tier 2 training inclusion.
# 3.9 = SiO2 reference value -- any material above this is a gate dielectric
# improvement candidate.  Changed from 10.0 which excluded 1,073 valid entries
# (Al2O3 k~9.1, MgO k~9.8, etc.) that define the lower boundary of the k spectrum.
TIER2_K_MIN     = 3.9

# -- ALIGNN hyperparameters per tier (from paper Table 1 + fine-tune scaling) -
ALIGNN_BASE_CONFIG = dict(
    alignn_layers  = 4,
    gcn_layers     = 4,
    edge_input_features = 80,
    triplet_input_features    = 40,
    embedding_features  = 64,
    hidden_features     = 256,
    output_features     = 256,  # Match hidden_features for embedding output
)

TIER1_TRAIN_CONFIG = dict(
    epochs         = 300,
    batch_size     = 64,
    learning_rate  = 1e-3,
    weight_decay   = 1e-5,

    # -- Scheduler ----------------------------------------------------------
    scheduler      = "onecycle",
    # pct_start=0.4: 120-epoch LR warmup (vs ALIGNN default 0.3 = 90 epochs).
    # Longer warmup suits our 250K multi-source dataset with gradient variance
    # from mixed functionals (OptB88vdW, PBE, GGA+U, B3LYP).
    pct_start      = 0.4,

    # -- Targets ------------------------------------------------------------
    loss           = "mse",
    target         = "formation_energy_per_atom",
    aux_targets    = ["band_gap", "k_total_log"],
    log_transform_aux = {"k_total_log": "k_total"},

    # -- Functional-weighted loss -------------------------------------------
    # QM9 B3LYP molecular entries (~130K) contribute noise for oxide targets.
    # r2SCAN/GGA+U are rare but highest-quality for TM-oxide dielectrics.
    # Quality hierarchy: r2SCAN > GGA+U ≈ OptB88vdW > PBE >> B3LYP(QM9)
    functional_loss_weights = {
        0: 1.5,   # OptB88vdW (JARVIS) -- DFPT quality
        1: 1.0,   # PBE       (MP)     -- baseline
        2: 2.0,   # r2SCAN    (MP)     -- best functional, rare → upweight
        3: 1.5,   # GGA+U     (MP)     -- TM-oxides, relevant to high-k
        4: 0.3,   # B3LYP     (QM9)    -- molecular DFT → downweight heavily
    },

    # -- Split --------------------------------------------------------------
    train_ratio    = 0.80,
    val_ratio      = 0.10,
    test_ratio     = 0.10,

    # -- Early stopping -----------------------------------------------------
    # ALIGNN paper: 300 full epochs, no early stopping.
    # Previous run stopped at epoch 58 (best ep28) -- insufficient exploration.
    # early_stopping=False runs all 300 epochs; best checkpoint saved throughout.
    # min_epochs=200 acts as a floor if early_stopping is re-enabled.
    early_stopping = False,
    min_epochs     = 200,
    patience       = 50,

    # -- Gradient clipping --------------------------------------------------
    max_grad_norm  = 1.0,
)

TIER2_TRAIN_CONFIG = dict(
    epochs            = 150,
    batch_size        = 32,
    # Fine-tuning LR: 2e-4 is already 5× lower than typical Tier 1 LR (1e-3).
    # A further reduction to 1e-4 is applied after the first freeze phase
    # by the unfreeze scheduler below.
    learning_rate     = 2e-4,
    learning_rate_unfreeze = 1e-4,  # LR after unfreeze (epoch > unfreeze_after)
    weight_decay      = 1e-5,
    scheduler         = "cosine",
    loss              = "mse",

    # LOG TRANSFORM (root cause fix for MAE plateau at 5.x):
    # k_total spans 3.9 to 386+ (100× dynamic range).  MSE on linear k
    # converges to predicting the distribution mean for all inputs, giving a
    # stable floor MAE of ~5–7 regardless of epochs.  Training on log(k_total)
    # normalises the range to ~1.4 decades and gives equal gradient weight to
    # materials at k=5 and k=300.
    #
    # Training target is log(k_total); evaluation reports both log-MAE and
    # original-unit MAE (exp-space) for interpretability.
    log_transform     = True,          # apply np.log to k_total before training
    log_original_col  = "k_total",     # source column
    target            = "k_total_log", # derived column used during training

    # AUXILIARY TARGETS:
    # Removed e_above_hull: Tier 2 is pre-filtered to stable oxides (near convex
    # hull), so e_above_hull clusters near 0 → MAD:MAE < 1 → noise not signal.
    # Replaced with formation_energy_per_atom which has meaningful variance
    # across the oxide dielectric family.
    aux_targets       = ["band_gap", "formation_energy_per_atom"],

    train_ratio       = 0.80,
    val_ratio         = 0.10,
    test_ratio        = 0.10,

    # UNFREEZE SCHEDULE: freeze first 2 ALIGNN layers for the first N epochs
    # (stability), then unfreeze all layers at a lower LR.
    freeze_layers     = 2,
    unfreeze_after    = 20,

    # -- Early stopping -----------------------------------------------------
    early_stopping    = True,
    min_epochs        = 50,   # don't stop before 50 even if val_MAE plateaus early
    patience          = 50,

    # -- Phase A: all oxide rows (v4 change #12) ----------------------------
    # Train on ALL df_tier2 (~30K rows) with band_gap as primary target for
    # phase_a_epochs before switching to k_total_log on the 5K subset.
    # This 6× more data in Phase A consolidates the backbone's oxide
    # representations before specialising for k_total in Phase B.
    phase_a_target    = "band_gap",
    phase_a_epochs    = 30,

    # -- Gradient clipping --------------------------------------------------
    max_grad_norm     = 1.0,
)

TIER3_TRAIN_CONFIG = dict(
    epochs         = 150,

    # FIX-T3-9C: batch_size 16→8. With ~170 training rows, batch_size=16 gives
    # only ~10 batches/epoch.  At 8 we get ~20 batches/epoch — experimental
    # rows appear more frequently and the process encoder receives more
    # gradient updates per epoch relative to DFT rows.
    batch_size     = 32,

    # FIX-T3-6: LR reduced 5× (5e-5 → 1e-5, applied in v4.5.3).
    learning_rate  = 5e-5,
    weight_decay   = 1e-4,
    scheduler      = "cosine",
    loss           = "mse",
    max_grad_norm  = 1.0,

    # FIX-T3-9A: switch to log-space target (mirrors Tier 2 k_total_log).
    # k_measured spans 3.9–386+ (same 100× dynamic range as k_total).
    # Linear-space MSE is dominated by high-k outliers; log-space aligns
    # Tier 3 supervision with the log-space backbone representations
    # learned in Tier 2.  run_tier3_finetune creates k_measured_log column
    # via log(k_measured) before build_dataloader, exactly mirroring Tier 2.
    target            = "k_total_log",    # UNIFY-K: was "k_measured_log" (v4.5.3-v4.5.10)
    log_transform     = True,
    log_original_col  = "k_total",         # UNIFY-K: was "k_measured" — aliased in build_tier3

    aux_targets    =[],
    train_ratio    = 0.70,
    val_ratio      = 0.15,
    test_ratio     = 0.15,
    freeze_backbone = True,
    unfreeze_backbone_after = 50,
    unfreeze_backbone_lr = 1e-5,

    # FIX-T3-9B: experimental row upweighting.
    # After structure imputation df_structural has ~165 DFT k_measured rows
    # (k_total aliased) vs ~80 experimental.  DFT rows already predict well
    # from Tier 2 → near-zero per-sample loss → experimental gradient drowned.
    # 3× upweight ensures the process-encoder correction signal dominates.
    exp_sample_weight = 1.0,

    # FIX-T3-9C: explicit early stopping tuned for small noisy dataset.
    early_stopping = True,
    min_epochs     = 80,    # was 40 — backbone + process encoder need more
                            #           epochs to jointly converge in log-space
    patience       = 40,    # was 30 — more tolerance for noisy val_MAE
    # Worst-case stop: min_epochs(60) + patience(40) = 100 = full training

    # ── FIX-T3-PHASE-A: band_gap consolidation before k_measured fine-tune ──
    # Run phase_a_steps gradient steps on full df_tier2 (band_gap primary target)
    # before Phase B k_measured_log fine-tuning on the small HfO2-family set.
    # Prevents catastrophic backbone forgetting: band_gap MAD:MAE 7.3 → 0.72
    # without this fix.  Mirrors the proven Tier 2 phase_a_epochs=30 strategy.
    # phase_a_lr: same gentle 1e-5 as Phase B to avoid overshooting Tier 2 opt.
    #
    # FIX-T3-PHASEACAP: Changed from phase_a_epochs=20 to phase_a_steps=200.
    # 20 epochs on 49K rows = 116,860 steps = 93% of all Tier 3 gradient steps.
    # Phase A was DOMINATING Tier 3, re-specialising backbone for full-oxide
    # band_gap and leaving only 8,300 steps for Phase B k_measured_log.
    # 200 steps = 0.03 epochs on 49K rows — brief anchor, not a training phase.
    # New Phase A:Phase B ratio = 200:1,200 = 0.17× (was 14.1×).
    phase_a_steps  = 0,          # RESIDUAL-PROC: skip Phase A for frozen DFT-base residual run
    phase_a_target = "band_gap",
    phase_a_lr     = 5e-5,

    # RESIDUAL-PROC: train a DFT-only base model plus a bounded ALD-process
    # correction in log(k) space, instead of fusing process embeddings into
    # the crystal embedding. This protects the strong Tier-2/Tier-3 DFT-only
    # baseline and lets process parameters explain only the residual error.
    # final_log_k = frozen_dft_base_log_k + bound * tanh(process_delta_raw)
    # A bound of 0.10 log units corresponds to roughly ±10.5% multiplicative
    # correction in k. Categorical process embeddings are disabled by default
    # for this residual path because the Tier-3 process dataset is small and
    # categories can easily memorize paper-specific effects.
    bounded_process_residual = True,
    process_delta_bound      = 0.10,
    process_delta_use_categorical = False,
    process_delta_train_base_head = False,

    # RESIDUAL-PROC-DIAG: save epoch-1 checkpoint and log epoch-0/epoch-1/best
    # validation/test metrics. This is critical when best_epoch=1: it tells us
    # whether Tier-3 training adds value or whether the Tier-2 checkpoint was
    # already the best model before process-residual fitting.
    save_epoch1_checkpoint = True,
)

# -- FIX5: DFT functional codes ------------------------------------------------
FUNCTIONAL_CODE = {
    "OptB88vdW": 0,   # JARVIS-DFT throughout
    "PBE":       1,   # MP standard GGA
    "r2SCAN":    2,   # MP newer calculations
    "GGA+U":     3,   # MP transition-metal oxides
    "B3LYP":     4,   # QM9 molecular DFT
}

# -- Task head name -> dataset column ------------------------------------------
TASK_TO_COLUMN = {
    "k_measured":               "k_measured",
    "k_total_log":              "k_total_log",   # UNIFY-K: was "k_measured_log" (Tier 3 log-space target)
    "k_total":                  "k_total",    # DFT total (ionic + elec); Tier 1/2 target
    "band_gap":                 "band_gap",
    "formation_energy_per_atom":"formation_energy_per_atom",
    "e_above_hull":             "e_above_hull",
    "J_g_log":                  "J_g_A_cm2",
    "E_BD":                     "E_BD_MV_cm",
}

# -- Paper benchmark MAD:MAE (Choudhary & DeCost 2021, Tables 2 & 3) ----------
PAPER_MAD_MAE_BENCHMARK = {
    "formation_energy_per_atom": 26.06,   # JARVIS-DFT Ef
    "band_gap":                   7.07,   # JARVIS-DFT OptB88vdW
    "k_total":                    1.63,   # JARVIS-DFT ε DFPT elec+ionic (all tiers)
    "k_total_log":                1.63,   # UNIFY-K: log-scale head uses same benchmark
    "J_g_log":                    None,
    "E_BD":                       None,
}

# ==============================================================================
# PROCESS / STACK CONTEXT BRANCH CONFIG  (v2.2)
# ==============================================================================
# These dicts drive HighKGraphDataset feature extraction AND encoder construction
# so the two are always in sync.  Column names must match the experimental CSV.
# "log_normalize" columns are transformed as log1p(max(x, 0)) before encoding.
# Categorical vocab lists are order-sensitive: index = position in list.
# Unknown / absent values map to vocab_size (the extra embedding slot).
# ==============================================================================

PROCESS_PARAMS_FEATURES: Dict[str, Any] = {
    "numerical": [
        "substrate_temp_C",          # ALD chuck / substrate temperature
        "anneal_temp_C",             # post-deposition anneal temperature
        "anneal_duration_min",       # anneal duration (log-normalised)
        "growth_rate_A_per_cycle",   # ALD GPC
        "n_cycles",                  # total ALD cycles → thickness proxy
        "film_thickness_A",          # target/nominal film thickness
        "pressure_mTorr",            # process chamber pressure (log-normalised)
    ],
    "log_normalize": ["anneal_duration_min", "pressure_mTorr"],
    "categorical": {
        # BUG-5 FIX: expanded from 33-row seed CSV vocab to cover all values
        # in the updated 120-row experimental CSV.  Old entries are kept at the
        # SAME index positions so checkpoints loaded with strict=False remain
        # compatible — new entries are appended after the originals.
        # Unknown values still fall to the final "other" slot, but dominant
        # precursors (TDMAH, TEMAH) and oxidants now get their own embeddings.
        "anneal_ambient": [
            "N2", "O2", "forming_gas", "vacuum",   # original indices 0-3
            "Ar", "ambient", "other",               # BUG-5: new CSV values
        ],
        "precursor_type": [
            "TDMA-Hf", "HfCl4", "TEMAZ", "TDMAZ", "other",  # original 0-4
            "TDMAHf",  "TDMAH",                               # Hf TDMA variants
            "TEMA-Hf", "TEMAH",                               # Hf TEMA variants
            "TDMAZr",  "TMA",                                 # Zr + Al (HZO, HfAlO)
        ],
        "oxidant_type": [
            "H2O", "O3", "O2_plasma", "other",     # original indices 0-3
            "O2 plasma",                            # BUG-5: space variant in CSV
            "O2", "H2O_O3",                         # BUG-5: additional CSV values
        ],
    },
    "embed_dim":  8,    # per-categorical embedding dimension
    "output_dim": 64,   # final proc_emb dimension
}

STACK_CONTEXT_FEATURES: Dict[str, Any] = {
    "numerical": [
        "IL_thickness_A",    # interfacial layer thickness (SiO2 or SiON)
        "CET_target_A",      # capacitance equivalent thickness design target
    ],
    "log_normalize": [],
    "categorical": {
        "substrate":     ["Si", "SiO2_Si", "TiN_Si", "GaAs", "other"],
        "IL_type":       ["SiO2", "SiON", "none", "other"],
        "top_electrode": ["TiN", "TaN", "W", "Al", "poly_Si", "other"],
        "stack_config":  ["MOS", "MIM", "MFIM", "MFIS", "other"],
    },
    "embed_dim":  8,
    "output_dim": 64,
}

# ==============================================================================
# SECTION 1 -- DATA EXTRACTION
# ==============================================================================

class DatasetExtractor:
    """
    Pulls full datasets from JARVIS-DFT, Materials Project, and QM9.
    Implements the row_hash deduplication strategy from ScalableDatasetManager.
    """

    # -- JARVIS dataset keys used in this pipeline -----------------------------
    JARVIS_DATASET_KEYS = {
        "dft_3d":         "Full JARVIS-DFT 3D dataset (~55,722 entries)",
        "qm9_std_jctc":   "QM9 standardized via JCTC (~130,829 molecules)",
    }

    # -- JARVIS property field mapping -> unified schema names -----------------
    JARVIS_FIELD_MAP = {
        "jid":                          "jid",
        "formula":                      "formula",
        "formation_energy_peratom":     "formation_energy_per_atom",
        "optb88vdw_bandgap":            "band_gap",
        "mbj_bandgap":                  "band_gap_mbj",
        "epsilon_ionic":                "epsilon_ionic",
        "epsilon_elec":                 "epsilon_elec",
        "ehull":                        "e_above_hull",
        "bulk_modulus_kv":              "bulk_modulus",
        "shear_modulus_gv":             "shear_modulus",
        "atoms":                        "atoms_dict",
    }

    # -- MP property fields requested from dielectric API endpoint -----------
    # DielectricDoc field naming in mp-api >= 0.39 (confirmed via schema):
    #
    #   TENSOR fields (tuple[tuple[float,float,float] x3]):
    #     "total"       -- full 3×3 total dielectric tensor
    #     "ionic"       -- full 3×3 ionic contribution tensor
    #     "electronic"  -- full 3×3 electronic contribution tensor
    #
    #   SCALAR fields (float, already isotropic trace-average):
    #     "e_total"       -- isotropic total dielectric constant
    #     "e_ionic"       -- isotropic ionic contribution
    #     "e_electronic"  -- isotropic electronic contribution
    #     "n"             -- refractive index
    #
    # FIX-MP3 incorrectly removed "total","ionic","electronic" as "stale
    # pre-0.39 names" -- they are current and are the TENSOR fields.
    # We request the SCALAR fields here because _parse_mp_entry now uses
    # _safe_float() directly on e_total/e_ionic/e_electronic rather than
    # routing them through _compute_k_from_tensor() (which expects tensors).
    MP_ELECTRIC_FIELDS = [
        "material_id", "formula_pretty",
        "e_total", "e_ionic", "e_electronic",   # scalar floats -- used directly
    ]

    def __init__(self, cache_dir: Path = DATA_ROOT / "raw_cache"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1a. JARVIS-DFT full pull
    # --------------------------------------------------------------------------
    def pull_jarvis_dft(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Pull full JARVIS-DFT 3D dataset via jarvis-tools figshare module.

        The jarvis-tools library downloads the JSON from Figshare and caches
        it locally under ~/.jarvis/ on first call. Subsequent calls use cache.

        Returns ~55,722 entries with atoms, formation energy, band gap,
        dielectric (where available), ehull, elastic moduli.
        """
        cache_path = self.cache_dir / "jarvis_dft_3d_full.h5"

        if cache_path.exists() and not force_refresh:
            log.info("Loading cached JARVIS-DFT dataset from %s", cache_path)
            return pd.read_hdf(cache_path, key="data")

        log.info("Downloading full JARVIS-DFT 3D dataset (~55,722 entries)...")
        log.info("This is a ~400 MB download from Figshare -- takes 5–10 min first run.")

        raw = jdata("dft_3d")   # jarvis-tools handles download + caching
        log.info("Raw JARVIS-DFT entries loaded: %d", len(raw))

        rows = []
        for entry in tqdm(raw, desc="Parsing JARVIS-DFT entries"):
            row = self._parse_jarvis_entry(entry, source="JARVIS-DFT")
            if row is not None:
                rows.append(row)

        df = pd.DataFrame(rows)
        log.info("Parsed JARVIS-DFT entries: %d", len(df))

        # Save to cache
        df.to_hdf(cache_path, key="data", mode="w", complevel=6, complib="blosc")
        log.info("JARVIS-DFT cached at %s", cache_path)
        return df

    def _parse_jarvis_entry(self, entry: dict, source: str) -> Optional[dict]:
        """
        Parse a single JARVIS-DFT entry dict into the unified schema row.

        Computes:
        - k_total = epsilon_ionic + epsilon_elec (trace average of tensor)
        - row_hash for deduplication
        - has_structure flag
        """
        try:
            # -- Extract atoms ----------------------------------------------
            atoms_dict = entry.get("atoms")
            if atoms_dict is None:
                return None

            j_atoms = JAtoms.from_dict(atoms_dict)
            formula = j_atoms.composition.reduced_formula
            has_structure = True

            # -- Extract dielectric constant -------------------------------
            eps_ionic = entry.get("epsilon_ionic", None)
            eps_elec  = entry.get("epsilon_elec",  None)
            k_total   = self._compute_k_from_tensor(eps_ionic, eps_elec)

            # -- Extract scalar properties ---------------------------------
            band_gap     = self._safe_float(entry.get("optb88vdw_bandgap"))
            band_gap_mbj = self._safe_float(entry.get("mbj_bandgap"))
            Ef           = self._safe_float(entry.get("formation_energy_peratom"))
            e_hull       = self._safe_float(entry.get("ehull"))
            bulk_mod     = self._safe_float(entry.get("bulk_modulus_kv"))
            shear_mod    = self._safe_float(entry.get("shear_modulus_gv"))

            # -- Row hash --------------------------------------------------
            jid      = entry.get("jid", "")
            row_hash = hashlib.md5(f"JARVIS_{jid}_{formula}".encode()).hexdigest()[:12]

            return {
                "source":                    source,
                "jid":                       jid,
                "mp_id":                     None,
                "formula":                   formula,
                "k_measured":                k_total,   # DFT total stored under k_measured for Tier 3 compat
                "k_total":                   k_total,   # explicit alias used by Tier 1/2 configs
                "k_ionic":                   self._compute_k_from_tensor(eps_ionic, None),
                "k_elec":                    self._compute_k_from_tensor(None, eps_elec),
                "band_gap":                  band_gap,
                "band_gap_mbj":              band_gap_mbj,
                "formation_energy_per_atom": Ef,
                "e_above_hull":              e_hull,
                "bulk_modulus":              bulk_mod,
                "shear_modulus":             shear_mod,
                "has_structure":             has_structure,
                "atoms_dict":                json.dumps(atoms_dict),
                # FIX5: functional labeling -- JARVIS always uses OptB88vdW
                "dft_functional":            "OptB88vdW",
                "functional_code":           FUNCTIONAL_CODE["OptB88vdW"],
                "band_gap_optb88vdw":        band_gap,
                "band_gap_pbe":              np.nan,
                "row_hash":                  row_hash,
                "tier":                      None,
                "date_added":                datetime.date.today().isoformat(),
            }

        except Exception as e:
            log.debug("Skipping JARVIS entry %s: %s", entry.get("jid", "?"), e)
            return None

    # --------------------------------------------------------------------------
    # 1b. Materials Project full pull
    # --------------------------------------------------------------------------
    def pull_materials_project(
        self, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Pull full Materials Project dataset via mp-api.

        Pulls ALL entries with dielectric data (~4,000–6,000 entries with k)
        PLUS all oxide entries with band gap data (~50,000+ entries).

        Two separate queries:
        1. Dielectric query  -- entries WITH epsilon computed (smaller set, has k)
        2. Oxide query       -- all oxide band gap entries (larger, no k but useful
                               for pretraining band gap and formation energy targets)
        """
        cache_path = self.cache_dir / "mp_full.h5"

        if cache_path.exists() and not force_refresh:
            log.info("Loading cached MP dataset from %s", cache_path)
            return pd.read_hdf(cache_path, key="data")

        if not MP_API_KEY:
            log.warning(
                "MP_API_KEY not set. Set env var MP_API_KEY to enable MP pull. "
                "Skipping Materials Project download."
            )
            return pd.DataFrame()

        log.info("Pulling Materials Project -- dielectric entries first...")
        rows = []

        with MPRester(MP_API_KEY) as mpr:
            # ── Query 1: single summary endpoint query for all dielectric materials ─
            #
            # ARCHITECTURE CHANGE (replaces two-query join):
            # Previous approach:
            #   a) mpr.materials.dielectric.search()   → gets e_total (no structure)
            #   b) mpr.materials.summary.search()      → gets structure (no e_total)
            #   c) Join on material_id                 → frequently incomplete
            #      → has_structure=False for most entries
            #      → Tier 2 training dataset stuck at 755 rows
            #
            # New approach:
            #   SummaryDoc (confirmed via sandbox schema inspection) has BOTH:
            #     e_total, e_ionic, e_electronic  (scalar float fields)
            #     structure                       (pymatgen Structure)
            #   One query, no join, has_structure always True.
            #   has_props=["dielectric"] confirmed as a valid summary filter.
            #   e_total/e_ionic/e_electronic confirmed as valid summary fields.
            #
            log.info("  MP Query 1: summary endpoint -- materials with dielectric data...")
            docs_dielectric = mpr.materials.summary.search(
                has_props=["dielectric"],
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "band_gap", "formation_energy_per_atom", "energy_above_hull",
                    "is_metal",
                    "e_total",       # total dielectric constant (ionic + electronic)
                    "e_ionic",       # ionic contribution
                    "e_electronic",  # electronic contribution
                ],
            )
            log.info("  MP dielectric entries returned: %d", len(docs_dielectric))

            # Diagnostic counters
            n_parsed        = 0
            n_e_total_set   = 0
            n_k_total_set   = 0
            n_k_above_10    = 0
            n_has_structure = 0
            _sample_etotal  = []

            for doc in tqdm(docs_dielectric, desc="Parsing MP dielectric entries"):
                # doc IS SummaryDoc: has structure + e_total + band_gap + Ef
                # Pass as summary_doc so _parse_mp_entry uses the summary branch
                # which accesses summary_doc.structure, summary_doc.band_gap, etc.
                raw_et = getattr(doc, "e_total", "MISSING")
                if len(_sample_etotal) < 5:
                    _sample_etotal.append(
                        f"{getattr(doc,'material_id','?')} e_total={raw_et!r:.40}"
                    )
                if raw_et not in (None, "MISSING"):
                    n_e_total_set += 1

                row = self._parse_mp_entry(doc, has_dielectric=True, summary_doc=doc)
                if row is not None:
                    n_parsed += 1
                    rows.append(row)
                    if row.get("k_total") is not None:
                        n_k_total_set += 1
                        if row["k_total"] > 10:
                            n_k_above_10 += 1
                    if row.get("has_structure"):
                        n_has_structure += 1

            log.info(
                "MP dielectric parse audit (single-query):"
                "\n  entries from summary search   : %d"
                "\n  entries with e_total present  : %d  (%.1f%%)"
                "\n  rows successfully parsed       : %d"
                "\n  rows with k_total non-null     : %d  (%.1f%% of parsed)"
                "\n  rows with k_total > 10         : %d  (Tier 2 candidates)"
                "\n  rows with has_structure=True   : %d  (%.1f%% of parsed)"
                "\n  ── both k_total and has_structure should be ~100%% ──────────",
                len(docs_dielectric),
                n_e_total_set,
                100.0 * n_e_total_set / max(len(docs_dielectric), 1),
                n_parsed,
                n_k_total_set,
                100.0 * n_k_total_set / max(n_parsed, 1),
                n_k_above_10,
                n_has_structure,
                100.0 * n_has_structure / max(n_parsed, 1),
            )
            if _sample_etotal:
                log.info("MP e_total sample (first 5):\n  %s",
                         "\n  ".join(_sample_etotal))

            # -- Query 2: all oxide entries (broader -- no dielectric filter) --
            # This adds the ~60K oxide entries that have Ef + Eg but no k.
            # Critically important for Tier 1 pretraining -- teaches oxide physics.
            log.info("  MP Query 2: all oxide entries (for Ef + Eg pretraining)...")
            docs_oxides = mpr.materials.summary.search(
                elements=["O"],
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "band_gap", "energy_above_hull",
                    "formation_energy_per_atom",
                ],
            )
            log.info("  MP oxide entries: %d", len(docs_oxides))

            # Collect mp_ids already in dielectric set to avoid duplicates
            existing_ids = {r["mp_id"] for r in rows if r}

            for doc in tqdm(docs_oxides, desc="Parsing MP oxide entries"):
                if doc.material_id in existing_ids:
                    continue     # already have it with dielectric data -- skip
                row = self._parse_mp_entry(doc, has_dielectric=False)
                if row is not None:
                    rows.append(row)

        df = pd.DataFrame([r for r in rows if r is not None])
        log.info("Total MP entries parsed: %d", len(df))

        df.to_hdf(cache_path, key="data", mode="w", complevel=6, complib="blosc")
        log.info("MP dataset cached at %s", cache_path)
        return df

    def _parse_mp_entry(self, doc, has_dielectric: bool, summary_doc=None) -> Optional[dict]:
        """Parse a single Materials Project API document into unified schema.
        FIX5: Detects DFT functional from is_hubbard; adds functional_code and
              band_gap_pbe columns so model can learn per-functional offsets.
              (run_type is not a valid summary endpoint field; is_hubbard used instead.)

        Args:
            doc: The Materials Project API document to parse.
            has_dielectric: Whether the material has dielectric data.
            summary_doc: The summary document containing additional information.
        """
        try:
            # Use summary_doc for properties if available (dielctric entries), else use doc
            if summary_doc:
                formula   = summary_doc.formula_pretty
                mp_id     = summary_doc.material_id
                band_gap  = self._safe_float(getattr(summary_doc, "band_gap", None))
                Ef        = self._safe_float(getattr(summary_doc, "formation_energy_per_atom", None))
                e_hull    = self._safe_float(getattr(summary_doc, "energy_above_hull", None))
                structure = getattr(summary_doc, "structure", None)
            else:
                formula   = getattr(doc, "formula_pretty", None)
                mp_id     = getattr(doc, "material_id", None)
                band_gap  = self._safe_float(getattr(doc, "band_gap", None))
                Ef        = self._safe_float(getattr(doc, "formation_energy_per_atom", None))
                e_hull    = self._safe_float(getattr(doc, "energy_above_hull", None))
                structure = getattr(doc, "structure", None)

            # FIX5 / FIX-MP2 (revised): functional detection via is_hubbard.
            #
            # Neither run_type nor is_hubbard is exposed as a requestable field
            # on the current mp-api summary endpoint (both raise MPRestError when
            # added to the fields list).  The getattr fallback below therefore
            # always returns False, and all MP entries receive functional_code=PBE.
            #
            # Impact: GGA+U entries (transition-metal oxides: Fe, Co, Ni, Mn…)
            # are mislabeled as PBE.  For the high-k oxide targets in this
            # pipeline (HfO2, ZrO2, TiO2, La2O3, Al2O3) GGA+U is rarely used,
            # so the mislabeling affects a small minority of Tier 1 rows and
            # does NOT affect the dielectric (k_total) or band_gap targets --
            # functional_code is stored as metadata only and is not a model
            # input feature in the current architecture.
            #
            # Accurate functional detection can be recovered in v4 via a
            # targeted TaskDoc query after Tier 2/3 residual analysis confirms
            # it matters for prediction quality.
            _run_src   = summary_doc if summary_doc is not None else doc
            is_hubbard = bool(getattr(_run_src, "is_hubbard", False) or False)

            if is_hubbard:
                dft_functional  = "GGA+U"
                functional_code = FUNCTIONAL_CODE["GGA+U"]
            else:
                dft_functional  = "PBE"
                functional_code = FUNCTIONAL_CODE["PBE"]

            # band_gap_pbe: valid for PBE and GGA+U; NaN for r2SCAN (different scale)
            band_gap_pbe = band_gap if functional_code in (
                FUNCTIONAL_CODE["PBE"], FUNCTIONAL_CODE["GGA+U"]
            ) else np.nan

            # FIX-DIELECTRIC-SCALAR: e_total / e_ionic / e_electronic are
            # SCALAR floats in the current mp-api DielectricDoc, not tensors.
            #
            # History of the bug:
            #   FIX-MP1: correctly changed doc.dielectric.e_total → doc.e_total
            #   FIX-MP3: incorrectly removed "total","ionic","electronic" from
            #            MP_ELECTRIC_FIELDS as "stale" -- they are the 3×3 tensor
            #            fields; what remained were the scalar e_* fields
            #   Review-2: correctly used _compute_k_from_tensor for consistency
            #            with JARVIS -- BUT _compute_k_from_tensor expects a 2D
            #            tensor; receiving a scalar float gives ndim=0, the
            #            dimension check (ndim==2) fails, and the function
            #            returns None for every MP dielectric entry
            #
            # Correct fix: use _safe_float directly on the scalar fields.
            # e_total is already the isotropic trace-average dielectric constant
            # (what we want as k_total). No tensor arithmetic needed.
            k_total = None
            k_ionic = None
            k_elec  = None
            if has_dielectric:
                k_total = self._safe_float(getattr(doc, "e_total",      None))
                k_ionic = self._safe_float(getattr(doc, "e_ionic",      None))
                k_elec  = self._safe_float(getattr(doc, "e_electronic", None))

            # Convert pymatgen Structure -> JARVIS Atoms for graph construction
            atoms_dict    = None
            has_structure = False
            if structure is not None:
                try:
                    j_atoms       = JarvisAtomsAdaptor.get_atoms(structure)
                    atoms_dict    = json.dumps(j_atoms.to_dict())
                    has_structure = True
                except Exception:
                    pass

            row_hash = hashlib.md5(f"MP_{mp_id}_{formula}".encode()).hexdigest()[:12]

            return {
                "source":                    "MaterialsProject",
                "jid":                       None,
                "mp_id":                     mp_id,
                "formula":                   formula,
                "k_measured":                k_total,   # DFT total stored under k_measured for Tier 3 compat
                "k_total":                   k_total,   # explicit alias used by Tier 1/2 configs
                "k_ionic":                   k_ionic,   # ionic contribution (trace avg of e_ionic)
                "k_elec":                    k_elec,    # electronic contribution (trace avg of e_electronic)
                "band_gap":                  band_gap,
                "band_gap_mbj":              None,
                "formation_energy_per_atom": Ef,
                "e_above_hull":              e_hull,
                "bulk_modulus":              None,
                "shear_modulus":             None,
                "has_structure":             has_structure,
                "atoms_dict":                atoms_dict,
                # FIX5: functional labeling
                "dft_functional":            dft_functional,
                "functional_code":           functional_code,
                "band_gap_optb88vdw":        np.nan,
                "band_gap_pbe":              band_gap_pbe,
                "row_hash":                  row_hash,
                "tier":                      None,
                "date_added":                datetime.date.today().isoformat(),
            }

        except Exception as e:
            log.debug("Skipping MP entry %s: %s", getattr(doc, "material_id", "?"), e)
            return None

    # --------------------------------------------------------------------------
    # 1c. QM9 pull
    # --------------------------------------------------------------------------
    def pull_qm9(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Pull QM9 molecular dataset via JARVIS figshare module.

        Key: 'qm9_std_jctc' -- standardized version used in the ALIGNN paper.
        ~130,829 molecules. Properties include HOMO, LUMO, gap, dipole,
        polarisability (alpha) -- alpha correlates with dielectric response.

        QM9 is used in Tier 1 ONLY. It teaches the model about:
        - Molecular polarisability (direct proxy for k)
        - Electronic gap sensitivity to bonding geometry
        - Organic/inorganic property range calibration

        Note: QM9 is excluded from Tier 2 and Tier 3 (molecular ≠ crystal).
        """
        cache_path = self.cache_dir / "qm9_full.h5"

        if cache_path.exists() and not force_refresh:
            log.info("Loading cached QM9 dataset from %s", cache_path)
            return pd.read_hdf(cache_path, key="data")

        log.info("Downloading QM9 via JARVIS figshare (~130K molecules, ~200 MB)...")
        raw = jdata("qm9_std_jctc")
        log.info("QM9 raw entries: %d", len(raw))

        rows = []
        for entry in tqdm(raw, desc="Parsing QM9 entries"):
            row = self._parse_qm9_entry(entry)
            if row is not None:
                rows.append(row)

        df = pd.DataFrame(rows)
        log.info("QM9 parsed entries: %d", len(df))

        df.to_hdf(cache_path, key="data", mode="w", complevel=6, complib="blosc")
        log.info("QM9 cached at %s", cache_path)
        return df

    def _parse_qm9_entry(self, entry: dict) -> Optional[dict]:
        """
        Parse a QM9 entry. Maps QM9 properties to unified schema.

        Key QM9 -> unified schema mappings:
        - alpha (polarisability, Bohr^3) -> stored as 'k_measured' proxy
          Note: polarisability ≠ dielectric constant but both depend on
          electronic response. Alpha is retained for pretraining signal only;
          it is excluded from Tier 2/3 evaluation.
        - gap (HOMO-LUMO gap, eV)        -> 'band_gap'
        - mu (dipole moment, Debye)      -> auxiliary feature
        """
        try:
            qm9_id   = entry.get("id", "")
            atoms_dict = entry.get("atoms")
            if atoms_dict is None:
                return None

            alpha = self._safe_float(entry.get("alpha"))    # polarisability
            gap   = self._safe_float(entry.get("gap"))      # HOMO-LUMO gap
            homo  = self._safe_float(entry.get("HOMO"))
            lumo  = self._safe_float(entry.get("LUMO"))
            mu    = self._safe_float(entry.get("mu"))       # dipole moment
            U0    = self._safe_float(entry.get("U0"))       # internal energy at 0K

            j_atoms  = JAtoms.from_dict(atoms_dict)
            formula  = j_atoms.composition.reduced_formula
            row_hash = hashlib.md5(f"QM9_{qm9_id}_{formula}".encode()).hexdigest()[:12]

            return {
                "source":                    "QM9",
                "jid":                       f"qm9_{qm9_id}",
                "mp_id":                     None,
                "formula":                   formula,
                # alpha = molecular polarisability (Bohr^3) -- pretrain proxy only
                # DO NOT use as k_measured in Tier2/3 evaluation
                "k_measured":                alpha,
                # k_total is None for QM9: alpha (polarisability) is NOT equivalent
                # to the ionic+electronic dielectric constant of a crystal.
                # Keeping k_total=None prevents QM9 entries from polluting the
                # Tier 1/2 dielectric target distribution.
                "k_total":                   None,
                "k_ionic":                   None,
                "k_elec":                    None,
                "band_gap":                  gap,
                "band_gap_mbj":              None,
                "formation_energy_per_atom": U0,    # closest analog to Ef
                "e_above_hull":              None,
                "bulk_modulus":              None,
                "shear_modulus":             None,
                "has_structure":             True,
                "atoms_dict":                json.dumps(atoms_dict),
                # QM9-specific extras stored as JSON string
                "qm9_extras":                json.dumps({
                    "HOMO": homo, "LUMO": lumo, "mu": mu, "alpha": alpha
                }),
                "row_hash":                  row_hash,
                "tier":                      None,
                "date_added":                datetime.date.today().isoformat(),
                "is_molecule":               True,    # flag -- excluded from Tier2+
            }

        except Exception as e:
            log.debug("Skipping QM9 entry %s: %s", entry.get("id", "?"), e)
            return None

    # --------------------------------------------------------------------------
    # 1d. Experimental process database (Tier 3 only)
    # --------------------------------------------------------------------------
    def load_experimental_process_db(
        self, path: Path = DATA_ROOT / "processed" / "process_db_clean.csv"
    ) -> pd.DataFrame:
        """
        Load the hand-curated experimental process database from Week 1.
        This is the Tier 3 experimental contribution -- real ALD/anneal data
        paired with measured k, J_g, D_it.

        If the file does not exist yet (early in the project), returns empty
        DataFrame with correct schema so the pipeline does not break.
        """
        if not path.exists():
            log.warning(
                "Experimental process DB not found at %s. "
                "Tier 3 will train on DFT entries only until Week 1 data is ready.",
                path,
            )
            return pd.DataFrame()

        df = pd.read_csv(path)
        df["source"]       = "Experimental"
        df["is_molecule"]  = False
        df["tier"]         = 3

        # -- Target column renames ------------------------------------------------
        # FIX-T3-8 / FIX-T3-12: The rename dict must match the ACTUAL column
        # names in process_clean_db.csv. The CSV was updated with new names
        # (k_dielectric_constant, J_g_A_cm2_at_1v, etc.) but the rename dict
        # still referenced old names (dielectric_constant_k,
        # leakage_J_A_cm2_at_field, etc.) that no longer exist ->0/120 rows filed
        #
        # Rule: list EVERY known alias for each target, map to the internal name.
        # Order matters for pd.rename — last match wins if multiple aliases exist;
        # the dict is applied once so the first-found key takes effect.
        target_renames = {
            # ── k_measured ────────────────────────────────────────────────────
            "k_dielectric_constant":        "k_measured",   # seed CSV schema
            "dielectric_constant_k":        "k_measured",   # old schema (legacy)
            "dielectric_k":                 "k_measured",   # common shorthand
            "k_total":                      "k_measured",   # DFT alias
            "k_measured":                   "k_measured",   # already correct
            "epsilon_total":                "k_measured",   # MP dielectric field
            # ── band_gap ──────────────────────────────────────────────────────
            "band_gap_eV":                  "band_gap",     # common variant
            "band_gap":                     "band_gap",     # already correct
            "Eg_eV":                        "band_gap",     # abbreviation
            # ── J_g (leakage current density) ─────────────────────────────────
            "J_g_A_cm2_at_1V":             "J_g_A_cm2",   # BUG-1 FIX: uppercase V (actual CSV col)
            "J_g_A_cm2_at_1v":             "J_g_A_cm2",   # lowercase v fallback for resilience
            "leakage_J_A_cm2_at_field":    "J_g_A_cm2",   # old CSV schema
            "leakage_current_J_A_cm2":     "J_g_A_cm2",   # alternative
            "J_g":                          "J_g_A_cm2",   # abbreviation
            "J_g_A_cm2":                    "J_g_A_cm2",   # already correct
            # ── E_BD (breakdown field) ─────────────────────────────────────────
            "breakdown_field_MV_cm":        "E_BD_MV_cm",  # seed CSV schema
            "E_breakdown_MV_cm":            "E_BD_MV_cm",  # alternative
            "E_BD":                         "E_BD_MV_cm",  # abbreviation
            "E_BD_MV_cm":                   "E_BD_MV_cm",  # already correct
            # -- Structural / identity columns needed by downstream code -------------
            "material_system":              "material",    # _impute structures expects "material"
            "row_id":                       "paper_id",    # hash function expects "paper_id"
        }
        df = df.rename(columns={k: v for k, v in target_renames.items() if k in df.columns})

        # After rename, log coverage of each target so mismatches are immediately
        # visible without waiting for training results.
        for tgt in ["k_measured", "band_gap", "J_g_A_cm2", "E_BD_MV_cm"]:
            n = int(df[tgt].notna().sum()) if tgt in df.columns else 0
            pct = 100.0 * n / max(len(df), 1)
            log.info(
                "  FIX-T3-8 target %-20s  %d/%d rows filled  (%.1f%%)",
                tgt + ":", n, len(df), pct,
            )

        # FIX-T3-11: Clean all numeric target and process columns.
        # CSV values from literature often contain ~prefix, ranges, unit suffixes
        # that pd.to_numeric(errors='coerce') silently converts to NaN.
        # _clean_numeric_series() recovers these values before they enter the HDF5.
        _numeric_target_cols = ["k_measured", "band_gap", "J_g_A_cm2", "E_BD_MV_cm"]
        _numeric_proc_cols   = [
            "substrate_temp_C", "anneal_temp_C", "anneal_duration_min",
            "growth_rate_A_per_cycle", "n_cycles", "film_thickness_A",
            "pressure_mTorr",
        ]
        log.info("FIX-T3-11: cleaning numeric columns (strip ~/range/suffix)...")
        for _col in _numeric_target_cols + _numeric_proc_cols:
            if _col in df.columns:
                df[_col] = _clean_numeric_series(df[_col], col_name=_col)

        # Re-log coverage after cleaning to show actual improvement
        log.info("FIX-T3-11 post-clean target coverage:")
        for tgt in ["k_measured", "band_gap", "J_g_A_cm2", "E_BD_MV_cm"]:
            n = int(df[tgt].notna().sum()) if tgt in df.columns else 0
            pct = 100.0 * n / max(len(df), 1)
            log.info(
                "  %-20s  %d/%d rows filled  (%.1f%%)",
                tgt + ":", n, len(df), pct,
            )

        # -- FIX-T3-1: Process parameter column renames --------------------------
        # process_db.csv uses human-readable column names that differ from the
        # PROCESS_PARAMS_FEATURES keys that drive ProcessParamsEncoder.  Without
        # this rename block _safe_val() returns None for every process column in
        # every experimental row → avail_flag stays 0.0 → process encoder never
        # activates even after structure imputation is applied.
        #
        # Keys in PROCESS_PARAMS_FEATURES["numerical"]:
        #   substrate_temp_C, anneal_temp_C, anneal_duration_min,
        #   growth_rate_A_per_cycle, n_cycles, film_thickness_A, pressure_mTorr
        # Keys in PROCESS_PARAMS_FEATURES["categorical"]:
        #   anneal_ambient, precursor_type, oxidant_type
        proc_renames = {
            # ── Substrate / deposition temperature ────────────────────────────
            # BUG-2 FIX: "ald_substrate_temp_c" had lowercase c → never fired.
            # CSV column is "ald_substrate_temp_C" (uppercase C).
            # substrate_temp_C was 0/120 rows — ProcessParamsEncoder blind to T_sub.
            "ald_substrate_temp_C":         "substrate_temp_C",  # BUG-2 FIX: uppercase C
            "ald_substrate_temp_c":         "substrate_temp_C",  # lowercase fallback
            "deposition_temp_C":            "substrate_temp_C",  # old seed CSV compat
            # ── Anneal temperature ────────────────────────────────────────────
            "anneal_temp_PDA_C":            "anneal_temp_C",
            "post_anneal_temp_C":           "anneal_temp_C",
            # ── Anneal duration (unit-aware string parse applied below) ───────
            "anneal_duration":              "anneal_duration_min",
            "anneal_duration_min":          "anneal_duration_min",
            # ── Cycle count ───────────────────────────────────────────────────
            "ald_cycle_count":              "n_cycles",
            "num_cycles_or_thickness":      "n_cycles",
            # ── Chamber pressure ─────────────────────────────────────────────
            # BUG-3 FIX: "chamber_base_pressure_mTorr" never matched CSV col
            # "chamber_base_pressure_Torr". Also needs ×1000 Torr→mTorr below.
            "chamber_base_pressure_Torr":   "pressure_mTorr",    # BUG-3 FIX: Torr suffix
            "chamber_base_pressure_mTorr":  "pressure_mTorr",    # old name fallback
            "chamber_pressure":             "pressure_mTorr",    # seed CSV compat
            # ── Anneal atmosphere ─────────────────────────────────────────────
            "anneal_atmosphere":            "anneal_ambient",
            "post_anneal_atm":              "anneal_ambient",
            # ── Precursor / oxidant ───────────────────────────────────────────
            "precursor_type_Hf":            "precursor_type",
            "precursor_metal":              "precursor_type",
            "precursor_oxidant":            "oxidant_type",
            "oxidant_type":                 "oxidant_type",
            # Unit-converting renames handled separately below (nm → Å)
        }
        df = df.rename(columns={k: v for k, v in proc_renames.items() if k in df.columns})

        # Unit conversions: process_db.csv stores GPC in nm/cycle and thickness
        # in nm; PROCESS_PARAMS_FEATURES expects Å (×10).  Apply before rename
        # so the encoder receives values on the same scale as the vocab/stats.
        if "gpc_nm_per_cycle" in df.columns:
            df["growth_rate_A_per_cycle"] = (
                pd.to_numeric(df["gpc_nm_per_cycle"], errors="coerce") * 10.0
            )
            df = df.drop(columns=["gpc_nm_per_cycle"])
            log.debug("FIX-T3-1: gpc_nm_per_cycle → growth_rate_A_per_cycle (×10)")

        if "film_thickness_nm" in df.columns:
            df["film_thickness_A"] = (
                pd.to_numeric(df["film_thickness_nm"], errors="coerce") * 10.0
            )
            df = df.drop(columns=["film_thickness_nm"])
            log.debug("FIX-T3-1: film_thickness_nm → film_thickness_A (×10)")

        # BUG-3 FIX: Torr → mTorr unit conversion.
        # After proc_renames, "pressure_mTorr" still holds values in Torr
        # (renamed from chamber_base_pressure_Torr which stores Torr floats).
        # ProcessParamsEncoder log1p-normalises pressure_mTorr; receiving Torr
        # values (1e-7 to 0.2) instead of mTorr (1e-4 to 200) compresses the
        # log1p range to near-zero, destroying the pressure signal entirely.
        if "pressure_mTorr" in df.columns:
            df["pressure_mTorr"] = (
                pd.to_numeric(df["pressure_mTorr"], errors="coerce") * 1000.0
            )
            n_p = int(df["pressure_mTorr"].notna().sum())
            log.debug(
                "BUG-3 FIX: pressure_mTorr ← chamber_base_pressure_Torr ×1000  "
                "(%d rows,  range [%.2e, %.2e] mTorr)",
                n_p,
                float(df["pressure_mTorr"].min()) if n_p > 0 else float("nan"),
                float(df["pressure_mTorr"].max()) if n_p > 0 else float("nan"),
            )

        # BUG-4 FIX: anneal_duration unit-aware parsing.
        # CSV stores values like "30s", "60 s", "20 min", "1h".
        # pd.to_numeric() and _clean_numeric_series() both return NaN on these.
        # anneal_duration_min was NaN for all 68 rows it maps via proc_renames.
        # Parse to float minutes so log1p normalisation works correctly.
        if "anneal_duration_min" in df.columns:
            import re as _re_dur

            def _parse_duration_to_min(raw) -> float:
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    return float("nan")
                s = str(raw).strip().lower()
                if not s or s in ("nan", "none", "n/a", "-"):
                    return float("nan")
                # seconds: "30s", "60 s", "1800 sec", "30seconds"
                m = _re_dur.match(r"^(\d+\.?\d*)\s*s(?:ec(?:onds?)?)?$", s)
                if m:
                    return float(m.group(1)) / 60.0
                # minutes: "20min", "5 min", "10m", "10minutes"
                m = _re_dur.match(r"^(\d+\.?\d*)\s*m(?:in(?:utes?)?)?$", s)
                if m:
                    return float(m.group(1))
                # hours: "1h", "2 hr", "0.5 hours"
                m = _re_dur.match(r"^(\d+\.?\d*)\s*h(?:r|ours?)?$", s)
                if m:
                    return float(m.group(1)) * 60.0
                # bare number → assume minutes
                try:
                    return float(s)
                except ValueError:
                    log.debug(
                        "BUG-4: anneal_duration_min unparseable value '%s' → NaN", raw
                    )
                    return float("nan")

            n_before = int(df["anneal_duration_min"].notna().sum())
            df["anneal_duration_min"] = df["anneal_duration_min"].apply(
                _parse_duration_to_min
            )
            n_after = int(df["anneal_duration_min"].notna().sum())
            log.debug(
                "BUG-4 FIX: anneal_duration_min unit-parse: "
                "%d non-null strings → %d valid floats (minutes)",
                n_before, n_after,
            )

        # FIX-PROC-COERCE: apply pd.to_numeric(errors='coerce') to all numerical
        # proc columns after proc_renames and unit conversions.
        # Some CSV cells contain non-numeric strings that pass proc_renames
        # undetected: 'RT' (substrate_temp_C), '90+10'/'varies' (n_cycles),
        # 'atm'/'10Pa' (pressure_mTorr), '30s'/'60s' already handled by
        # BUG-4 duration parser but residual strings may still exist.
        # Coerce them to NaN here so _extract_proc_feats and fit_stats
        # never receive string values in numeric columns.
        _proc_num_final = [
            "substrate_temp_C", "anneal_temp_C", "anneal_duration_min",
            "growth_rate_A_per_cycle", "n_cycles", "film_thickness_A",
            "pressure_mTorr",
        ]
        for _col in _proc_num_final:
            if _col in df.columns:
                _before = int(df[_col].notna().sum())
                df[_col] = pd.to_numeric(df[_col], errors="coerce")
                _after  = int(df[_col].notna().sum())
                if _before != _after:
                    log.info(
                        "FIX-PROC-COERCE: %s coerced %d non-numeric strings → NaN"
                        "  (%d → %d valid rows)",
                        _col, _before - _after, _before, _after,
                    )
        proc_num_cols = ["substrate_temp_C", "anneal_temp_C", "anneal_duration_min",
                         "growth_rate_A_per_cycle", "n_cycles",
                         "film_thickness_A", "pressure_mTorr"]
        proc_cat_cols = ["anneal_ambient", "precursor_type", "oxidant_type"]
        for col in proc_num_cols + proc_cat_cols:
            n_filled = int(df[col].notna().sum()) if col in df.columns else 0
            pct      = 100.0 * n_filled / max(len(df), 1)
            log.info(
                "  FIX-T3-1 proc col %-28s  %d/%d rows filled  (%.1f%%)",
                col + ":", n_filled, len(df), pct,
            )

        # For experimental entries the measured value IS the total dielectric constant.
        # Populate k_total so Tier 2/3 configs that reference "k_total" work correctly.
        if "k_measured" in df.columns:
            df["k_total"] = df["k_measured"]
        else:
            df["k_total"] = np.nan

        # Row hash for dedup — FIX-T3-5: previous 3-field key (doi+material+temp)
        # caused silent hash collisions: ~62 of 120 rows lost when doi is absent
        # (NaN → '?') and multiple rows share the same material+temperature.
        # E.g. all HfO2 rows at 250°C with no DOI → hash "EXP_?_HfO2_250.0"
        # → only 1 survives drop_duplicates.
        #
        # Fix: 7-field key that is unique across all realistic ALD combinations.
        #   paper_id      — always set in our CSV schema (P001, P002 ...)
        #   material      — HfO2, Al2O3, ZrO2 etc.
        #   substrate_temp_C — ALD chuck temperature
        #   precursor_type   — distinguishes TEMA-Hf vs TDMAHf at same temp
        #   oxidant_type     — H2O vs O3 vs O2_plasma
        #   n_cycles         — distinguishes thickness series at same T
        #   row_index        — last-resort tie-breaker: guarantees uniqueness
        #                      even when all named fields happen to match
        def _safe_hash_val(row, *keys):
            for k in keys:
                v = row.get(k, None)
                if v is not None and str(v) not in ("nan", "None", ""):
                    return str(v)
            return "?"

        df["row_hash"] = [
            hashlib.md5(
                (
                    f"EXP"
                    f"_{_safe_hash_val(row, 'paper_id', 'doi')}"
                    f"_{_safe_hash_val(row, 'material')}"
                    f"_{_safe_hash_val(row, 'substrate_temp_C')}"
                    f"_{_safe_hash_val(row, 'precursor_type')}"
                    f"_{_safe_hash_val(row, 'oxidant_type')}"
                    f"_{_safe_hash_val(row, 'n_cycles')}"
                    f"_{idx}"                           # row index — absolute tie-breaker
                ).encode()
            ).hexdigest()[:12]
            for idx, row in enumerate(df.to_dict("records"))
        ]

        log.info("Experimental process DB loaded: %d rows", len(df))
        return df

    # --------------------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _compute_k_from_tensor(
        eps_ionic: Any, eps_elec: Any
    ) -> Optional[float]:
        """
        Compute scalar isotropic dielectric constant from ionic + electronic
        contributions, each of which may be a 3x3 tensor or scalar.

        k_total = trace_average(epsilon_ionic) + trace_average(epsilon_elec)

        For anisotropic crystals (e.g. monoclinic HfO2) the trace average
        gives the orientationally averaged k measured in C-V experiments.
        """
        def _tensor_trace_avg(x):
            if x is None or x == "na" or x == "":
                return None
            try:
                if isinstance(x, (list, tuple)):
                    arr = np.array(x, dtype=float)
                    if arr.ndim == 2 and arr.shape == (3, 3):
                        return float(np.mean(np.diag(arr)))
                    elif arr.ndim == 1 and len(arr) == 3:
                        return float(np.mean(arr))
                    elif arr.ndim == 1 and len(arr) == 1:
                        return float(arr[0])
                return float(x)
            except Exception:
                return None

        ionic = _tensor_trace_avg(eps_ionic)
        elec  = _tensor_trace_avg(eps_elec)

        if ionic is not None and elec is not None:
            return ionic + elec
        elif ionic is not None:
            return ionic         # ionic only -- underestimate but usable
        elif elec is not None:
            return elec          # electronic only -- underestimate but usable
        return None

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        """
        Convert to float, returning np.nan on failure.

        Returns np.nan (not Python None) so that pandas columns built from
        lists of _safe_float results stay dtype float64 instead of dtype
        object.  dtype('O') columns cause get_stratified_split to fail with
        "Cannot cast array data from dtype('O') to dtype('float64') according
        to rule 'safe'" because np.ndarray.astype(float) rejects object arrays
        containing None under NumPy >= 1.20 safe-casting rules.
        """
        if x is None or x == "na" or x == "":
            return float("nan")
        try:
            v = float(x)
            return float("nan") if (np.isnan(v) or np.isinf(v)) else v
        except Exception:
            return float("nan")


# ==============================================================================
# SECTION 2 -- THREE-TIER DATASET BUILDER
# ==============================================================================

class TierDatasetBuilder:
    """
    Assembles the three-tier dataset from extracted raw data.

    Tier assignment logic:
    ---------------------
    Tier 1: all entries (JARVIS + MP + QM9)  -- general oxide + molecular physics
    Tier 2: subset -- oxide dielectrics with k > 10, Eg > 1 eV, has_structure=True
            contains at least one Tier 2 cation
    Tier 3: subset -- HfO2 family specifically + experimental entries
    """

    TIER_PATHS = {
        1: DATA_ROOT / "tier1_foundation.h5",
        2: DATA_ROOT / "tier2_domain.h5",
        3: DATA_ROOT / "tier3_project.h5",
    }

    MANIFEST_PATH = DATA_ROOT / "dataset_manifest.json"

    def __init__(self):
        self._load_or_init_manifest()

    def _load_or_init_manifest(self):
        """Load manifest or create empty version."""
        if self.MANIFEST_PATH.exists():
            with open(self.MANIFEST_PATH) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "schema_version": "1.2",
                "created":        datetime.date.today().isoformat(),
                "last_updated":   datetime.date.today().isoformat(),
                "tiers": {str(i): {"row_count": 0, "last_updated": None}
                          for i in [1, 2, 3]},
                "growth_log": [],
            }

    def _save_manifest(self):
        self.manifest["last_updated"] = datetime.date.today().isoformat()
        with open(self.MANIFEST_PATH, "w") as f:
            json.dump(self.manifest, f, indent=2)

    # --------------------------------------------------------------------------
    # FIX4: Cross-source structural deduplication helper
    # --------------------------------------------------------------------------
    @staticmethod
    def _atoms_json_to_pymatgen(atoms_json: Optional[str]):
        """Convert stored atoms_dict JSON to pymatgen Structure."""
        if not atoms_json:
            return None
        try:
            j_atoms = JAtoms.from_dict(json.loads(atoms_json))
            return JarvisAtomsAdaptor.get_structure(j_atoms)
        except Exception:
            return None

    @staticmethod
    def _backfill_k_total(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Ensure k_total is populated from k_measured wherever k_total is absent or NaN.

        Handles two scenarios:
          a) Stale raw_cache (jarvis_dft.h5) built before 'k_total' was explicitly
             added to _parse_jarvis_entry: column is absent entirely.
          b) Column present but all-NaN (old v2.0 schema, or JARVIS entries loaded
             from a pre-FIX-OBS2 cache).

        Called on df_jarvis (and optionally df_mp_clean) BEFORE pd.concat in
        build_tier1 so that JARVIS k_total is correctly populated prior to
        cross-source dedup and the physical validity filter.
        """
        if "k_measured" not in df.columns:
            return df                          # nothing to backfill from

        if "k_total" not in df.columns:
            df = df.copy()
            df["k_total"] = df["k_measured"]
            n = int(df["k_total"].notna().sum())
            log.info(
                "  k_total backfill [%s]: column absent → created from k_measured"
                "  (%d non-null)", name, n
            )
            return df

        # Column exists but may be partially or wholly NaN (stale cache)
        mask = df["k_total"].isna() & df["k_measured"].notna()
        n    = int(mask.sum())
        if n > 0:
            df = df.copy()
            df.loc[mask, "k_total"] = df.loc[mask, "k_measured"]
            log.info(
                "  k_total backfill [%s]: %d NaN k_total filled from k_measured",
                name, n
            )
        return df

    def deduplicate_cross_source(
        self,
        df_jarvis: pd.DataFrame,
        df_mp:     pd.DataFrame,
    ) -> tuple:
        """
        FIX4: Remove MP entries structurally identical to JARVIS entries.

        The row_hash uses source-prefixed keys (JARVIS_jid vs MP_mpid), so
        same physical crystal from both databases survives row_hash dedup.
        This causes the model to see the same crystal twice with contradictory
        band-gap targets (OptB88vdW vs PBE systematic offset).

        Strategy: for each MP entry whose formula exists in JARVIS, run
        pymatgen StructureMatcher. Drop the MP entry on match -- keep JARVIS
        (OptB88vdW is more consistent for dielectric constant calculations).

        DEDUP-MERGE: when the kept JARVIS entry has k_total=NaN but the
        dropped MP entry has a valid k_total (from the dielectric endpoint),
        copy k_total/k_ionic/k_elec/k_measured from MP to JARVIS so that
        the dielectric information is not permanently discarded.

        Returns (df_jarvis_updated, df_mp_unique).  Callers must capture
        both return values.

        Runtime: ~5–20 min for full datasets. Skip with --skip_cross_dedup.
        """
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher         = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        shared_formulas = set(df_mp["formula"]).intersection(set(df_jarvis["formula"]))
        jv_by_formula   = df_jarvis.groupby("formula")
        drop_idx        = set()
        n_merged        = 0

        # Work on a copy so we can safely write k_total merges into JARVIS rows
        df_jarvis_mut = df_jarvis.copy()

        log.info("FIX4 cross-source dedup: %d shared formulas between JARVIS and MP",
                 len(shared_formulas))

        mp_shared = df_mp[df_mp["formula"].isin(shared_formulas)]
        for idx, mp_row in tqdm(mp_shared.iterrows(), total=len(mp_shared),
                                desc="FIX4 cross-source dedup"):
            mp_struct = self._atoms_json_to_pymatgen(mp_row.get("atoms_dict"))
            if mp_struct is None:
                continue
            for _, jv_row in jv_by_formula.get_group(mp_row["formula"]).iterrows():
                jv_struct = self._atoms_json_to_pymatgen(jv_row.get("atoms_dict"))
                if jv_struct is None:
                    continue
                try:
                    if matcher.fit(mp_struct, jv_struct):
                        drop_idx.add(idx)

                        # DEDUP-MERGE: copy k_total from MP → JARVIS when JARVIS lacks it.
                        # The MP entry (from the dielectric endpoint) has k_total populated
                        # while the matching JARVIS entry may have k_total=NaN (JARVIS DFT
                        # subset without DFPT dielectric calculation).  Without this merge
                        # the dielectric data is silently discarded even though we keep
                        # the JARVIS structural entry.
                        jv_idx = jv_row.name          # pandas integer index
                        jv_k   = df_jarvis_mut.at[jv_idx, "k_total"] \
                                 if "k_total" in df_jarvis_mut.columns else float("nan")
                        mp_k   = mp_row.get("k_total")
                        if (pd.isna(jv_k) and mp_k is not None
                                and not (isinstance(mp_k, float) and np.isnan(mp_k))):
                            df_jarvis_mut.at[jv_idx, "k_total"]    = mp_k
                            df_jarvis_mut.at[jv_idx, "k_measured"] = mp_k
                            if "k_ionic" in df_jarvis_mut.columns:
                                df_jarvis_mut.at[jv_idx, "k_ionic"] = mp_row.get("k_ionic")
                            if "k_elec" in df_jarvis_mut.columns:
                                df_jarvis_mut.at[jv_idx, "k_elec"]  = mp_row.get("k_elec")
                            n_merged += 1
                        break
                except Exception:
                    continue

        df_mp_unique = df_mp.drop(index=list(drop_idx)).reset_index(drop=True)
        log.info(
            "FIX4 dedup: dropped %d MP duplicates  →  %d MP entries remain"
            "  |  k_total merged to %d JARVIS entries (DEDUP-MERGE)",
            len(drop_idx), len(df_mp_unique), n_merged,
        )
        return df_jarvis_mut, df_mp_unique

    # --------------------------------------------------------------------------
    # 2a. Build Tier 1 -- Foundation (~55K JARVIS + ~70K MP + ~130K QM9)
    # --------------------------------------------------------------------------
    def build_tier1(
        self,
        df_jarvis:        pd.DataFrame,
        df_mp:            pd.DataFrame,
        df_qm9:           pd.DataFrame,
        force_rebuild:    bool = False,
        skip_cross_dedup: bool = False,
    ) -> pd.DataFrame:
        """
        Assemble full Tier 1 foundation dataset.
        FIX4: Cross-source structural deduplication (MP vs JARVIS).
        FIX5: Ensures functional columns present on all sources before concat.
        """
        if self.TIER_PATHS[1].exists() and not force_rebuild:
            log.info("Loading existing Tier 1 from %s", self.TIER_PATHS[1])
            return pd.read_hdf(self.TIER_PATHS[1], key="data")

        log.info("Building Tier 1 foundation dataset...")

        # -- Bug fix: backfill k_total from k_measured BEFORE cross-source dedup ----
        # Root cause of JARVIS k_total=0: df_jarvis loaded from a stale raw_cache
        # (jarvis_dft.h5 built before k_total was explicitly in _parse_jarvis_entry).
        # k_total column exists but is all-NaN; k_measured has ~5,200 valid values.
        # After pd.concat, JARVIS rows contribute k_total=NaN to df_all throughout.
        # _ensure_k_total is a no-op because the column already exists.
        # Fix: backfill before concat and before dedup, so that DEDUP-MERGE below
        # correctly detects which JARVIS entries already have k_total.
        df_jarvis = self._backfill_k_total(df_jarvis, "JARVIS")
        df_mp     = self._backfill_k_total(df_mp,     "MP")

        # FIX4: cross-source structural deduplication
        if skip_cross_dedup:
            log.warning("--skip_cross_dedup: same crystal may appear twice with "
                        "contradictory DFT targets (PBE vs OptB88vdW).")
            df_mp_clean = df_mp.copy()
        else:
            if len(df_mp) > 0 and len(df_jarvis) > 0:
                # Returns (df_jarvis_updated, df_mp_unique).
                # df_jarvis_updated has k_total merged from MP for matched entries
                # where JARVIS had no dielectric data (DEDUP-MERGE fix).
                df_jarvis, df_mp_clean = self.deduplicate_cross_source(df_jarvis, df_mp)
            else:
                df_mp_clean = df_mp.copy()

        # FIX5: ensure functional columns exist on all sources
        for df in [df_jarvis, df_mp_clean, df_qm9]:
            if len(df) == 0:
                continue
            if "dft_functional" not in df.columns:
                df["dft_functional"] = "PBE"
            if "functional_code" not in df.columns:
                df["functional_code"] = FUNCTIONAL_CODE["PBE"]
            if "band_gap_optb88vdw" not in df.columns:
                df["band_gap_optb88vdw"] = np.nan
            if "band_gap_pbe" not in df.columns:
                df["band_gap_pbe"] = np.nan

        # Tag tiers
        for df, t in [(df_jarvis, 1), (df_mp_clean, 1), (df_qm9, 1)]:
            if len(df):
                df["tier"] = t

        # Concatenate all three sources
        dfs = [df for df in [df_jarvis, df_mp_clean, df_qm9] if len(df) > 0]
        df_all = pd.concat(dfs, ignore_index=True, sort=False)
        log.info("Before dedup: %d rows", len(df_all))
        # k_total diagnostic: count by source immediately after concat
        for src in ["JARVIS-DFT", "MaterialsProject", "QM9"]:
            mask = df_all["source"] == src
            n_src  = int(mask.sum())
            n_k    = int(df_all.loc[mask, "k_total"].notna().sum())
            log.info("  k_total audit post-concat | %-22s %6d rows  k_total=%d",
                     src, n_src, n_k)

        # Deduplicate by row_hash
        df_all = df_all.drop_duplicates(subset=["row_hash"], keep="first")
        log.info("After dedup: %d rows", len(df_all))

        # Physical validity filters for crystalline entries
        # (QM9 is flagged as is_molecule=True and kept regardless)
        is_molecule = df_all.get("is_molecule", pd.Series(False, index=df_all.index))
        # Handle NaN values by treating them as False (not a molecule = crystal)
        is_molecule = is_molecule.fillna(False).astype(bool)
        is_crystal = ~is_molecule

        # For crystals: exclude obviously bad entries
        bad_k = (df_all["k_measured"] < 1) | (df_all["k_measured"] > 500)
        bad_ef = df_all["formation_energy_per_atom"].abs() > 20
        bad_gap = df_all["band_gap"] < 0

        exclude = is_crystal & (bad_k.fillna(False) |
                                bad_ef.fillna(False) |
                                bad_gap.fillna(False))

        df_tier1 = df_all[~exclude].copy()
        df_tier1["tier"] = 1
        log.info("Tier 1 final count: %d rows", len(df_tier1))
        # k_total final audit
        kt_t1 = df_tier1["k_total"].notna().sum()
        log.info("  k_total audit Tier 1 final | total=%d  JARVIS=%d  MP=%d  QM9=%d",
                 kt_t1,
                 int(df_tier1.loc[df_tier1["source"]=="JARVIS-DFT","k_total"].notna().sum()),
                 int(df_tier1.loc[df_tier1["source"].str.startswith("Material"),"k_total"].notna().sum()),
                 int(df_tier1.loc[df_tier1["source"]=="QM9","k_total"].notna().sum()))

        # Save
        df_tier1.to_hdf(self.TIER_PATHS[1], key="data", mode="w",
                        complevel=6, complib="blosc")

        # Update manifest
        self.manifest["tiers"]["1"]["row_count"]    = len(df_tier1)
        self.manifest["tiers"]["1"]["last_updated"] = datetime.date.today().isoformat()
        self.manifest["tiers"]["1"]["breakdown"] = {
            "JARVIS-DFT":         int((df_tier1["source"] == "JARVIS-DFT").sum()),
            "MaterialsProject":   int((df_tier1["source"] == "MaterialsProject").sum()),
            "QM9":                int((df_tier1["source"] == "QM9").sum()),
            "mp_dupes_removed_fix4": int(len(df_mp) - len(df_mp_clean)) if not skip_cross_dedup else 0,
        }
        self._save_manifest()
        self._log_stats("Tier 1", df_tier1)
        return df_tier1

    # --------------------------------------------------------------------------
    # 2b. Build Tier 2 -- Domain (~8,000–15,000 entries)
    # --------------------------------------------------------------------------
    def build_tier2(
        self,
        df_tier1: pd.DataFrame,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """
        Derive Tier 2 from Tier 1 by applying domain-specific filters:
        - Not a molecule (QM9 excluded)
        - Contains at least one Tier 2 high-k cation
        - k_measured > 10 (if available) OR is a relevant oxide without k
        - band_gap > 1.0 eV (exclude metals)
        - has_structure = True (need crystal graph for ALIGNN)
        """
        if self.TIER_PATHS[2].exists() and not force_rebuild:
            log.info("Loading existing Tier 2 from %s", self.TIER_PATHS[2])
            return pd.read_hdf(self.TIER_PATHS[2], key="data")

        log.info("Deriving Tier 2 domain dataset from Tier 1...")
        # k_total diagnostic: how much k_total does df_tier1 carry in?
        log.info("  k_total audit Tier 2 INPUT  | total=%d  JARVIS=%d  MP=%d",
                 int(df_tier1["k_total"].notna().sum()),
                 int(df_tier1.loc[df_tier1["source"]=="JARVIS-DFT","k_total"].notna().sum()),
                 int(df_tier1.loc[df_tier1["source"].str.startswith("Material"),"k_total"].notna().sum()))

        # Exclude QM9 molecules
        is_molecule_t2 = df_tier1.get("is_molecule", pd.Series(False, index=df_tier1.index))
        is_molecule_t2 = is_molecule_t2.fillna(False).astype(bool)
        df_cryst = df_tier1[~is_molecule_t2].copy()

        # Must have crystal structure for ALIGNN graph construction
        df_struct = df_cryst[df_cryst["has_structure"] == True].copy()

        # CATION FILTER REMOVED:
        # The original TIER2_CATIONS filter excluded 48% of valid k_total entries
        # (3,464 of 7,239) including Fe, Mn, Co, Ni, Zn, Pb, Bi oxides that have
        # genuine dielectric data.  For ML training the crystal-graph → k_total
        # signal is valid regardless of which cation is present; the model learns
        # structure→property, not element identity.  The bandgap and k_total
        # filters below are sufficient physical selectors.
        # (TIER2_CATIONS is retained as a reference constant but not applied here.)
        df_cation = df_struct.copy()
        log.info("  k_total audit post-structure filter | rows=%d  k_total=%d",
                 len(df_cation), int(df_cation["k_total"].notna().sum()))

        # Band gap > 1 eV (exclude metals and semimetals)
        # Allow NaN (some entries don't have gap computed -- keep them)
        df_cation = df_cation[
            df_cation["band_gap"].isna() | (df_cation["band_gap"] > 1.0)
        ].copy()
        log.info("  k_total audit post-bandgap filter   | rows=%d  k_total=%d",
                 len(df_cation), int(df_cation["k_total"].notna().sum()))

        # If k_total is present, must be > TIER2_K_MIN (above SiO2 reference value).
        # Changed from >10 to >TIER2_K_MIN (3.9):
        #   - k>10 excluded 1,073 entries with 3.9 < k <= 10 (Al2O3 k~9.1,
        #     MgO k~9.8, etc.) that define the lower boundary of the dielectric
        #     spectrum the model needs to learn.
        #   - k>3.9 is the physically motivated threshold: any material above SiO2
        #     is a potential gate dielectric improvement.
        #   - More training data across the full k spectrum improves model
        #     generalisation to novel high-k candidates.
        has_k   = df_cation["k_total"].notna()
        valid_k = df_cation["k_total"] > TIER2_K_MIN
        df_tier2 = df_cation[~has_k | valid_k].copy()
        log.info(
            "  k_total audit post-k>%.1f filter   | rows=%d  k_total=%d  "
            "(★ Tier 2 training dataset size)",
            TIER2_K_MIN, len(df_tier2), int(df_tier2["k_total"].notna().sum()),
        )
        df_tier2["tier"] = 2
        log.info("Tier 2 final count: %d rows", len(df_tier2))

        df_tier2.to_hdf(self.TIER_PATHS[2], key="data", mode="w",
                        complevel=6, complib="blosc")

        self.manifest["tiers"]["2"]["row_count"]    = len(df_tier2)
        self.manifest["tiers"]["2"]["last_updated"] = datetime.date.today().isoformat()
        self._save_manifest()
        self._log_stats("Tier 2", df_tier2)
        return df_tier2

    # --------------------------------------------------------------------------
    # 2c. Build Tier 3 -- Project (~1,580 entries)
    # --------------------------------------------------------------------------
    def build_tier3(
        self,
        df_tier2:    pd.DataFrame,
        df_exp:      pd.DataFrame,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """
        Derive Tier 3 from Tier 2 (HfO2-family DFT entries) + experimental DB.

        HfO2-family filter: contains Hf OR (contains Zr AND Hf) i.e. HZO family.
        Experimental entries: all rows from process_db_clean.csv -- they are
        project-specific by construction.
        """
        if self.TIER_PATHS[3].exists() and not force_rebuild:
            log.info("Loading existing Tier 3 from %s", self.TIER_PATHS[3])
            return pd.read_hdf(self.TIER_PATHS[3], key="data")

        log.info("Deriving Tier 3 project dataset...")

        # HfO2-family filter
        # FIX-DONOR-POOL: expand donor pool from Hf-only to all high-k oxide
        # families present in the experimental CSV. The original filter only
        # included Hf-containing formulas, so Al2O3, ZrO2, TiO2, Ta2O5,
        # SrTiO3, La2O3, Y2O3 DFT rows were absent from donor_pool.
        # _find_donor alias table mapped these CSV materials to HfO2 as proxy
        # but that wastes the actual matched DFT structure. Including them in
        # donor_pool gives _find_donor exact-match donors instead.
        HIGH_K_ELEMENTS = {"Hf", "Zr", "Al", "Ti", "Ta", "Sr", "La", "Y",
                           "Ba", "Nb", "Ga", "In", "Sc", "Ce", "Pr", "Nd"}

        def is_hfo2_family(formula):
            if not isinstance(formula, str):
                return False
            has_o = "O" in formula
            has_highk = any(el in formula for el in HIGH_K_ELEMENTS)
            return has_o and has_highk

        def is_hfzr_core(formula):
            """Strict Hf/Zr-only filter for Tier 3 DFT TRAINING rows.
            FIX-DONOR-POOL: separate the donor pool (broad, for _find_donor)
            from the DFT training rows (Hf/Zr-only, for df_structural).
            Using the broad filter for df_hf caused 1128-row df_structural
            (Al2O3, TiO2, SrTiO3, etc. DFT rows included) which diluted the
            experimental proc-param signal from 79% → ~10% of each batch.
            df_hf (training) stays Hf/Zr-family only (~28 rows).
            _impute_structures uses the broad donor_pool separately.
            """
            if not isinstance(formula, str):
                return False
            return ("Hf" in formula or "Zr" in formula) and "O" in formula

        # df_hf: Hf/Zr-family DFT rows only — used for Tier 3 TRAINING
        df_hf = df_tier2[df_tier2["formula"].apply(is_hfzr_core)].copy()
        # df_donor_pool_wide: all high-k families — used ONLY for _impute_structures
        # (passed into _impute_structures as the donor source, not into df_tier3 itself)
        df_donor_pool_wide = df_tier2[df_tier2["formula"].apply(is_hfo2_family)].copy()

        # FIX-T3-8B: Propagate k_total → k_measured for DFT rows that have DFPT
        # dielectric data but no explicit k_measured column.
        # HfO2-family entries from JARVIS with DFPT calculations have k_total set
        # by _parse_jarvis_entry.  Making them visible as k_measured adds ~125-166
        # valid training rows to the Tier 3 k_measured head without requiring any
        # experimental measurement.
        if "k_total" in df_hf.columns:
            has_k_total    = df_hf["k_total"].notna()
            lacks_k_meas   = ~df_hf.get("k_measured", pd.Series(False, index=df_hf.index)).notna() if "k_measured" in df_hf.columns else pd.Series(True, index=df_hf.index)
            propagate_mask = has_k_total & lacks_k_meas
            if propagate_mask.any():
                df_hf.loc[propagate_mask, "k_measured"] = df_hf.loc[propagate_mask, "k_total"]
                log.info(
                    "FIX-T3-8B: propagated k_total → k_measured for %d DFT HfO2-family rows",
                    int(propagate_mask.sum()),
                )
        log.info("  HfO2-family DFT entries: %d", len(df_hf))

        # Merge with experimental data
        if len(df_exp) > 0:
            # Experimental entries may not have atoms_dict -- that is acceptable
            # They are used for process-parameter regression branches of Model 1
            # but cannot feed into ALIGNN graph path directly
            df_exp_aligned = df_exp.copy()

            # Ensure consistent column presence
            for col in df_hf.columns:
                if col not in df_exp_aligned.columns:
                    df_exp_aligned[col] = None

            # FIX-PROC-DROP: was df_exp_aligned[df_hf.columns] which silently
            # dropped ALL process parameter columns (substrate_temp_C, anneal_temp_C,
            # growth_rate_A_per_cycle, n_cycles, film_thickness_A, pressure_mTorr,
            # anneal_duration_min, anneal_ambient, precursor_type, oxidant_type)
            # because df_hf is DFT-only and has none of them.
            # This was the root cause of proc_avail_pct=0% through all runs.
            # Fix: concat full df_exp_aligned so proc param columns are preserved.
            # Pandas fills DFT rows with NaN for the proc columns automatically.
            df_tier3 = pd.concat(
                [df_hf, df_exp_aligned],
                ignore_index=True, sort=False
            )
        else:
            df_tier3 = df_hf.copy()

        # FIX-T3-5: Log per-source row counts BEFORE dedup so any hash collision
        # losses are immediately visible in the run log.
        n_pre_dedup   = len(df_tier3)
        n_exp_pre     = int((df_tier3["source"] == "Experimental").sum())
        n_dft_pre     = n_pre_dedup - n_exp_pre
        log.info(
            "Tier 3 pre-dedup  : %d total  (DFT-HfO2=%d  Experimental=%d)",
            n_pre_dedup, n_dft_pre, n_exp_pre,
        )

        # Final dedup
        df_tier3 = df_tier3.drop_duplicates(subset=["row_hash"], keep="first")

        n_post_dedup  = len(df_tier3)
        n_exp_post    = int((df_tier3["source"] == "Experimental").sum())
        n_dft_post    = n_post_dedup - n_exp_post
        n_lost        = n_pre_dedup - n_post_dedup
        n_exp_lost    = n_exp_pre - n_exp_post
        log.info(
            "Tier 3 post-dedup : %d total  (DFT-HfO2=%d  Experimental=%d)  "
            "lost=%d (DFT=%d  Experimental=%d)",
            n_post_dedup, n_dft_post, n_exp_post,
            n_lost, n_lost - n_exp_lost, n_exp_lost,
        )
        if n_exp_lost > 0:
            log.warning(
                "FIX-T3-5: %d experimental rows lost to row_hash deduplication.  "
                "Check process_db.csv for duplicate (paper_id + material + "
                "substrate_temp_C + precursor_type + oxidant_type + n_cycles) "
                "combinations — these are true duplicates that should be merged "
                "or given distinct paper_id values.",
                n_exp_lost,
            )
        df_tier3 = df_tier3.drop_duplicates(subset=["row_hash"], keep="first")
        df_tier3["tier"] = 3
        log.info("Tier 3 final count: %d rows", len(df_tier3))

        df_tier3.to_hdf(self.TIER_PATHS[3], key="data", mode="w",
                        complevel=6, complib="blosc")

        self.manifest["tiers"]["3"]["row_count"]    = len(df_tier3)
        self.manifest["tiers"]["3"]["last_updated"] = datetime.date.today().isoformat()
        self.manifest["tiers"]["3"]["breakdown"] = {
            "DFT_HfO2_family": int((df_tier3["source"] != "Experimental").sum()),
            "Experimental":    int((df_tier3["source"] == "Experimental").sum()),
        }
        self._save_manifest()
        self._log_stats("Tier 3", df_tier3)
        return df_tier3

    # --------------------------------------------------------------------------
    # 2d. Scalable append -- add new entries to any tier
    # --------------------------------------------------------------------------
    def append_to_tier(
        self,
        df_new: pd.DataFrame,
        tier: int,
        source_label: str = "external",
    ) -> pd.DataFrame:
        """
        Safely append new rows to any tier without breaking existing data.

        This is the scalability entry point -- called when:
        - JARVIS releases a new version (monthly)
        - New experimental papers are processed
        - Active learning queries new candidates
        - Lab synthesis results arrive

        The row_hash deduplication guarantees no duplicates even if this
        method is called multiple times with overlapping data.
        """
        tier_path = self.TIER_PATHS[tier]

        if tier_path.exists():
            df_existing    = pd.read_hdf(tier_path, key="data")
            existing_hashes = set(df_existing["row_hash"].tolist())
        else:
            df_existing    = pd.DataFrame()
            existing_hashes = set()

        # Compute hashes for new entries if missing
        if "row_hash" not in df_new.columns:
            df_new = df_new.copy()
            df_new["row_hash"] = df_new.apply(
                lambda r: hashlib.md5(
                    f"{source_label}_{r.get('formula','')}_{r.get('jid',r.get('mp_id','?'))}".encode()
                ).hexdigest()[:12],
                axis=1,
            )

        # Remove duplicates
        df_unique = df_new[~df_new["row_hash"].isin(existing_hashes)].copy()
        df_unique["tier"]       = tier
        df_unique["date_added"] = datetime.date.today().isoformat()

        n_added    = len(df_unique)
        n_skipped  = len(df_new) - n_added
        log.info("Tier %d append: %d new rows added, %d duplicates skipped",
                 tier, n_added, n_skipped)

        if n_added == 0:
            return df_existing

        df_combined = pd.concat([df_existing, df_unique],
                                 ignore_index=True, sort=False)
        df_combined.to_hdf(tier_path, key="data", mode="w",
                           complevel=6, complib="blosc")

        # Log growth event
        self.manifest["tiers"][str(tier)]["row_count"] = len(df_combined)
        self.manifest["growth_log"].append({
            "date":       datetime.date.today().isoformat(),
            "tier":       tier,
            "rows_added": n_added,
            "source":     source_label,
        })
        self._save_manifest()
        return df_combined

    @staticmethod
    def _log_stats(label: str, df: pd.DataFrame):
        log.info("-" * 60)
        log.info(" %s  statistics", label)
        log.info("  Total rows:          %d", len(df))
        if "k_measured" in df.columns:
            k = df["k_measured"].dropna()
            log.info("  k_measured:          %d rows, mean=%.1f, max=%.1f",
                     len(k), k.mean() if len(k) else 0, k.max() if len(k) else 0)
            log.info("  k > 35 (target):     %d rows (%.1f%%)",
                     (k > 35).sum(), 100 * (k > 35).mean() if len(k) else 0)
        if "band_gap" in df.columns:
            bg = df["band_gap"].dropna()
            log.info("  band_gap:            %d rows, mean=%.2f eV", len(bg),
                     bg.mean() if len(bg) else 0)
        if "source" in df.columns:
            for src, cnt in df["source"].value_counts().items():
                log.info("  %-22s %d", src, cnt)
        log.info("-" * 60)


# ==============================================================================
# SECTION 3 -- ALIGNN GRAPH CONSTRUCTION
# ==============================================================================

class HighKGraphDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that wraps the tiered HDF5 stores and builds ALIGNN
    DGL graphs on-the-fly from stored atoms_dict JSON strings.

    Graph construction follows the ALIGNN paper exactly:
    - 12-nearest-neighbor periodic graph for crystals
    - RBF edge features: bond distances expanded over [0, 8 Å]
    - Line graph: nodes are bonds, edges are bond-angle triplets
    - RBF triplet features: bond angle cosines

    For QM9 molecules: same construction with 5 Å cutoff (no periodicity).
    For experimental entries without atoms_dict: returns None (excluded from
    ALIGNN path, used only in the process-parameter MLP branch).
    """

    RBF_CUTOFF_CRYSTAL  = 8.0     # Å -- matches ALIGNN paper
    RBF_CUTOFF_MOLECULE = 5.0     # Å -- matches QM9 treatment in paper
    N_NEIGHBORS         = 12      # periodic nearest-neighbors

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "k_measured",
        aux_cols: List[str] = None,
        cutoff: float = None,
        use_canonize: bool = True,
    ):
        self.df          = df.reset_index(drop=True)
        self.target_col  = target_col
        self.aux_cols    = aux_cols or []
        self.cutoff      = cutoff
        self.use_canonize = use_canonize

        # -- Build proc/stack vocab maps for fast __getitem__ extraction -------
        # Keyed by PROCESS_PARAMS_FEATURES / STACK_CONTEXT_FEATURES so dataset
        # and encoder classes always use identical vocabularies.
        _pf = PROCESS_PARAMS_FEATURES
        self._proc_num_cols = list(_pf["numerical"])
        self._proc_log_cols = set(_pf.get("log_normalize", []))
        self._proc_cat_cols = list(_pf["categorical"].keys())
        self._proc_vocabs   = {
            name: {v: i for i, v in enumerate(vocab)}
            for name, vocab in _pf["categorical"].items()
        }
        _sf = STACK_CONTEXT_FEATURES
        self._stack_num_cols = list(_sf["numerical"])
        self._stack_cat_cols = list(_sf["categorical"].keys())
        self._stack_vocabs   = {
            name: {v: i for i, v in enumerate(vocab)}
            for name, vocab in _sf["categorical"].items()
        }

        # -- Context fast-path: detect once whether any context columns exist --
        # For Tier 1/2 (JARVIS/MP/QM9) no ALD or stack columns are present in
        # the dataframe.  _has_context=False lets __getitem__ skip the entire
        # _extract_proc_feats / _extract_stack_feats iteration and return
        # pre-built zero lists directly -- eliminates ~4M pandas row lookups
        # per Tier 1 epoch with zero change in correctness.
        _all_ctx_cols = (self._proc_num_cols + self._proc_cat_cols +
                         self._stack_num_cols + self._stack_cat_cols)
        self._has_context = any(c in self.df.columns for c in _all_ctx_cols)

        # Pre-compute zero values (built once at init, reused for every row
        # when _has_context=False -- avoids per-sample list allocation)
        self._zero_proc_num  = [0.0] * len(self._proc_num_cols)
        self._zero_proc_cat  = [len(self._proc_vocabs[c])
                                 for c in self._proc_cat_cols]   # padding idx
        self._zero_stack_num = [0.0] * len(self._stack_num_cols)
        self._zero_stack_cat = [len(self._stack_vocabs[c])
                                 for c in self._stack_cat_cols]

        # Pre-filter to rows with valid target AND valid structure
        valid = (
            self.df[target_col].notna() &
            self.df["atoms_dict"].notna() &
            self.df["has_structure"].fillna(False)
        )
        self.valid_idx = self.df[valid].index.tolist()
        log.info(
            "HighKGraphDataset: %d/%d rows have valid target '%s' + structure",
            len(self.valid_idx), len(self.df), target_col
        )

    # -- Context feature extraction helpers -----------------------------------
    @staticmethod
    def _safe_val(row, col):
        """Return row[col] as float if present and numeric, else None.

        FIX-SAFE-VAL: previously only guarded against None and float NaN.
        After FIX-PROC-DROP proc columns now survive into df_structural but
        some cells contain non-numeric strings from the CSV ('RT', '90+10',
        'varies', 'atm', '10Pa') that were not caught by _clean_numeric_series.
        pd.to_numeric conversion is now applied so strings → None (skipped).
        """
        try:
            v = row[col]
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            # Coerce strings to numeric; non-convertible → NaN → None
            fv = pd.to_numeric(v, errors="coerce")
            if pd.isna(fv):
                return None
            return fv
        except (KeyError, TypeError):
            return None

    def _extract_proc_feats(
        self, row
    ) -> Tuple[List[float], List[int], float]:
        """
        Extract ALD process parameter features from a row.

        Returns (num_vals, cat_indices, avail_flag).
        avail_flag = 1.0 if ANY feature is non-missing, else 0.0.
        For all Tier 1/2 rows the columns are absent → avail_flag = 0.0
        → proc branch contributes zero to the fused embedding.
        """
        num_vals    = []
        any_present = False
        for col in self._proc_num_cols:
            v = self._safe_val(row, col)
            if v is not None:
                fv = float(v)
                if col in self._proc_log_cols:
                    fv = float(np.log1p(max(fv, 0.0)))
                num_vals.append(fv)
                any_present = True
            else:
                num_vals.append(0.0)   # masked; normalisation centres these near 0

        cat_indices = []
        for col in self._proc_cat_cols:
            v    = self._safe_val(row, col)
            vmap = self._proc_vocabs[col]
            if v is not None:
                idx = vmap.get(str(v), len(vmap))   # unknown value → vocab_size slot
                any_present = True
            else:
                idx = len(vmap)                      # absent → vocab_size slot
            cat_indices.append(idx)

        return num_vals, cat_indices, 1.0 if any_present else 0.0

    def _extract_stack_feats(
        self, row
    ) -> Tuple[List[float], List[int], float]:
        """Extract device stack context features from a row."""
        num_vals    = []
        any_present = False
        for col in self._stack_num_cols:
            v = self._safe_val(row, col)
            if v is not None:
                num_vals.append(float(v))
                any_present = True
            else:
                num_vals.append(0.0)

        cat_indices = []
        for col in self._stack_cat_cols:
            v    = self._safe_val(row, col)
            vmap = self._stack_vocabs[col]
            if v is not None:
                idx = vmap.get(str(v), len(vmap))
                any_present = True
            else:
                idx = len(vmap)
            cat_indices.append(idx)

        return num_vals, cat_indices, 1.0 if any_present else 0.0

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        row_idx = self.valid_idx[idx]
        # FIX-ILOC: use .loc[] not .iloc[] — valid_idx stores index LABELS
        # (from self.df[valid].index.tolist()), not integer positions.
        # When df has a non-contiguous index, iloc[label] reads the wrong row.
        row     = self.df.loc[row_idx]

        # -- Parse atoms ---------------------------------------------------
        try:
            atoms_dict = json.loads(row["atoms_dict"])
            j_atoms    = JAtoms.from_dict(atoms_dict)
        except Exception as e:
            log.debug("Atom parse failed for row %d: %s", row_idx, e)
            return None

        # -- Cutoff selection ----------------------------------------------
        # FIX-DGL-SCHEMA: force is_mol=False for imputed experimental rows.
        # Imputed rows inherit atoms_dict from JARVIS/MP crystal donors.
        # If the donor row had is_molecule=True (e.g. a QM9 molecule matched
        # via formula), use_lattice_prop would be False, building a graph
        # with different node schema than crystal rows → DGLError on dgl.batch().
        # Experimental ALD films are always crystalline → always use crystal path.
        is_mol = False if bool(row.get("is_experimental", 0.0)) else \
                 bool(row.get("is_molecule", False))
        cutoff  = self.cutoff or (
            self.RBF_CUTOFF_MOLECULE if is_mol else self.RBF_CUTOFF_CRYSTAL
        )

        # -- Build ALIGNN graph + line graph -------------------------------
        try:
            graph, line_graph = Graph.atom_dgl_multigraph(
                j_atoms,
                cutoff           = cutoff,
                max_neighbors    = self.N_NEIGHBORS,
                use_canonize     = self.use_canonize,
                use_lattice_prop = not is_mol,
                compute_line_graph = True,    # CRITICAL: enables line graph
             )
        except Exception as e:
            log.debug("DGL graph build failed for row %d: %s", row_idx, e)
            return None
        except BaseException as e:
            # FIX-DGL-SCHEMA: DGLError inherits from BaseException not Exception
            # in some DGL versions. Catch explicitly so bad graphs return None
            # rather than crashing the DataLoader worker process.
            log.debug("DGLError in graph build for row %d: %s", row_idx, e)
            return None

        # FIX-DGL-SCHEMA: validate graph schema before returning.
        # Graphs with 0 nodes, 0 edges, or non-standard atom_features shape
        # cause dgl.batch() to fail with schema mismatch in collate_fn.
        # Return None so collate_fn filters these out safely.
        try:
            _nf = graph.ndata.get("atom_features")
            if (graph.num_nodes() < 2 or
                    graph.num_edges() < 1 or
                    line_graph.num_nodes() < 1 or
                    _nf is None or
                    _nf.shape[-1] != 92):     # ALIGNN standard: 92-dim one-hot
                log.debug(
                    "Row %d: degenerate graph "
                    "(nodes=%d edges=%d lg_nodes=%d feat_dim=%s) → skip",
                    row_idx, graph.num_nodes(), graph.num_edges(),
                    line_graph.num_nodes(),
                    str(_nf.shape[-1]) if _nf is not None else "None",
                )
                return None
        except Exception:
            return None

        # -- Target --------------------------------------------------------
        target = torch.tensor([float(row[self.target_col])], dtype=torch.float32)

        # -- Auxiliary targets (multi-task) --------------------------------
        # FIX2: always return float32 tensor -- NaN if missing.
        # This lets collate_fn stack them and evaluate_multitask() mask them.
        aux_targets = {}
        for col in self.aux_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                aux_targets[col] = torch.tensor([float(val)], dtype=torch.float32)
            else:
                aux_targets[col] = torch.tensor([float("nan")], dtype=torch.float32)

        # -- Process params + stack context (v2.2 context branches) -----------
        # Fast path: skip pandas column iteration entirely when no context
        # columns exist in this tier's dataframe (all Tier 1/2 rows).
        # avail_flag=0.0 guarantees zero contribution in _fuse regardless.
        if self._has_context:
            p_num, p_cat, p_avail = self._extract_proc_feats(row)
            s_num, s_cat, s_avail = self._extract_stack_feats(row)
        else:
            p_num,  p_cat,  p_avail = self._zero_proc_num,  self._zero_proc_cat,  0.0
            s_num,  s_cat,  s_avail = self._zero_stack_num, self._zero_stack_cat, 0.0
        proc_context = {
            "num":   torch.tensor(p_num,   dtype=torch.float32),
            "cat":   torch.tensor(p_cat,   dtype=torch.long),
            "avail": torch.tensor([p_avail], dtype=torch.float32),
        }
        stack_context = {
            "num":   torch.tensor(s_num,   dtype=torch.float32),
            "cat":   torch.tensor(s_cat,   dtype=torch.long),
            "avail": torch.tensor([s_avail], dtype=torch.float32),
        }

        # v4 #11: functional_code wired through to model conditioning.
        # IMPORTANT: use pd.isna() not `or 1` — 0 (OptB88vdW) is falsy in
        # Python so `int(0.0) or 1` = 1, which silently relabels every JARVIS
        # entry (functional_code=0) as PBE=1, defeating functional conditioning
        # for the entire JARVIS source.
        _fc_raw = pd.to_numeric(row.get("functional_code", 1), errors="coerce")
        _fc_int = int(_fc_raw) if not pd.isna(_fc_raw) else 1

        return {
            "graph":            graph,
            "line_graph":       line_graph,
            "target":           target,
            "aux_targets":      aux_targets,
            "proc_context":     proc_context,
            "stack_context":    stack_context,
            "row_idx":          row_idx,
            "formula":          row.get("formula", ""),
            "source":           row.get("source", ""),
            "functional_code":  torch.tensor(_fc_int, dtype=torch.long),
            # FIX-T3-9B: flag for per-sample experimental upweighting in loss.
            # 1.0 = experimental ALD measurement; 0.0 = DFT computed row.
            "is_experimental":  torch.tensor(
                [1.0 if str(row.get("source", "")) == "Experimental" else 0.0],
                dtype=torch.float32,
            ),
        }

    @staticmethod
    def collate_fn(batch):
        """FIX2: stacks aux_targets per-task for multi-task evaluation.
        v2.2: also stacks proc_context and stack_context tensors per batch.
        All aux_target values are NaN-tensors (not None) after __getitem__ fix,
        so torch.stack works safely across the whole batch.

        FIX-DGL-SCHEMA: validate graph schema before dgl.batch() to prevent
        DGLError when imputed rows produce graphs with different node schemas.
        Graphs that don't match the batch majority schema are silently dropped.
        """
        import dgl
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # FIX-DGL-SCHEMA: filter out graphs with non-standard schema
        # before dgl.batch() to prevent worker process crashes.
        # Reference schema from the first valid graph.
        def _graph_schema_key(g):
            """Return a hashable schema signature for a DGL graph."""
            try:
                nf = g.ndata.get("atom_features")
                return (
                    tuple(sorted(g.ndata.keys())),
                    tuple(sorted(g.edata.keys())),
                    nf.shape[-1] if nf is not None else 0,
                )
            except Exception:
                return None

        ref_schema = _graph_schema_key(batch[0]["graph"])
        valid_batch = []
        for item in batch:
            try:
                if (_graph_schema_key(item["graph"]) == ref_schema and
                        _graph_schema_key(item["line_graph"]) is not None):
                    valid_batch.append(item)
                else:
                    log.debug(
                        "collate_fn: dropping row %s — graph schema mismatch",
                        item.get("row_idx", "?"),
                    )
            except Exception:
                pass
        batch = valid_batch
        if not batch:
            return None

        try:
            graphs      = dgl.batch([b["graph"]      for b in batch])
            line_graphs = dgl.batch([b["line_graph"] for b in batch])
        except BaseException as e:
            # Last-resort: if dgl.batch still fails, log and skip this batch
            log.warning(
                "collate_fn: dgl.batch() failed (%s) — dropping batch of %d",
                str(e)[:120], len(batch),
            )
            return None

        targets     = torch.stack([b["target"]   for b in batch])

        # FIX2: stack aux_targets per task key (all are tensors after __getitem__ fix)
        aux_targets = {}
        if batch[0].get("aux_targets"):
            for key in batch[0]["aux_targets"].keys():
                aux_targets[key] = torch.stack(
                    [b["aux_targets"][key] for b in batch]
                )

        # v2.2: stack proc/stack context -- keys: num, cat, avail
        proc_context  = None
        stack_context = None
        if batch[0].get("proc_context") is not None:
            proc_context = {
                k: torch.stack([b["proc_context"][k]  for b in batch])
                for k in batch[0]["proc_context"].keys()
            }
            stack_context = {
                k: torch.stack([b["stack_context"][k] for b in batch])
                for k in batch[0]["stack_context"].keys()
            }

        # v4 #11: stack functional codes for per-sample functional conditioning
        functional_codes = None
        if "functional_code" in batch[0]:
            functional_codes = torch.stack([b["functional_code"] for b in batch])

        # FIX-T3-9B: stack is_experimental flags for per-sample loss upweighting
        is_experimental = None
        if "is_experimental" in batch[0]:
            is_experimental = torch.stack([b["is_experimental"] for b in batch])

        return {
            "graph":            graphs,
            "line_graph":       line_graphs,
            "target":           targets,
            "aux_targets":      aux_targets,
            "proc_context":     proc_context,
            "stack_context":    stack_context,
            "formulas":         [b["formula"] for b in batch],
            "functional_code":  functional_codes,   # LongTensor [B] or None
            "is_experimental":  is_experimental,    # FloatTensor [B,1] or None
            "row_indices":      torch.tensor([b["row_idx"] for b in batch]), # [B]
        }


def get_stratified_split(
    dataset: HighKGraphDataset,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
    target_col: str   = None,
) -> Tuple[Subset, Subset, Subset]:
    """
    Stratified split on target bins to ensure balanced distribution across
    all three splits. Falls back to random split if stratification fails.

    For continuous targets, bins are created based on percentile ranges.
    If any bin has < 2 samples, falls back to random split.

    FIX: use pd.to_numeric(errors='coerce') instead of .astype(float).
    Root cause of WARNING: target column has dtype('O') (Python object) because
    _safe_float returns Python None (not np.nan) for missing values.  When
    pandas builds a DataFrame from dicts containing None mixed with float, the
    column dtype becomes object.  np.ndarray.astype(float) with casting='safe'
    (NumPy >= 1.20) rejects object→float64 and raises, triggering the except
    branch every run.  pd.to_numeric(errors='coerce') handles None, np.nan,
    Python float objects and numeric strings without ever raising.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    # FIX-ILOC-2: use .loc[] not .iloc[] — valid_idx contains index LABELS,
    # not integer positions. iloc would read wrong rows when df has a non-
    # contiguous index (e.g. after filtering df_structural without reset_index).
    df = dataset.df.loc[dataset.valid_idx].copy()

    if target_col is None:
        target_col = dataset.target_col

    # -- FIX: safe conversion regardless of column dtype -------------------
    # pd.to_numeric(errors='coerce') converts:
    #   Python float / int / np.float64  → float64
    #   Python None                      → NaN  (was causing the cast failure)
    #   np.nan                           → NaN
    #   non-numeric string               → NaN  (coerced, no exception)
    raw_vals       = df[target_col]
    target_vals_f  = pd.to_numeric(raw_vals, errors="coerce").values  # float64 always
    valid_mask     = ~np.isnan(target_vals_f)

    if not valid_mask.any():
        log.warning("All target values are NaN. Falling back to random split.")
        return get_random_split(dataset, train_frac, val_frac, seed)

    try:
        n_valid = int(valid_mask.sum())
        n_bins  = min(6, n_valid // 10)   # at least 10 samples per bin
        if n_bins < 2:
            log.warning(
                "Not enough samples (%d) for stratification. "
                "Falling back to random split.", n_valid
            )
            return get_random_split(dataset, train_frac, val_frac, seed)

        # Percentile-based bin edges on valid (non-NaN) values only
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins_edges  = np.percentile(target_vals_f[valid_mask], percentiles)
        bins_edges  = np.unique(bins_edges)
        if len(bins_edges) < 2:
            log.warning(
                "Not enough unique bins for stratification. "
                "Falling back to random split."
            )
            return get_random_split(dataset, train_frac, val_frac, seed)

        # Assign bins -- use float64 array throughout; assign NaN rows to
        # the median bin so they are distributed proportionally across splits
        target_bins = np.digitize(target_vals_f, bins_edges[1:-1]).astype(int)
        nan_rows    = ~valid_mask
        if nan_rows.any():
            median_bin = int(np.median(target_bins[valid_mask]))
            target_bins[nan_rows] = median_bin

        # Composite stratification: stratify by (target_bin, proc_avail) so each
        # rank gets a representative mix of DFT (proc=0) and experimental (proc=1) rows.
        try:
            proc_avail = np.array([
                int(dataset.df.loc[dataset.valid_idx[i], "source"] == "Experimental")
                for i in range(len(dataset))
            ])
        except Exception:
            proc_avail = np.zeros(len(dataset), dtype=int)

        composite_bins = target_bins * 2 + proc_avail

        unique, counts = np.unique(target_bins, return_counts=True)
        if (counts < 2).any():
            log.warning(
                "Some bins have < 2 samples. Falling back to random split."
            )
            return get_random_split(dataset, train_frac, val_frac, seed)

        # FIX : Handle edge case where train_Frac=0.0 (evaluation-only).
        # when train_frac=0.0, test_size=1.0 in the first stratifiedShuffleSplit,
        # which sklearn rejects ("test_size=1.0 should be < n_samples").
        # Fix: skip the first split and go directly to val/test split
        if train_frac < 1e-9:
            # All data goes to temp, split directly into val/test
            train_idx = np.array([], dtype=int)
            temp_idx  = np.arange(len(dataset))
            composite_temp = composite_bins
            target_bins_temp = target_bins
        else:
            sss     = StratifiedShuffleSplit(
                n_splits=1, test_size=(1 - train_frac), random_state=seed
            )
            idx_all = np.arange(len(dataset))
            for train_idx, temp_idx in sss.split(idx_all, composite_bins):
                pass

            target_bins_temp = target_bins[temp_idx]
            composite_temp = composite_bins[temp_idx]

        test_ratio       = (1 - train_frac - val_frac) / max(1 - train_frac, 1e-9)

        # -- Fix: guard sss2 against two failure modes --------------------
        #
        # Mode A — test_ratio == 0 (Phase A uses train_frac=0.95, val_frac=0.05
        #   so test_ratio = 0/0.05 = 0.0).  sklearn converts test_size=0.0 to
        #   max(1, int(0×N)) = 1, then raises "test_size=1 should be >= n_classes=6".
        #   Fix: when test_ratio < epsilon assign all temp to val, no test split.
        #
        # Mode B — n_test_int < n_unique_bins_temp (any small dataset where
        #   there are fewer test slots than stratification bins).
        #   Fix: fall back to a deterministic random shuffle for val/test only.
        if test_ratio < 1e-9:
            # No test set needed (Phase A: train+val covers 100%)
            val_idx  = temp_idx
            test_idx = np.array([], dtype=int)
        else:
            n_unique_bins_temp = len(np.unique(composite_temp))
            n_test_int         = max(1, int(test_ratio * len(temp_idx)))
            if n_test_int < n_unique_bins_temp:
                # Too few test slots for stratified split — use random shuffle
                log.warning(
                    "Second stratified split skipped (n_test=%d < n_bins=%d); "
                    "using random val/test split on the %d temp samples.",
                    n_test_int, n_unique_bins_temp, len(temp_idx),
                )
                rng_local = np.random.default_rng(seed + 1)
                perm      = rng_local.permutation(len(temp_idx))
                cut       = max(1, int(len(temp_idx) * (val_frac / (1 - train_frac))))
                val_idx   = temp_idx[perm[:cut]]
                test_idx  = temp_idx[perm[cut:]]
            else:
                sss2 = StratifiedShuffleSplit(
                    n_splits=1, test_size=test_ratio, random_state=seed
                )
                for val_idx_local, test_idx_local in sss2.split(
                    np.arange(len(temp_idx)), composite_temp
                ):
                    pass
                val_idx  = temp_idx[val_idx_local]
                test_idx = temp_idx[test_idx_local]

    except Exception as e:
        log.warning("Stratification failed: %s. Falling back to random split.", e)
        return get_random_split(dataset, train_frac, val_frac, seed)

    log.info(
        "Stratified split -- train: %d  val: %d  test: %d"
        "  (bins: %d  target: %s)",
        len(train_idx), len(val_idx), len(test_idx), n_bins, target_col,
    )
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )

def get_random_split(
    dataset: HighKGraphDataset,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Fallback random split if stratification fails."""
    np.random.seed(seed)
    idx_all = np.arange(len(dataset))
    np.random.shuffle(idx_all)

    n_train = int(len(idx_all) * train_frac)
    n_val = int(len(idx_all) * val_frac)

    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:n_train + n_val]
    test_idx = idx_all[n_train + n_val:]

    log.info(
        "Random split -- train: %d  val: %d  test: %d",
        len(train_idx), len(val_idx), len(test_idx)
    )
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


# ==============================================================================
# SECTION 4 -- PROCESS / STACK CONTEXT ENCODERS  (v2.2)
# ==============================================================================

class ProcessParamsEncoder(nn.Module):
    """
    Encodes ALD/deposition process parameters into a fixed 64-dim embedding.

    Numerical features are z-score normalised (log1p first for skewed columns).
    Normalization stats are fitted from training data via fit_stats(); for
    Tier 1/2 checkpoints where no proc data exists, stats stay at mean=0 std=1
    (correct because all inputs are 0.0 in those tiers).

    Categorical features use per-feature nn.Embedding tables (dim=8).
    Absent / unknown values map to index vocab_size (the extra embedding slot).

    Final gate:  emb = mlp(cat(num_norm, cat_embs)) * avail_flag
    When avail_flag = 0 the output is a zero vector -- no contribution to fusion.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        cfg = cfg or PROCESS_PARAMS_FEATURES
        self._num_cols  = list(cfg["numerical"])
        self._log_cols  = set(cfg.get("log_normalize", []))
        self._cat_spec  = cfg["categorical"]           # {name: [val, ...]}
        self._cat_cols  = list(self._cat_spec.keys())
        embed_dim       = cfg.get("embed_dim",  8)
        output_dim      = cfg.get("output_dim", 64)

        # vocab_size for each categorical (unknown/absent → vocab_size index)
        self._vocab_sizes = {name: len(v) for name, v in self._cat_spec.items()}

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(vocab) + 1, embed_dim)
            for name, vocab in self._cat_spec.items()
        })

        n_num     = len(self._num_cols)
        n_cat_emb = len(self._cat_cols) * embed_dim
        self.mlp  = nn.Sequential(
            nn.Linear(n_num + n_cat_emb, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.SiLU(),
        )

        # Running normalisation buffers -- fitted per-tier from training data
        self.register_buffer("num_mean", torch.zeros(n_num))
        self.register_buffer("num_std",  torch.ones(n_num))

    def fit_stats(self, df: pd.DataFrame):
        """Fit z-score normalisation stats from a training dataframe.

        FIX-FIT-STATS: apply pd.to_numeric(errors='coerce') before dropna().
        After FIX-PROC-DROP proc columns now reach df_structural but some
        cells contain non-numeric strings ('RT', '90+10', 'varies', 'atm',
        '10Pa') that pass _safe_val and then crash vals.mean() with TypeError.
        """
        means, stds = [], []
        for col in self._num_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if col in self._log_cols:
                    vals = np.log1p(vals.clip(lower=0))
                means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
                stds.append(float(vals.std())   if len(vals) > 1 else 1.0)
            else:
                means.append(0.0)
                stds.append(1.0)
        self.num_mean.copy_(torch.tensor(means, dtype=torch.float32))
        self.num_std.copy_(
            torch.tensor(stds,  dtype=torch.float32).clamp(min=1e-6)
        )

    def forward(
        self,
        num_feats:   torch.Tensor,   # (B, n_num)  -- raw (pre-normalisation)
        cat_indices: torch.Tensor,   # (B, n_cat)
        avail_flag:  torch.Tensor,   # (B, 1)      -- 0.0 if all features absent
    ) -> torch.Tensor:               # (B, output_dim)
        # Normalise numericals; zero-mask the whole vector when branch absent
        num_norm = (num_feats - self.num_mean) / self.num_std
        num_norm = num_norm * avail_flag

        # Categorical embeddings
        cat_embs = [
            self.embeddings[name](cat_indices[:, i])
            for i, name in enumerate(self._cat_cols)
        ]                                             # each (B, embed_dim)

        x   = torch.cat([num_norm] + cat_embs, dim=-1)  # (B, n_num + n_cat*embed_dim)
        emb = self.mlp(x)
        return emb * avail_flag                           # final gate


class StackContextEncoder(nn.Module):
    """
    Encodes device stack / interface context into a fixed 64-dim embedding.

    Same architecture as ProcessParamsEncoder; different feature set.
    See STACK_CONTEXT_FEATURES for full feature list.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        cfg = cfg or STACK_CONTEXT_FEATURES
        self._num_cols  = list(cfg["numerical"])
        self._log_cols  = set(cfg.get("log_normalize", []))
        self._cat_spec  = cfg["categorical"]
        self._cat_cols  = list(self._cat_spec.keys())
        embed_dim       = cfg.get("embed_dim",  8)
        output_dim      = cfg.get("output_dim", 64)

        self._vocab_sizes = {name: len(v) for name, v in self._cat_spec.items()}

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(vocab) + 1, embed_dim)
            for name, vocab in self._cat_spec.items()
        })

        n_num    = len(self._num_cols)
        n_cat_emb = len(self._cat_cols) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_num + n_cat_emb, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.SiLU(),
        )

        self.register_buffer("num_mean", torch.zeros(n_num))
        self.register_buffer("num_std",  torch.ones(n_num))

    def fit_stats(self, df: pd.DataFrame):
        """Fit z-score normalisation stats from a training dataframe.

        FIX-FIT-STATS: same pd.to_numeric(errors='coerce') guard as
        ProcessParamsEncoder.fit_stats — stack context columns may also
        contain non-numeric strings after FIX-PROC-DROP.
        """
        means, stds = [], []
        for col in self._num_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if col in self._log_cols:
                    vals = np.log1p(vals.clip(lower=0))
                means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
                stds.append(float(vals.std())   if len(vals) > 1 else 1.0)
            else:
                means.append(0.0)
                stds.append(1.0)
        self.num_mean.copy_(torch.tensor(means, dtype=torch.float32))
        self.num_std.copy_(
            torch.tensor(stds,  dtype=torch.float32).clamp(min=1e-6)
        )

    def forward(
        self,
        num_feats:   torch.Tensor,
        cat_indices: torch.Tensor,
        avail_flag:  torch.Tensor,
    ) -> torch.Tensor:
        num_norm = (num_feats - self.num_mean) / self.num_std
        num_norm = num_norm * avail_flag

        cat_embs = [
            self.embeddings[name](cat_indices[:, i])
            for i, name in enumerate(self._cat_cols)
        ]

        x   = torch.cat([num_norm] + cat_embs, dim=-1)
        emb = self.mlp(x)
        return emb * avail_flag


# ==============================================================================
# SECTION 5 -- ALIGNN MODEL WITH TRANSFER LEARNING  (was Section 4)
# ==============================================================================

class HighKALIGNN(nn.Module):
    """
    ALIGNN model extended with:
    1. Multi-task output heads (k, band_gap, J_g, E_BD)
    2. Layer-selective freezing for transfer learning
    3. Uncertainty quantification via MC-Dropout

    Architecture matches paper Table 1 defaults:
    - 4 ALIGNN layers + 4 GCN layers
    - Hidden dim: 256, edge features: 80, triplet features: 40
    """

    def __init__(
        self,
        config: dict = None,
        n_output_tasks: int = 1,
        dropout_rate: float = 0.1,
        task_names: list = None,
    ):
        super().__init__()
        cfg = {**ALIGNN_BASE_CONFIG, **(config or {})}
        self.n_tasks     = n_output_tasks
        self.dropout_rate = dropout_rate

        # Default task names if not specified.
        # CKPT-RT1 FIX: updated from ["k_measured","band_gap","J_g_log","E_BD"]
        # which caused RuntimeError when loading v4.14+ checkpoints (k_total_log
        # head) into any code path that omits task_names (evaluate, ablation).
        # Defaults now match TIER3_TRAIN_CONFIG target + aux_targets exactly.
        if task_names is None:
            task_names = ["k_total_log", "band_gap", "J_g_A_cm2", "E_BD_MV_cm"]
        self.task_names = task_names

        # -- Core ALIGNN backbone ------------------------------------------
        alignn_cfg = ALIGNNConfig(
            name            = "alignn",
            alignn_layers   = cfg["alignn_layers"],
            gcn_layers      = cfg["gcn_layers"],
            edge_input_features  = cfg["edge_input_features"],
            triplet_input_features = cfg["triplet_input_features"],
            embedding_features = cfg["embedding_features"],
            hidden_features = cfg["hidden_features"],
            output_features = cfg["output_features"], # Match hidden for embedding output
        )
        self.backbone = ALIGNN(alignn_cfg)

        # Remove the default single output head
        # We replace it with multi-task heads
        backbone_out_dim = cfg["hidden_features"]

        # -- Multi-task heads ----------------------------------------------
        self.task_heads = nn.ModuleDict({
            name: self._make_head(backbone_out_dim) for name in task_names
        })

        # -- Dropout for MC uncertainty ------------------------------------
        self.dropout = nn.Dropout(p=dropout_rate)

        # -- Track which ALIGNN layers are frozen -------------------------
        self.frozen_layers = 0

        # -- v2.2: Process / stack context branches -----------------------
        # Compiled into ALL tier models (Tier 1/2/3) for clean weight transfer.
        # Branches contribute ZERO for Tier 1/2 rows because avail_flag=0
        # (no process/stack columns in JARVIS/MP/QM9 data).
        self.proc_encoder  = ProcessParamsEncoder()
        self.stack_encoder = StackContextEncoder()

        self.alpha = nn.Parameter(torch.tensor(1e-3))  # proc branch scale
        self.beta  = nn.Parameter(torch.tensor(1e-3))  # stack branch scale

        # Residual context projection  ctx[128] → delta[256]
        # ─────────────────────────────────────────────────────────────────
        # Input is ONLY cat([alpha*proc_emb, beta*stack_emb]) -- crystal_emb
        # is NOT passed through this MLP.  This is the key difference from
        # the earlier fusion_mlp formulation (which broke the Tier 1/2
        # pure-ALIGNN guarantee by transforming crystal_emb unconditionally).
        #
        # ReZero guarantee (zero-init on final Linear):
        #   ctx = 0  →  hidden = SiLU(W₁·0 + b₁) = SiLU(b₁)   [non-zero]
        #           →  delta  = W₂·SiLU(b₁) + 0  = 0           [W₂=0 init]
        #           →  fused  = crystal_emb + 0   = crystal_emb  ✓
        #
        # As Tier 3 training proceeds W₂ grows from zero and delta becomes a
        # learnable process/stack correction on top of the unmodified crystal
        # embedding -- without ever disturbing the Tier 1/2 baseline path.
        self.context_proj = nn.Sequential(
            nn.Linear(64 + 64, backbone_out_dim),          # 128 → 256
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(backbone_out_dim, backbone_out_dim), # 256 → 256  zero-init
        )
        nn.init.zeros_(self.context_proj[-1].weight)
        nn.init.zeros_(self.context_proj[-1].bias)

        # -- RESIDUAL-PROC: bounded process residual head -------------------
        # This head predicts a small additive correction in log(k) space on top
        # of the frozen DFT/crystal-only prediction:
        #   y = y_dft_base + process_delta_bound * tanh(delta_raw)
        # The final layer is zero-initialised so the model starts exactly at
        # the DFT-only baseline. By default it uses only the 7 numerical process
        # scalars; categorical embeddings can be enabled via config if enough
        # paper-diverse experimental data is available.
        n_proc_num = len(PROCESS_PARAMS_FEATURES["numerical"])
        n_proc_cat = len(PROCESS_PARAMS_FEATURES["categorical"]) * PROCESS_PARAMS_FEATURES.get("embed_dim", 8)
        self.process_delta_use_categorical = False
        self.process_delta_bound = 0.10
        self.use_bounded_process_residual = False
        self.process_delta_head = nn.Sequential(
            nn.Linear(n_proc_num + n_proc_cat, 64),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.process_delta_head[-1].weight)
        nn.init.zeros_(self.process_delta_head[-1].bias)

        # -- v4 change #11: functional_code conditioning -------------------
        # functional_code (0=OptB88vdW,1=PBE,2=r2SCAN,3=GGA+U,4=B3LYP) is
        # stored in every row but was metadata-only.  Adding it as a model
        # input allows the backbone to learn per-functional offsets for Ef
        # and band_gap, closing the dominant source of the 2.7× gap between
        # our Ef MAE and ALIGNN's paper result.
        #
        # ReZero init on func_proj: zero weight → no effect at the start of
        # training.  As gradient flows, the model learns to correct for
        # "this is PBE, add +0.16 eV/atom" vs "this is OptB88vdW, no shift".
        # Compatible with strict=False weight loading from older checkpoints
        # that lack these parameters.
        n_func = len(FUNCTIONAL_CODE) + 1   # +1 for unknown/missing code
        self.func_embedding = nn.Embedding(n_func, 16)
        self.func_proj      = nn.Linear(16, backbone_out_dim, bias=False)
        nn.init.zeros_(self.func_proj.weight)  # ReZero: no effect initially

    @staticmethod
    def _make_head(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def freeze_alignn_layers(self, n_layers: int):
        """
        Freeze the first n_layers ALIGNN update layers.
        Called at start of fine-tuning to preserve pretrained representations
        while allowing upper layers to adapt to the new domain.
        """
        self.frozen_layers = n_layers
        frozen_count = 0
        for i, layer in enumerate(self.backbone.alignn_layers[:n_layers]):
            for param in layer.parameters():
                param.requires_grad = False
            frozen_count += sum(p.numel() for p in layer.parameters())
        log.info(
            "Froze %d ALIGNN layers (%d parameters)",
            n_layers, frozen_count
        )

    def unfreeze_all(self):
        """Unfreeze all layers for final fine-tuning stage."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.frozen_layers = 0
        log.info("All layers unfrozen for fine-tuning.")

    def freeze_backbone(self):
        """Freeze ALIGNN backbone; keep task_heads, context, encoders, alpha/beta trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.context_proj.parameters():
            param.requires_grad = True
        for param in self.proc_encoder.parameters():
            param.requires_grad = True
        for param in self.stack_encoder.parameters():
            param.requires_grad = True
        for param in [ self.alpha, self.beta ]:
            param.requires_grad = True
        for param in self.task_heads.parameters():
            param.requires_grad = True
        for param in self.func_embedding.parameters():
            param.requires_grad = True
        for param in self.func_proj.parameters():
            param.requires_grad = True
        log.info(" ALIGNN backbone frozen. Training: task_heads, context_proj, proc_encoder, stack_encoder alpha, beta, func_embedding")

    def unfreeze_backbone(self, lr=None):
        """Unfreeze the ALIGNN backbone for gentle joint adaptation after Phase B."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        if lr is not None:
            log.info("Unfroze backbone at lr=%.2e", lr)
        else:
            log.info("ALIGNN backbone unfrozen (LR unchanged).")

    def freeze_for_bounded_process_residual(self, train_base_head: bool = False):
        """Freeze DFT base; train only the bounded process residual correction.

        If train_base_head=True, the k_total_log task head is also trainable, but
        the default is False so the run starts exactly from the DFT-only baseline
        and process parameters can only learn the residual.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.process_delta_head.parameters():
            p.requires_grad = True
        if train_base_head and "k_total_log" in self.task_heads:
            for p in self.task_heads["k_total_log"].parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "RESIDUAL-PROC freeze: DFT base frozen; train_base_head=%s; trainable_params=%d",
            train_base_head, n_trainable,
        )

    def _fuse(
        self,
        crystal_emb:   torch.Tensor,
        proc_context:  Optional[dict],
        stack_context: Optional[dict],
    ) -> torch.Tensor:
        """
        Residual context fusion.

        fused = crystal_emb + context_proj( cat([alpha*proc_emb,
                                                  beta*stack_emb]) )

        crystal_emb is NOT passed through context_proj -- it flows unchanged
        as the residual base.  context_proj only computes a delta correction
        from the process/stack branches.

        ReZero guarantee: context_proj[-1] is zero-initialised, so delta=0
        whenever ctx=0 (all Tier 1/2 samples, or ablation mode).  This means
        Tier 1/2 training is mathematically identical to pure ALIGNN.

        Performance fast-path: if both avail_flag tensors are all-zero (true
        for every Tier 1/2 batch), skip all encoder and projection computation
        and return crystal_emb directly.  The result is numerically identical
        to the full path (delta=0 by ReZero) but avoids all GPU kernel launches
        for encoders and context_proj on Tier 1/2.

        When either context dict is None (ablation flag or inference without
        context data) the corresponding branch contributes zero.
        """
        dev = crystal_emb.device
        B   = crystal_emb.size(0)

        # Fast path: skip all context computation when no data present.
        # Works for every Tier 1/2 batch (avail_flag all-zero from dataset)
        # and for ablation mode (proc_context/stack_context both None).
        proc_active  = (proc_context  is not None and
                        bool(proc_context["avail"].any()))
        stack_active = (stack_context is not None and
                        bool(stack_context["avail"].any()))
        if not proc_active and not stack_active:
            return crystal_emb   # exact pure-ALIGNN result; zero extra compute

        if proc_context is not None:
            proc_emb = self.proc_encoder(
                proc_context["num"].to(dev),
                proc_context["cat"].to(dev),
                proc_context["avail"].to(dev),
            )
        else:
            proc_emb = torch.zeros(B, 64, device=dev)

        if stack_context is not None:
            stack_emb = self.stack_encoder(
                stack_context["num"].to(dev),
                stack_context["cat"].to(dev),
                stack_context["avail"].to(dev),
            )
        else:
            stack_emb = torch.zeros(B, 64, device=dev)

        ctx   = torch.cat([self.alpha * proc_emb,
                           self.beta  * stack_emb], dim=-1)   # (B, 128)
        delta = self.context_proj(ctx)                         # (B, 256)  ≈ 0 at init
        return crystal_emb + delta                             # residual

    def fit_encoder_stats(self, df: pd.DataFrame):
        """
        Fit ProcessParamsEncoder and StackContextEncoder normalisation stats.

        Must be called AFTER loading pretrained weights so Tier 3 training
        data overwrites the uninformative zeros stored in Tier 1/2 checkpoints.
        For Tier 1/2 (no proc/stack columns in df) stats remain at mean=0,
        std=1, which is correct because all numerical inputs are 0.0 there.
        """
        self.proc_encoder.fit_stats(df)
        self.stack_encoder.fit_stats(df)
        log.info(
            "Encoder stats fitted  rows=%d  "
            "proc_num=%d  stack_num=%d",
            len(df),
            len(PROCESS_PARAMS_FEATURES["numerical"]),
            len(STACK_CONTEXT_FEATURES["numerical"]),
        )

    def configure_bounded_process_residual(
        self,
        enabled: bool = True,
        bound: float = 0.10,
        use_categorical: bool = False,
    ):
        """Enable/disable the DFT-base + bounded process-residual path."""
        self.use_bounded_process_residual = bool(enabled)
        self.process_delta_bound = float(bound)
        self.process_delta_use_categorical = bool(use_categorical)
        log.info(
            "RESIDUAL-PROC mode: enabled=%s  bound=%.3f log(k)  use_categorical=%s",
            self.use_bounded_process_residual,
            self.process_delta_bound,
            self.process_delta_use_categorical,
        )

    def _process_bounded_delta(self, proc_context: Optional[dict]) -> Optional[torch.Tensor]:
        """Return bounded process correction [B,1] in log(k) units, or None."""
        if proc_context is None:
            return None
        dev = next(self.parameters()).device
        avail = proc_context["avail"].to(dev)
        if not bool(avail.any()):
            # Build a differentiable zero correction so batches containing only
            # DFT rows still have a valid computation graph when the base model
            # is frozen. The gradient is zero, as intended.
            num = proc_context["num"].to(dev)
            dummy_in = torch.zeros(num.shape[0], self.process_delta_head[0].in_features, device=dev)
            return 0.0 * self.process_delta_head(dummy_in)

        num = proc_context["num"].to(dev)
        num_norm = (num - self.proc_encoder.num_mean.to(dev)) / self.proc_encoder.num_std.to(dev)
        num_norm = num_norm * avail

        if self.process_delta_use_categorical:
            cat = proc_context["cat"].to(dev)
            cat_embs = [
                self.proc_encoder.embeddings[name](cat[:, i])
                for i, name in enumerate(self.proc_encoder._cat_cols)
            ]
            x = torch.cat([num_norm] + cat_embs, dim=-1)
        else:
            # Keep the head input dimension fixed; zero-fill categorical slots.
            n_extra = self.process_delta_head[0].in_features - num_norm.shape[-1]
            if n_extra > 0:
                zeros = torch.zeros(num_norm.shape[0], n_extra, device=dev)
                x = torch.cat([num_norm, zeros], dim=-1)
            else:
                x = num_norm

        raw = self.process_delta_head(x)
        return self.process_delta_bound * torch.tanh(raw) * avail

    def forward(
        self,
        graph,
        line_graph,
        task:            str                     = "k_measured",
        proc_context:    Optional[dict]          = None,
        stack_context:   Optional[dict]          = None,
        functional_code: Optional[torch.Tensor]  = None,
    ):
        """
        Unified forward pass.

        task="__all__"  → returns Dict[str, Tensor]  (multi-task training)
        task=<name>     → returns Tensor             (single-task eval)

        Using a single forward() entry point is required for DDP:
        each process calls self.model(graph, lg, task="__all__", ...) and
        DDP synchronises gradients via AllReduce during backward — no
        custom method dispatch needed, no DataParallel scatter issues.
        """
        embedding = self.backbone((graph, line_graph, None))
        # In RESIDUAL-PROC mode the DFT base is frozen, so keep its prediction
        # deterministic during residual fitting. Standard modes retain dropout.
        if not self.use_bounded_process_residual:
            embedding = self.dropout(embedding)
        embedding = self._apply_func_conditioning(embedding, functional_code)

        # RESIDUAL-PROC mode: protect the DFT/crystal-only baseline.
        # The base prediction is made from the crystal embedding alone; process
        # data can only add a small bounded correction to k_total_log. This avoids
        # the randomly initialised process branch perturbing the representation.
        if self.use_bounded_process_residual:
            delta = self._process_bounded_delta(proc_context)
            if task == "__all__":
                out = {}
                for t, head in self.task_heads.items():
                    base = head(embedding)
                    out[t] = base + delta if (t == "k_total_log" and delta is not None) else base
                return out
            base = self.task_heads[task](embedding)
            if task == "k_total_log" and delta is not None:
                return base + delta
            return base

        fused     = self._fuse(embedding, proc_context, stack_context)

        if task == "__all__":
            return {t: head(fused) for t, head in self.task_heads.items()}
        return self.task_heads[task](fused)

    def forward_all_tasks(
        self,
        graph,
        line_graph,
        proc_context:    Optional[dict]          = None,
        stack_context:   Optional[dict]          = None,
        functional_code: Optional[torch.Tensor]  = None,
    ) -> Dict[str, torch.Tensor]:
        """Convenience wrapper — delegates to forward(task='__all__')."""
        return self(
            graph, line_graph,
            task            = "__all__",
            proc_context    = proc_context,
            stack_context   = stack_context,
            functional_code = functional_code,
        )

    def _apply_func_conditioning(
        self,
        crystal_emb:     torch.Tensor,
        functional_code: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Add functional-specific offset to crystal embedding.

        functional_code: LongTensor [B] with values in [0, n_func-1].
        If None or missing (old checkpoints), returns crystal_emb unchanged.

        ReZero init on func_proj means delta=0 at the start of training,
        preserving the exact backbone output until gradients teach the model
        the per-functional systematic offsets (PBE vs OptB88vdW for Ef, etc.).
        """
        if functional_code is None:
            return crystal_emb
        n_func = self.func_embedding.num_embeddings
        fc     = functional_code.long().clamp(0, n_func - 1).to(crystal_emb.device)
        delta  = self.func_proj(self.func_embedding(fc))   # [B, 256]
        return crystal_emb + delta

    def load_pretrained_weights(
        self, checkpoint_path: Path, strict: bool = False
    ):
        """
        Load pretrained weights with flexible matching.
        strict=False allows loading Tier N weights into Tier N+1 model
        even if output heads differ.

        CKPT-RT3 FIX: runs _remap_state_dict() before load_state_dict so
        checkpoints from v4.5.3–v4.5.10 (which used k_measured / k_measured_log
        as task head names) load correctly into v4.5.11+ models (k_total_log).
        """
        ckpt  = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        state = _remap_state_dict(state)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            log.info(
                "load_pretrained_weights: %d missing keys (new heads, "
                "will train from scratch): %s",
                len(missing), missing[:5],
            )
        if unexpected:
            log.info(
                "load_pretrained_weights: %d unexpected keys (old heads, "
                "skipped): %s",
                len(unexpected), unexpected[:5],
            )
        log.info(
            "Loaded weights from %s  (missing=%d  unexpected=%d)",
            checkpoint_path, len(missing), len(unexpected),
        )
        return self


def _remap_state_dict(state: dict) -> dict:
    """
    Translate legacy checkpoint key names and strip size-mismatched tensors
    before load_state_dict, so both strict=True and strict=False work cleanly.

    CKPT-RT3 — key renames (v4.5.3→v4.5.11):
      task_heads.k_measured_log.* → task_heads.k_total_log.*
      task_heads.k_measured.*     → task_heads.k_total.*

    CKPT-RT4 — size-mismatch strip (BUG-5 vocab expansion):
      proc_encoder.embeddings.anneal_ambient.weight  [5,8] → [8,8]
      proc_encoder.embeddings.precursor_type.weight  [6,8] → [12,8]
      proc_encoder.embeddings.oxidant_type.weight    [5,8] → [8,8]

      These three keys exist in both the checkpoint AND the model but with
      DIFFERENT shapes — strict=False cannot handle size mismatches and raises
      RuntimeError. Stripping them causes load_state_dict to treat them as
      MISSING (random-reinitialised), which is correct because:
        - Tier 2 trains with avail_flag=0 for all rows (no ALD process data)
        - These embedding tables received ZERO gradient during Tier 2 training
        - Their checkpoint values are == random init == no information to preserve
        - Tier 3 trains them from scratch with real ALD data regardless

      The prefix set covers any future additional categorical embedding that
      might also expand — not just the three known ones.
    """
    # ── Step 1: rename legacy task-head keys ─────────────────────────────────
    KEY_REMAP = {
        "task_heads.k_measured_log.": "task_heads.k_total_log.",
        "task_heads.k_measured.":     "task_heads.k_total.",
    }
    # ── Step 2: strip proc_encoder categorical embedding size mismatches ──────
    # These keys are present in both old and new checkpoints but with different
    # row counts after BUG-5 vocab expansion. Stripping → missing → random init.
    STRIP_PREFIXES = (
        "proc_encoder.embeddings.anneal_ambient.",
        "proc_encoder.embeddings.precursor_type.",
        "proc_encoder.embeddings.oxidant_type.",
    )

    remapped  = {}
    n_renamed = 0
    n_stripped = 0
    for key, val in state.items():
        # Strip size-mismatched embedding keys first
        if key.startswith(STRIP_PREFIXES):
            n_stripped += 1
            continue   # omit from output → treated as missing → random init

        # Rename legacy task-head keys
        new_key = key
        for old_prefix, new_prefix in KEY_REMAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                n_renamed += 1
                break
        remapped[new_key] = val

    if n_renamed:
        log.info(
            "_remap_state_dict: renamed %d legacy task-head keys "
            "(k_measured[_log] → k_total[_log])",
            n_renamed,
        )
    if n_stripped:
        log.info(
            "_remap_state_dict: stripped %d proc_encoder embedding keys "
            "with vocab-size mismatch (BUG-5 expansion). "
            "These tables will be randomly re-initialised — safe because "
            "Tier 2 never trains these (avail_flag=0 for all T2 rows).",
            n_stripped,
        )
    return remapped


# ==============================================================================
# SECTION 6 -- MULTI-TASK MASKED LOSS  (was Section 5)
# ==============================================================================

class MaskedMultiTaskLoss(nn.Module):
    """
    Multi-task MSE loss with:
    1. Masking for missing targets (NaN → excluded from loss)
    2. Per-task loss weighting
    3. High-k upweighting: entries above per-task threshold get 5× weight

    HIGH_K_THRESHOLDS covers k-column variants across all tiers:
      "k_total"     → linear scale: threshold = 35.0  (Tier 1/2/3)
      "k_total_log" → log scale:    threshold = log(35) ≈ 3.555  (Tier 2/3)

    Review fix #2: previous code had `tgt_m > 35.0` for "k_total_log",
    which is never True in log space (values range 1.36–5.96 for k=3.9–386).
    UNIFY-K: "k_measured": 35.0 removed — k_total:35.0 covers all tiers.
    """

    HIGH_K_THRESHOLDS = {
        "k_total":     35.0,
        "k_total_log": math.log(35.0),   # ≈ 3.555  (log scale)
    }
    HIGH_K_MULTIPLIER = 5.0

    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        upweight_high_k: bool = True,
    ):
        super().__init__()
        self.task_weights = task_weights or {
            # UNIFY-K: "k_measured" and "k_measured_log" entries removed.
            # k_total:2.0 and k_total_log:2.0 below now cover all three tiers.
            # Runtime lookup falls back to default 1.0 if a head name is absent,
            # but "k_total_log" is the canonical Tier 3 target from v4.5.11.
            "k_total":                   2.0,   # Tier 1/2/3 dielectric (linear scale)
            "k_total_log":               2.0,   # Tier 2/3 primary (log scale)
            "band_gap":                  1.0,
            "formation_energy_per_atom": 1.0,
            "e_above_hull":              0.5,
            # FIX-T3-3: TIER3_TRAIN_CONFIG names aux heads by their dataframe
            # column names ("J_g_A_cm2", "E_BD_MV_cm") which never matched
            # the old alias keys ("J_g_log", "E_BD") → both silently fell
            # through to the default weight of 1.0.  Both the canonical column
            # names and the old aliases are retained so any tier config using
            # either naming convention receives the correct weight.
            "J_g_A_cm2":                1.5,   # Tier 3 canonical (column name)
            "E_BD_MV_cm":               1.0,   # Tier 3 canonical (column name)
            "J_g_log":                  1.5,   # legacy alias (kept for compat)
            "E_BD":                     1.0,   # legacy alias (kept for compat)
        }
        self.upweight_high_k = upweight_high_k
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions:        Dict[str, torch.Tensor],
        targets:            Dict[str, torch.Tensor],
        functional_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        functional_weights: float Tensor [B] of per-sample loss multipliers
        Accumulates task losses via torch.stack+mean to avoid disconnected
        tensor when all targets are NaN.
        """

        device=next(iter(predictions.values())).device
        task_losses = []

        for task, pred in predictions.items():
            if task not in targets:
                continue

            tgt  = targets[task]
            mask = ~torch.isnan(tgt)

            if mask.sum() == 0:
                continue

            pred_m = pred[mask].squeeze()
            tgt_m  = tgt[mask].squeeze()

            # Per-sample MSE
            per_sample_loss = self.mse(pred_m, tgt_m)

            # High-k upweighting: per-task thresholds (linear vs log scale)
            # k_total_log threshold = log(35) ≈ 3.555; linear tasks use 35.0
            if task in self.HIGH_K_THRESHOLDS and self.upweight_high_k:
                threshold      = self.HIGH_K_THRESHOLDS[task]
                high_k_mask    = tgt_m > threshold
                hw             = torch.ones_like(per_sample_loss)
                hw[high_k_mask] = self.HIGH_K_MULTIPLIER
                per_sample_loss = per_sample_loss * hw

            # Functional-weighted loss: apply after NaN mask so indexing aligns
            if functional_weights is not None:
                mask_1d = mask.view(-1) if mask.ndim > 1 else mask
                fw_m = functional_weights[mask_1d].squeeze()
                per_sample_loss = per_sample_loss * fw_m

            task_loss   = per_sample_loss.mean()
            task_losses.append(self.task_weights.get(task, 1.0) * task_loss)

        if task_losses:
            return torch.stack(task_losses).mean()
        else:
            # No active tasks - return a zero tensor on the correct device.
            # using .zero_() on a prediction tensor creates a valid leaf
            # tensor that is a proper result of a troch operation (zero_),
            # so backward() works without "does not require grad" error.
            return next(iter(predictions.values())).new_zeros(())


# ==============================================================================
# SECTION 7 -- TRAINING ENGINE  (was Section 6)
# ==============================================================================

class ALIGNNTrainer:
    """
    Training engine for a single tier.
    Handles: optimizer setup, lr scheduling, checkpoint saving,
    early stopping, and metric logging.
    """

    def __init__(
        self,
        model:           HighKALIGNN,
        tier_cfg:        dict,
        device:          str  = None,
        ckpt_prefix:     str  = "tier",
        ablate_context:  bool = False,
    ):
        # Device: use DDP-assigned device when active, else CUDA/CPU
        if device is None:
            device = _DIST["device"] if _DIST["active"] \
                     else ("cuda" if torch.cuda.is_available() else "cpu")

        self.device         = device
        self.ckpt_prefix    = ckpt_prefix
        self.ablate_context = ablate_context

        model = model.to(device)

        # DDP wrapping — each rank runs its own forward; AllReduce syncs grads.
        # find_unused_parameters=True is required because ProcessParamsEncoder
        # and StackContextEncoder are bypassed when proc_context/stack_context
        # are None (Tier 1/2 have no process data).  Without this flag DDP
        # waits forever for a gradient reduction on those parameters that never
        # fires, raising "Expected to have finished reduction in the prior
        # iteration before starting a new one."
        # model_core keeps a reference to the raw HighKALIGNN so we can call
        # methods like freeze_alignn_layers / fit_encoder_stats directly.
        if _DIST["active"]:
            self.model      = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids             = [_DIST["rank"]],
                output_device          = _DIST["rank"],
                find_unused_parameters = True,    # proc/stack encoders unused in Tier 1/2
                gradient_as_bucket_view= True,    # ~33% peak VRAM reduction, no correctness cost
            )
            self.model_core = self.model.module   # raw HighKALIGNN
        else:
            self.model      = model
            self.model_core = model

        self.cfg = tier_cfg

        # Optimizer — parameters from model_core (always the raw module)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model_core.parameters()),
            lr           = tier_cfg["learning_rate"],
            weight_decay = tier_cfg["weight_decay"],
        )

        # Loss
        self.criterion = MaskedMultiTaskLoss(upweight_high_k=True)

        # Best metric tracking
        self.best_val_mae  = float("inf")
        self.best_epoch    = 0
        self.patience      = int(tier_cfg.get("patience", 30))   # per-tier config
        self.patience_ctr  = 0
        log.info("Trainer patience: %d epochs", self.patience)

    def build_scheduler(self, n_steps_per_epoch: int, n_epochs: int):
        """
        One-cycle policy (paper section Methods) for pretrain.
        Cosine annealing for fine-tune stages.
        """
        sched_type = self.cfg.get("scheduler", "onecycle")
        if sched_type == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr        = self.cfg["learning_rate"],
                total_steps   = n_steps_per_epoch * n_epochs,
                pct_start     = self.cfg.get("pct_start", 0.3),
            )
        else:  # cosine
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = n_epochs,
                eta_min = self.cfg["learning_rate"] * 0.01,
            )

    def train_epoch(
        self,
        loader: DataLoader,
        scheduler,
        target_col: str,
    ) -> dict:
        """
        Run one training epoch.

        Returns a dict with:
          loss             -- mean training loss for this epoch
          proc_avail_pct   -- % of samples where process params were present
          stack_avail_pct  -- % of samples where stack context was present
        These are logged per-epoch alongside alpha/beta (Obs 2).
        """
        self.model.train()
        total_loss        = 0.0
        n_batches         = 0
        proc_avail_total  = 0.0
        stack_avail_total = 0.0
        n_samples         = 0

        for batch in loader:
            if batch is None:
                continue

            graph      = batch["graph"].to(self.device)
            line_graph = batch["line_graph"].to(self.device)
            target     = batch["target"].to(self.device)

            # v2.2: move context dicts to device
            proc_ctx  = batch.get("proc_context")
            stack_ctx = batch.get("stack_context")
            if proc_ctx is not None:
                proc_ctx  = {k: v.to(self.device) for k, v in proc_ctx.items()}
                stack_ctx = {k: v.to(self.device) for k, v in stack_ctx.items()}
                # Obs 2: accumulate availability before possible ablation
                proc_avail_total  += proc_ctx["avail"].sum().item()
                stack_avail_total += stack_ctx["avail"].sum().item()
                n_samples         += proc_ctx["avail"].numel()

            # Obs 3: ablation flag forces context branches to zero contribution
            if self.ablate_context:
                proc_ctx = stack_ctx = None

            # v4 #11: extract functional_code for model conditioning + loss weighting
            func_code = batch.get("functional_code")
            if func_code is not None:
                func_code = func_code.to(self.device)

            # Build per-sample functional loss weights from config
            func_loss_weights = None
            fw_map = self.cfg.get("functional_loss_weights", {})
            if fw_map and func_code is not None:
                func_loss_weights = torch.ones(
                    func_code.shape[0], dtype=torch.float32, device=self.device
                )
                for code, w in fw_map.items():
                    func_loss_weights[func_code == int(code)] = float(w)

            # FIX-T3-9B: experimental row upweighting.
            # Multiplies per-sample loss weight for experimental (ALD-measured)
            # rows by exp_sample_weight (default 3.0).  Combined with functional
            # weights so both corrections apply simultaneously.
            exp_w = float(self.cfg.get("exp_sample_weight", 1.0))
            if exp_w != 1.0:
                is_exp = batch.get("is_experimental")
                if is_exp is not None:
                    is_exp = is_exp.to(self.device).squeeze(-1)   # [B]
                    exp_scale = 1.0 + (exp_w - 1.0) * is_exp     # 1.0 for DFT, exp_w for experimental
                    func_loss_weights = (
                        exp_scale if func_loss_weights is None
                        else func_loss_weights * exp_scale
                    )

            self.optimizer.zero_grad()

            # Forward through DDP wrapper (task="__all__" returns dict).
            # When DDP is active each rank processes its own shard; AllReduce
            # synchronises gradients during backward() automatically.
            preds = self.model(
                graph, line_graph,
                task            = "__all__",
                proc_context    = proc_ctx,
                stack_context   = stack_ctx,
                functional_code = func_code,
            )

            # ----------------------------------------------------------------
            # Build targets dict for MaskedMultiTaskLoss.
            # FIX-OBS1: route each aux head to its batch tensor so all heads
            # receive gradient on every batch where their target is non-NaN.
            # ----------------------------------------------------------------
            targets_dict = {target_col: target}
            aux_batch = batch.get("aux_targets", {})
            for task_name in preds.keys():
                if task_name == target_col:
                    continue
                if task_name in aux_batch:
                    targets_dict[task_name] = aux_batch[task_name].to(self.device)
                else:
                    col_name = TASK_TO_COLUMN.get(task_name, task_name)
                    if col_name in aux_batch:
                        targets_dict[task_name] = aux_batch[col_name].to(self.device)

            loss = self.criterion(preds, targets_dict, functional_weights=func_loss_weights)
            loss.backward()

            # Gradient clipping -- max_norm from config (default 1.0)
            max_norm = self.cfg.get("max_grad_norm", 1.0)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

            self.optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        return {
            "loss":            total_loss / max(n_batches, 1),
            "proc_avail_pct":  100.0 * proc_avail_total  / max(n_samples, 1),
            "stack_avail_pct": 100.0 * stack_avail_total / max(n_samples, 1),
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader:       DataLoader,
        target_col:   str,
        return_preds: bool = False,
    ):
        """
        Evaluate MAE and RMSE on a validation/test split.

        return_preds=True: also return (preds_tensor, targets_tensor) for
        post-hoc exp-space MAE computation on log-transformed targets.
        (Review fix #1: exact exp-space MAE requires stored predictions.)

        Review fix #5: functional_code is now passed to model.forward() so
        evaluation uses the same conditioning as training.  Previously the
        model received functional_code=None during validation/test, creating
        a train/eval inconsistency.
        """
        self.model.eval()
        preds_all   = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                graph      = batch["graph"].to(self.device)
                line_graph = batch["line_graph"].to(self.device)
                target     = batch["target"]

                proc_ctx  = batch.get("proc_context")
                stack_ctx = batch.get("stack_context")
                if proc_ctx is not None:
                    proc_ctx  = {k: v.to(self.device) for k, v in proc_ctx.items()}
                    stack_ctx = {k: v.to(self.device) for k, v in stack_ctx.items()}
                if self.ablate_context:
                    proc_ctx = stack_ctx = None

                # Review fix #5: wire functional_code through evaluate()
                func_code = batch.get("functional_code")
                if func_code is not None:
                    func_code = func_code.to(self.device)

                pred = self.model(
                    graph, line_graph, task=target_col,
                    proc_context    = proc_ctx,
                    stack_context   = stack_ctx,
                    functional_code = func_code,
                )
                preds_all.append(pred.cpu())
                targets_all.append(target)

        if not preds_all:
            empty = torch.tensor([])
            return (float("inf"), float("inf"), empty, empty) if return_preds \
                   else (float("inf"), float("inf"))

        preds   = torch.cat(preds_all).squeeze()
        targets = torch.cat(targets_all).squeeze()
        valid   = ~torch.isnan(targets)
        mae     = (preds[valid] - targets[valid]).abs().mean().item()
        rmse    = ((preds[valid] - targets[valid]) ** 2).mean().sqrt().item()

        if return_preds:
            return mae, rmse, preds, targets
        return mae, rmse

    @torch.no_grad()
    def evaluate_multitask(
        self,
        loader:   DataLoader,
        tier_cfg: dict,
        df_full:  Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        FIX2 + MAD:MAE: Evaluate ALL task heads on a data split.
        Returns per-task: mae, rmse, mad, mad_mae_ratio, n, coverage_pct.

        MAD is computed from df_full (full tier dataframe) when provided --
        this matches the paper's reporting and gives accurate MAD:MAE ratios
        comparable to Table 2 and Table 3. Falls back to test-split MAD
        when df_full is not provided (still representative due to stratification).
        """
        self.model.eval()

        primary_col = tier_cfg["target"]
        aux_cols    = tier_cfg.get("aux_targets", [])

        # Build task-head -> batch-key routing map
        task_to_batch_key = {primary_col: "__primary__"}

        task_preds   = {h: [] for h in self.model_core.task_heads}
        task_targets = {h: [] for h in self.model_core.task_heads}
        n_batches    = 0

        for batch in loader:
            if batch is None:
                continue
            graph      = batch["graph"].to(self.device)
            line_graph = batch["line_graph"].to(self.device)

            # v2.2: context branches
            proc_ctx  = batch.get("proc_context")
            stack_ctx = batch.get("stack_context")
            if proc_ctx is not None:
                proc_ctx  = {k: v.to(self.device) for k, v in proc_ctx.items()}
                stack_ctx = {k: v.to(self.device) for k, v in stack_ctx.items()}
            if self.ablate_context:
                proc_ctx = stack_ctx = None

            # Review fix #5 (propagated to multitask path): evaluate_multitask
            # previously called forward_all_tasks WITHOUT functional_code, so
            # _apply_func_conditioning() received None and the per-functional
            # offset (PBE vs OptB88vdW, etc.) the model was TRAINED with was
            # dropped at eval. That train/eval mismatch inflated every head's
            # MAE in the "ALL TASK HEADS" table (e.g. formation_energy showed
            # ~0.32 here vs ~0.037 from the single-task evaluate()). Mirror the
            # extraction already done in evaluate() so the table reflects the
            # same conditioning regime as training.
            func_code = batch.get("functional_code")
            if func_code is not None:
                func_code = func_code.to(self.device)

            all_preds  = self.model_core.forward_all_tasks(
                graph, line_graph,
                proc_context=proc_ctx,
                stack_context=stack_ctx,
                functional_code=func_code,
            )
            n_batches += 1

            for head_name, pred_t in all_preds.items():
                task_preds[head_name].append(pred_t.cpu())
                # Look up targets from df_full using row_indices (positional indices)
                if df_full is not None and "row_indices" in batch:
                    row_idxs = batch["row_indices"].tolist()
                    tgt_list = []
                    for ri in row_idxs:
                        try:
                            row_data = df_full.iloc[ri]
                            col = TASK_TO_COLUMN.get(head_name, head_name)
                            val = row_data.get(col)
                            if val is not None and not (
                                isinstance(val, float) and np.isnan(val)
                            ):
                                tgt_list.append(float(val))
                            else:
                                tgt_list.append(float("nan"))
                        except (KeyError, IndexError):
                            tgt_list.append(float("nan"))
                    tgt = torch.tensor(tgt_list, dtype=torch.float32)
                else:
                    tgt = torch.full((len(batch["row_indices"]),), float("nan"), dtype=torch.float32)
                task_targets[head_name].append(tgt)

        if n_batches == 0:
            return {}

        # Pre-compute full-dataset MAD per task when df_full provided
        full_mads: Dict[str, float] = {}
        if df_full is not None:
            for head_name, col_name in TASK_TO_COLUMN.items():
                if col_name in df_full.columns:
                    vals = df_full[col_name].dropna()
                    if len(vals) > 10:
                        full_mads[head_name] = float(
                            (vals - vals.mean()).abs().mean()
                        )
            # Also handle primary target directly
            if primary_col in df_full.columns:
                vals = df_full[primary_col].dropna()
                if len(vals) > 10:
                    full_mads[primary_col] = float(
                        (vals - vals.mean()).abs().mean()
                    )

        total_rows = len(torch.cat(task_targets[primary_col]).squeeze())
        results    = {}

        for head_name in self.model_core.task_heads:
            if not task_preds[head_name]:
                results[head_name] = {
                    "mae": float("nan"), "rmse": float("nan"),
                    "mad": float("nan"), "mad_mae_ratio": float("nan"),
                    "n": 0, "coverage_pct": 0.0,
                }
                continue

            preds   = torch.cat(task_preds[head_name]).squeeze()
            targets = torch.cat(task_targets[head_name]).squeeze()
            valid   = ~torch.isnan(targets)
            n_valid = int(valid.sum().item())

            if n_valid == 0:
                results[head_name] = {
                    "mae": float("nan"), "rmse": float("nan"),
                    "mad": float("nan"), "mad_mae_ratio": float("nan"),
                    "n": 0, "coverage_pct": 0.0,
                }
                continue

            mae  = (preds[valid] - targets[valid]).abs().mean().item()
            rmse = ((preds[valid] - targets[valid]) ** 2).mean().sqrt().item()

            # MAD: prefer full-dataset value for accuracy matching paper reporting
            if head_name in full_mads:
                mad        = full_mads[head_name]
                mad_source = "full dataset"
            else:
                tgt_valid  = targets[valid]
                mad        = float((tgt_valid - tgt_valid.mean()).abs().mean().item())
                mad_source = "test split"

            results[head_name] = {
                "mae":           mae,
                "rmse":          rmse,
                "mad":           mad,
                "mad_mae_ratio": mad / max(mae, 1e-9),
                "mad_source":    mad_source,
                "n":             n_valid,
                "coverage_pct":  100.0 * n_valid / max(total_rows, 1),
            }

        return results

    def print_multitask_results(
        self,
        results:    Dict[str, Dict[str, float]],
        split_name: str = "TEST",
        tier_name:  str = "",
    ):
        """
        FIX2 + MAD:MAE: Print all task heads in a formatted table.
        Columns: MAE | RMSE | MAD:MAE | Paper ref | N valid | Coverage
        Paper benchmarks from PAPER_MAD_MAE_BENCHMARK.
        v = below paper reference   ^ = beating paper reference

        Example output:
        ===================================================================
         TIER 1 -- TEST SET -- ALL TASK HEADS
        ===================================================================
          Task               |    MAE    |   RMSE    | MAD:MAE | Paper ref |  N valid | Coverage
          formation_energy   |  0.0329   |  0.0601   |   26.14 |   26.06   |   44,577 | 100.0%  <- primary
          band_gap           |  0.1423   |  0.2234   |    6.95 |    7.07 ^ |   38,421 |  86.2%
          k_measured         | 18.3214   | 24.5621   |    1.51 |    1.63 v |   12,345 |  27.7%
          J_g_log            |     --     |     --     |     --   |     --     |        0 |   0.0%  (no data)
          E_BD               |     --     |     --     |     --   |     --     |        0 |   0.0%  (no data)
        ===================================================================
        """
        hdr  = f" {tier_name} -- {split_name} SET -- ALL TASK HEADS"
        line = "=" * max(76, len(hdr) + 2)
        log.info("\n%s", line)
        log.info(hdr)
        log.info("%s", line)
        log.info(
            "  %-20s | %9s | %9s | %7s | %9s | %9s | %8s",
            "Task", "MAE", "RMSE", "MAD:MAE", "Paper ref", "N valid", "Coverage"
        )
        log.info("  %s", "-" * 80)

        primary_col = self.cfg.get("target", "")

        for task_name, m in results.items():
            n      = m.get("n", 0)
            cov    = m.get("coverage_pct", 0.0)
            is_pri = "<- primary" if task_name == primary_col else ""
            note   = "(no data this tier)" if n == 0 else is_pri

            if n > 0 and not np.isnan(m.get("mae", float("nan"))):
                mae_s   = f"{m['mae']:9.4f}"
                rmse_s  = f"{m['rmse']:9.4f}"
                ratio_s = f"{m['mad_mae_ratio']:7.2f}"
            else:
                mae_s   = "    --    "
                rmse_s  = "    --    "
                ratio_s = "   --   "

            paper_ref = PAPER_MAD_MAE_BENCHMARK.get(task_name)
            if paper_ref is not None and n > 0 and not np.isnan(m.get("mad_mae_ratio", float("nan"))):
                gap_pct = 100.0 * (paper_ref - m["mad_mae_ratio"]) / paper_ref
                flag    = " v" if gap_pct > 5 else (" ^" if gap_pct < -5 else "  ")
                paper_s = f"{paper_ref:7.2f}{flag}"
            elif paper_ref is not None:
                paper_s = f"{paper_ref:7.2f}  "
            else:
                paper_s = "   --     "

            log.info(
                "  %-20s | %s | %s | %s | %s | %9s | %7.1f%%  %s",
                task_name, mae_s, rmse_s, ratio_s, paper_s,
                f"{n:,}", cov, note,
            )

        log.info("%s", line)
        log.info(
            "  MAD:MAE > 5 = good predictive model (paper standard)  "
            "v = below paper  ^ = beating paper"
        )
        mad_src = next(
            (m.get("mad_source","test split") for m in results.values() if m.get("n",0) > 0),
            "N/A"
        )
        log.info("  MAD computed from: %s", mad_src)
        log.info("%s\n", line)

    def save_checkpoint(self, epoch: int, val_mae: float, tag: str = "best"):
        path     = CKPT_ROOT / f"{self.ckpt_prefix}_{tag}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        # Atomic write: save to .tmp then rename.
        # Prevents corrupted reads if the process is interrupted mid-write,
        # and prevents rank-1 from reading a partially-written file during DDP.
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.model_core.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "val_mae":          val_mae,
            "config":           self.cfg,
        }, tmp_path)
        tmp_path.replace(path)   # atomic on POSIX; near-atomic on Windows
        log.info("Checkpoint saved -> %s  (epoch=%d, val_mae=%.4f)",
                 path, epoch, val_mae)

    def train(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        target_col:    str,
        train_sampler = None,   # DistributedSampler; must call set_epoch each epoch
    ):
        """Full training loop with early stopping and checkpoint saving."""
        n_epochs   = self.cfg["epochs"]
        scheduler  = self.build_scheduler(len(train_loader), n_epochs)
        history    = []

        log.info("Starting training: %d epochs, target='%s', device=%s",
                 n_epochs, target_col, self.device)

        for epoch in range(1, n_epochs + 1):
            # DDP: reshuffle shard assignments each epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t_start    = time.time()
            epoch_stats = self.train_epoch(train_loader, scheduler, target_col)
            train_loss  = epoch_stats["loss"]
            val_mae, val_rmse = self.evaluate(val_loader, target_col)

            if not isinstance(scheduler,
                              torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Track best — only rank-0 saves checkpoints to avoid file collisions
            improved = val_mae < self.best_val_mae
            if improved:
                self.best_val_mae = val_mae
                self.best_epoch   = epoch
                self.patience_ctr = 0
                if is_rank0():
                    self.save_checkpoint(epoch, val_mae, tag="best")
            else:
                self.patience_ctr += 1

            elapsed = time.time() - t_start

            # Obs 2: log alpha, beta, proc/stack availability every epoch
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_MAE=%.4f  val_RMSE=%.4f  "
                "best=%.4f (ep%d)  %.1fs  "
                "alpha=%.4f beta=%.4f  proc=%.1f%% stack=%.1f%%  %s",
                epoch, n_epochs, train_loss, val_mae, val_rmse,
                self.best_val_mae, self.best_epoch, elapsed,
                self.model_core.alpha.item(), self.model_core.beta.item(),
                epoch_stats["proc_avail_pct"], epoch_stats["stack_avail_pct"],
                "✓" if improved else "",
            )

            history.append({
                "epoch":           epoch,
                "train_loss":      train_loss,
                "val_mae":         val_mae,
                "val_rmse":        val_rmse,
                "alpha":           self.model_core.alpha.item(),
                "beta":            self.model_core.beta.item(),
                "proc_avail_pct":  epoch_stats["proc_avail_pct"],
                "stack_avail_pct": epoch_stats["stack_avail_pct"],
            })

            # Optional epoch-1 checkpoint for Tier-3 diagnostics.
            # This lets us verify whether the best validation result is simply
            # the loaded Tier-2 baseline after one update, or whether later
            # Tier-3 training genuinely improves the model.
            if epoch == 1 and self.cfg.get("save_epoch1_checkpoint", False) and is_rank0():
                self.save_checkpoint(epoch, val_mae, tag="epoch1")

            # Safety checkpoint every 50 epochs — rank-0 only
            if epoch % 50 == 0 and is_rank0():
                self.save_checkpoint(epoch, val_mae, tag=f"ep{epoch}")

            # -- Early stopping guard --------------------------------------
            # early_stopping=False: run full epochs (ALIGNN paper behaviour).
            # min_epochs: don't stop even if patience exhausted before this.
            use_es   = self.cfg.get("early_stopping", True)
            min_ep   = self.cfg.get("min_epochs", 0)
            if use_es and epoch >= min_ep and self.patience_ctr >= self.patience:
                log.info(
                    "Early stopping at epoch %d "
                    "(no improvement for %d epochs, min_epochs=%d)",
                    epoch, self.patience, min_ep,
                )
                break

        # Save training history
        hist_path = REPORT_ROOT / f"{self.ckpt_prefix}_training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        log.info(
            "Training complete. Best val_MAE=%.4f at epoch %d.",
            self.best_val_mae, self.best_epoch
        )
        return history


# ==============================================================================
# SECTION 8 -- FULL THREE-TIER PIPELINE  (was Section 7)
# ==============================================================================

class BalancedDistributedSampler(torch.utils.data.Sampler):
    """
    Distributed Sampler that guraantees balances proc_avail (DFT vs experimental)
    across ranks every epoch via round-robin grouping.
    """

    def __init__(
            self,
            dataset,
            proc_avail: list[int],
            num_replicas: int = 2,
            rank: int =0,
            seed: int = 42,
            shuffle: bool = True,
        ):
            self.dataset = dataset
            self.proc_avail = proc_avail
            self.num_replicas = num_replicas
            self.rank = rank
            self.seed = seed
            self.shuffle = shuffle
            self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))

        # Group by proc_avail
        groups: dict[int, list[int]] = {}
        for idx, pa in zip(indices, self.proc_avail):
            groups.setdefault(pa, []).append(idx)

        # Shuffle within each group
        if self.shuffle:
            for pa in groups:
                rng.shuffle(groups[pa])

        # Round-robin through groups
        group_keys = sorted(groups.keys())
        group_iters = {k: iter(v) for k, v in groups.items()}
        balanced: list[int] = []
        while True:
            added = False
            for gk in group_keys:
                try:
                    balanced.append(next(group_iters[gk]))
                    added = True
                except StopIteration:
                    pass
            if not added:
                break

        # Split by rank (even->rank0, odd->rank1, ...)
        # keep only items for this rank
        rank_indices = [balanced[i] for i in range(self.rank, len(balanced), self.num_replicas)]
        return iter(rank_indices)

    def __len__(self):
        return len(self.proc_avail) // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

def build_dataloader(
    df: pd.DataFrame,
    target_col: str,
    aux_cols:   List[str],
    train_frac: float,
    val_frac:   float,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, BalancedDistributedSampler, HighKGraphDataset]:
    """Build train/val/test DataLoaders with stratified splitting.

    Returns: (train_loader, val_loader, test_loader, train_sampler, dataset).
    The dataset is returned for correct traget lookup in evaluate_multitask.
    """
    dataset = HighKGraphDataset(df, target_col=target_col, aux_cols=aux_cols)

    train_ds, val_ds, test_ds = get_stratified_split(
        dataset, train_frac=train_frac, val_frac=val_frac
    )

    collate = HighKGraphDataset.collate_fn

    # DDP: BalancedDistributedSampler guarantees balanced proc_avail across ranks.
    if _DIST["active"]:
        train_indices = list(range(len(train_ds)))
        try:
            proc_avail = [
                int(dataset.df.loc[dataset.valid_idx[i], "source"] == "Experimental")
                for i in train_indices
            ]
        except Exception:
            proc_avail = [0] * len(train_indices)

        train_sampler = BalancedDistributedSampler(
            train_ds,
            proc_avail   =  proc_avail,
            num_replicas = _DIST["world"],
            rank         = _DIST["rank"],
            shuffle      = True,
        )
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag  = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=shuffle_flag, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, train_sampler, dataset


def safe_load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """
    Load a checkpoint with integrity verification and clear error reporting.

    Guards against three failure modes that cause "read error" on tier2 test:
      1. File not written yet  (rank-1 loads before rank-0 finishes saving)
         → caller should call dist_barrier() first; this adds a size sanity check
      2. Truncated write       (process killed during torch.save before rename)
         → save_checkpoint now writes atomically via .tmp → rename, so this
           should not happen; guard retained as belt-and-suspenders
      3. Corrupted pickle      (disk error, NFS issue)
         → caught by the except block; returns None so caller can skip eval

    Always call dist_barrier() before this function in DDP mode so rank 0
    has fully flushed the file before any rank loads it.
    """
    if not path.exists():
        log.error("Checkpoint not found: %s", path)
        return None
    # Size sanity: an empty or tiny file is definitely corrupt
    size_bytes = path.stat().st_size
    if size_bytes < 1024:
        log.error("Checkpoint appears corrupt (size=%d B): %s", size_bytes, path)
        return None
    try:
        ckpt = torch.load(path, map_location=map_location)
        if "model_state_dict" not in ckpt:
            raise KeyError("model_state_dict key missing")
        log.info("Checkpoint loaded: %s  (epoch=%s  val_mae=%.4f  size=%.1f MB)",
                 path, ckpt.get("epoch", "?"), ckpt.get("val_mae", float("nan")),
                 size_bytes / 1e6)
        return ckpt
    except Exception as exc:
        log.error("Failed to load checkpoint %s: %s", path, exc)
        return None


def run_tier1_pretrain(df_tier1: pd.DataFrame, ablate_context: bool = False):
    """
    Tier 1 -- Foundation pretraining on full JARVIS-DFT + MP + QM9.

    Primary target: formation_energy_per_atom (most data, teaches oxide physics)
    Aux targets:    band_gap, k_measured (where available)

    After 300 epochs the model has learned:
    - General oxide bonding geometry -> atom embeddings
    - Formation energy as function of crystal structure
    - Band gap sensitivity to bond angles (critical for HfO2 phase discrimination)
    - Polarisability correlates with dielectric response (from QM9 alpha target)
    """
    log.info("=" * 70)
    log.info(" TIER 1 -- Foundation Pretrain")
    log.info(" Rows: %d   Target: formation_energy_per_atom", len(df_tier1))
    log.info("=" * 70)

    cfg = TIER1_TRAIN_CONFIG

    # -- Log-transform aux k_total for Tier 1/Tier 2 consistency -----------
    # TIER1_TRAIN_CONFIG.log_transform_aux maps derived column → source column.
    # e.g. {"k_total_log": "k_total"}: creates k_total_log = log(k_total)
    # for rows that have k_total.  Rows without k_total keep NaN (masked in loss).
    #
    # WHY THIS MATTERS FOR TRANSFER LEARNING:
    # If Tier 1 trains the k_total head on linear k and Tier 2 loads those weights
    # then trains on log(k_total), the head must "unlearn" the linear scale before
    # learning log scale — wasting 20-30 Tier 2 epochs.  Training Tier 1 on
    # k_total_log ensures the head weight scale is consistent at the hand-off.
    for log_col, src_col in cfg.get("log_transform_aux", {}).items():
        if src_col in df_tier1.columns:
            df_tier1 = df_tier1.copy()
            # pd.to_numeric: handles object-dtype columns from stale HDF5 cache
            # (k_total stored as dtype('O') → must convert before np.log)
            src_numeric = pd.to_numeric(df_tier1[src_col], errors="coerce")
            mask = src_numeric.notna()
            df_tier1[log_col] = np.nan
            df_tier1.loc[mask, log_col] = np.log(
                src_numeric[mask].clip(lower=0.1)
            )
            n_transformed = int(mask.sum())
            log.info(
                "Tier 1 aux log transform: %s → %s  "
                "(%d non-null rows, range=[%.3f, %.3f])",
                src_col, log_col, n_transformed,
                df_tier1[log_col].dropna().min(),
                df_tier1[log_col].dropna().max(),
            )

    task_names = [cfg["target"]] + cfg["aux_targets"]
    log.info("Tier 1 training with targets: %s", task_names)

    train_loader, val_loader, test_loader, train_sampler, _main_ds_t1 = build_dataloader(
        df          = df_tier1,
        target_col  = cfg["target"],
        aux_cols    = cfg["aux_targets"],
        train_frac  = cfg["train_ratio"],
        val_frac    = cfg["val_ratio"],
        batch_size  = cfg["batch_size"],
    )

    model   = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names)
    # v2.2: fit encoder normalisation stats from training data.
    # For Tier 1 no proc/stack columns exist → stats stay mean=0, std=1 (correct).
    model.fit_encoder_stats(df_tier1)
    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix="tier1",
                            ablate_context=ablate_context)
    history = trainer.train(train_loader, val_loader, target_col=cfg["target"], train_sampler=train_sampler)

    # FIX2+MAD:MAE: load best checkpoint then evaluate all task heads
    best_ckpt = CKPT_ROOT / "tier1_best.pt"
    dist_barrier()
    ckpt = safe_load_checkpoint(best_ckpt)
    if ckpt is not None:
        trainer.model_core.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        log.warning("Could not load best checkpoint; evaluating with final weights.")

    test_mae, test_rmse = trainer.evaluate(test_loader, cfg["target"])
    log.info("Tier 1 TEST  MAE=%.4f  RMSE=%.4f  (target: %s)",
             test_mae, test_rmse, cfg["target"])

    mt_results = trainer.evaluate_multitask(test_loader, cfg, df_full=_main_ds_t1.df)  # FIX-2: use dataset.df (reset_index applied) not raw df_tier1
    trainer.print_multitask_results(mt_results, split_name="TEST", tier_name="TIER 1")

    out_path = REPORT_ROOT / "tier1_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"primary_mae": test_mae, "primary_rmse": test_rmse,
                   "multitask": mt_results}, f, indent=2)
    log.info("Tier 1 test results saved -> %s", out_path)

    return CKPT_ROOT / "tier1_best.pt"


def run_tier2_finetune(
    df_tier2: pd.DataFrame,
    pretrained_weights: Path,
    ablate_context: bool = False,
):
    """
    Tier 2 -- Domain fine-tuning on oxide dielectrics (k > 10).

    Loads Tier 1 pretrained weights.
    Freezes first 2 ALIGNN layers (preserve low-level geometry features).
    Trains upper layers on k prediction for oxide dielectrics.
    """
    log.info("=" * 70)
    log.info(" TIER 2 -- Domain Fine-tune (Oxide Dielectrics)")
    log.info(" Rows: %d   Target: k_total", len(df_tier2))
    log.info("=" * 70)

    cfg = TIER2_TRAIN_CONFIG

    # Only use rows with k_total for Tier 2.
    df_t2_k = df_tier2[df_tier2["k_total"].notna()].copy()
    log.info("Tier 2 rows with k_total: %d", len(df_t2_k))

    # -- Log transform (root cause of MAE plateau at 5.x) ----------------
    # k_total spans 3.9–386+ (100× range).  MSE on linear k converges to
    # predicting the mean for all inputs regardless of epochs.
    # Training on log(k_total) normalises the range to ~1.4 decades and gives
    # equal gradient weight across the full dielectric spectrum.
    if cfg.get("log_transform", False):
        orig_col  = cfg["log_original_col"]   # "k_total"
        log_col   = cfg["target"]             # "k_total_log"
        # pd.to_numeric: handles object-dtype k_total from stale HDF5 cache.
        # np.log raises "loop of ufunc does not support argument 0 of type float
        # which has no callable log method" when the Series is dtype('O'), even
        # when every value is a valid Python float.  Converting to float64 first
        # makes np.log work regardless of what dtype the HDF5 cache stored.
        k_numeric        = pd.to_numeric(df_t2_k[orig_col], errors="coerce")
        df_t2_k[log_col] = np.log(k_numeric.clip(lower=0.1))
        log.info(
            "Log transform: %s → %s  range=[%.3f, %.3f]  "
            "(original k range=[%.1f, %.1f])",
            orig_col, log_col,
            df_t2_k[log_col].min(), df_t2_k[log_col].max(),
            k_numeric.min(),        k_numeric.max(),
        )
    target_col = cfg["target"]   # "k_total_log" if log_transform else "k_total"

    train_loader, val_loader, test_loader, train_sampler, _main_ds_t2 = build_dataloader(
        df         = df_t2_k,
        target_col = target_col,
        aux_cols   = cfg["aux_targets"],
        train_frac = cfg["train_ratio"],
        val_frac   = cfg["val_ratio"],
        batch_size = cfg["batch_size"],
    )

    task_names = [target_col] + cfg["aux_targets"]
    log.info("Tier 2 training with task heads: %s", task_names)
    model = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names)
    model.load_pretrained_weights(pretrained_weights, strict=False)
    model.fit_encoder_stats(df_t2_k)
    model.unfreeze_all()   # all layers active from the start for Phase A

    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix="tier2",
                            ablate_context=ablate_context)

    # ── Phase A (v4 change #12): ALL oxide rows, band_gap as primary ───────
    # df_tier2 has ~30K rows with valid band_gap (vs only 5K with k_total).
    # Running 30 epochs on the full oxide set consolidates backbone
    # representations before Phase B specialises for k_total.
    phase_a_target = cfg.get("phase_a_target", "band_gap")
    phase_a_epochs = cfg.get("phase_a_epochs", 30)
    log.info(
        "Tier 2 Phase A: ALL oxide rows=%d  target=%s  epochs=%d",
        len(df_tier2), phase_a_target, phase_a_epochs,
    )
    pa_loader, _, _, pa_sampler,_ds = build_dataloader(
        df         = df_tier2,
        target_col = phase_a_target,
        aux_cols   = [c for c in cfg["aux_targets"] if c != phase_a_target],
        train_frac = 0.95,        # use 95% for Phase A (validation not needed)
        val_frac   = 0.05,
        batch_size = cfg["batch_size"],
    )
    cfg_pa    = {**cfg, "epochs": phase_a_epochs, "scheduler": "cosine",
                 "early_stopping": False}
    trainer.cfg = cfg_pa
    sched_pa    = trainer.build_scheduler(len(pa_loader), phase_a_epochs)
    for epoch in range(1, phase_a_epochs + 1):
        ep_stats = trainer.train_epoch(pa_loader, sched_pa, phase_a_target)
        log.info("Phase A  ep %3d/%d  loss=%.4f", epoch, phase_a_epochs,
                 ep_stats["loss"])

    # ── Phase B: k_total_log rows, lower LR, full early-stopping ───────────
    log.info(
        "Tier 2 Phase B: k_total_log rows=%d  lr=%.2e  epochs=%d",
        len(df_t2_k),
        cfg.get("learning_rate_unfreeze", cfg["learning_rate"] * 0.5),
        cfg["epochs"] - phase_a_epochs,
    )
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.get("learning_rate_unfreeze",
                                cfg["learning_rate"] * 0.5),
        weight_decay = cfg["weight_decay"],
    )
    remaining_epochs = cfg["epochs"] - phase_a_epochs
    cfg_pb           = {**cfg, "epochs": remaining_epochs}
    trainer.cfg      = cfg_pb
    history = trainer.train(train_loader, val_loader, target_col=target_col, train_sampler=train_sampler)

    # -- Final evaluation --------------------------------------------------
    best_ckpt = CKPT_ROOT / "tier2_best.pt"
    # dist_barrier(): ensure rank 0 has fully written the checkpoint before
    # any rank tries to load it.  Without this, rank 1 can start reading
    # a partially-flushed file and get a "read error" or UnpicklingError.
    dist_barrier()
    ckpt = safe_load_checkpoint(best_ckpt)
    if ckpt is not None:
        trainer.model_core.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        log.warning("Could not load best checkpoint; evaluating with final weights.")

    test_mae_log, test_rmse_log, preds_log, trues_log = trainer.evaluate(
        test_loader, target_col, return_preds=True
    )

    if cfg.get("log_transform", False):
        valid       = ~torch.isnan(trues_log)
        p_valid     = preds_log[valid]
        t_valid     = trues_log[valid]

        # Review fix #1: exact exp-space MAE — compute from stored predictions.
        # Previous code used k_mean*(exp(log_mae)-1) which is only correct when
        # all k values equal k_mean.  Exact formula: mean|exp(pred)-exp(true)|
        mae_k_exact  = (torch.exp(p_valid) - torch.exp(t_valid)).abs().mean().item()
        rmse_k_exact = ((torch.exp(p_valid) - torch.exp(t_valid))**2).mean().sqrt().item()

        # Review fix #4: keep the diagnostic log-space numbers separate from the
        # benchmark metric used for publication/comparison on the linear-k scale.
        log.info("")
        log.info("─" * 68)
        log.info("  TIER 2 TEST RESULTS  (k_total_log target)")
        log.info("─" * 68)
        log.info("  Diagnostic log-space  MAE  = %.4f  [log(k) units]", test_mae_log)
        log.info("  Diagnostic log-space  RMSE = %.4f", test_rmse_log)
        log.info("  exp(MAE)                  = %.4f×  → ±%.1f%% average relative error",
                 math.exp(test_mae_log), (math.exp(test_mae_log) - 1) * 100)
        log.info("")
        log.info("  PRIMARY benchmark (linear-k) MAE  = %.4f  [dielectric units, exact]",
                 mae_k_exact)
        log.info("  PRIMARY benchmark (linear-k) RMSE = %.4f  [dielectric units]", rmse_k_exact)
        log.info("  Benchmark scale for publication: LINEAR-K")
        log.info("  ALIGNN paper    MAE  ≈ 0.81  [linear k, JARVIS DFPT, k range 5–100]")
        log.info("  Our distribution     k = 3.9–386 (100× wider → higher linear MAE expected)")
        log.info("  MAD:MAE (benchmark scale)  — see multitask table below for full breakdown")
        log.info("─" * 68)
    else:
        log.info("Tier 2 TEST  MAE=%.4f  RMSE=%.4f  (target: k_total)",
                 test_mae_log, test_rmse_log)

    mt_results = trainer.evaluate_multitask(test_loader, cfg, df_full=_main_ds_t2.df)  # FIX-2: use dataset.df (reset_index applied) not raw df_t2_k
    trainer.print_multitask_results(mt_results, split_name="TEST", tier_name="TIER 2")

    out_path = REPORT_ROOT / "tier2_test_results.json"
    primary_metrics = {
        "mae":                    test_mae_log,
        "rmse":                   test_rmse_log,
        "mae_log_k":              test_mae_log,
        "rmse_log_k":             test_rmse_log,
        "exp_mae_multiplier":     math.exp(test_mae_log),
        "benchmark_scale":        "linear",
        "diagnostic_scale":       "log",
    }
    if cfg.get("log_transform", False):
        primary_metrics["mae_linear_k"]  = mae_k_exact
        primary_metrics["rmse_linear_k"] = rmse_k_exact
        primary_metrics["mae"]           = mae_k_exact
        primary_metrics["rmse"]          = rmse_k_exact
    with open(out_path, "w") as f:
        json.dump({"primary": primary_metrics, "multitask": mt_results}, f, indent=2)
    log.info("Tier 2 test results saved → %s", out_path)

    return CKPT_ROOT / "tier2_best.pt"



# ==============================================================================
# FIX-T3-7: Option A structure imputation for process-only experimental rows
# ==============================================================================

def _impute_structures(
    df_process_only: "pd.DataFrame",
    df_structural:   "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Assign the best available JARVIS/MP crystal structure to experimental
    rows that have ALD process params + measured k but no atoms_dict.

    This is TODO(v4) Option A from run_tier3_finetune.  It is the lowest-
    friction path: no architecture change, no new model component.  The
    risk — that the imputed crystal embedding adds noise when the actual ALD
    film phase differs from the donor structure — is acceptable at this stage
    because:
      - Most ALD HfO2-family films at T<400°C are amorphous or monoclinic
      - The backbone already learned a smooth latent space over HfO2 phases
        in Tier 2 → a slight phase mismatch is a soft bias, not a hard error
      - The process encoder correction on top of the crystal embedding is
        exactly the signal we want the model to learn

    Matching algorithm (per process-only row):
      1. Exact formula match:  row["material"] == structural["formula"]
      2. Reduced formula fallback: pymatgen Composition normalisation
         (handles "Hf0.5Zr0.5O2" → reduced → HfO2/ZrO2 mix, etc.)
      3. Donor priority: JARVIS-DFT > MP (OptB88vdW more consistent with
         the functional conditioning the model learned in Tier 2)
      4. Stability tie-break: lowest |formation_energy_peratom| among donors

    Returns
    -------
    df_imputed  : rows from df_process_only that found a donor; atoms_dict,
                  imputed_structure, imputed_from, functional_code populated
    df_unmatched: rows where no donor was found (still no atoms_dict)
    """
    if len(df_process_only) == 0:
        return df_process_only.copy(), pd.DataFrame(columns=df_process_only.columns)

    # ── Build formula → best donor lookup from df_structural ─────────────────
    # df_structural contains all HfO2-family DFT rows (JARVIS + MP).
    # We need: atoms_dict, formation_energy_peratom, source, jid/mp_id.
    donor_pool = df_structural[df_structural["atoms_dict"].notna()].copy()

    # Prefer JARVIS donors (OptB88vdW matches Tier 2 functional conditioning)
    donor_pool["_source_rank"] = donor_pool["source"].apply(
        lambda s: 0 if str(s).startswith("JARVIS") else 1
    )

    # Ensure formation_energy_peratom is numeric
    donor_pool["_ef_abs"] = pd.to_numeric(
        donor_pool.get("formation_energy_peratom", pd.Series(dtype=float)),
        errors="coerce"
    ).abs().fillna(999.0)

    # Sort: JARVIS first, then by |Ef| ascending
    donor_pool = donor_pool.sort_values(
        ["_source_rank", "_ef_abs"], ascending=[True, True]
    )

    # formula → first (best) donor row
    formula_to_donor: dict[str, pd.Series] = {}
    for formula, grp in donor_pool.groupby("formula", sort=False):
        formula_to_donor[str(formula)] = grp.iloc[0]

    # ── Pymatgen reduced-formula normaliser (optional fallback) ───────────────
    try:
        from pymatgen.core import Composition
        def _reduced(f: str) -> str:
            try:
                return Composition(f).reduced_formula
            except Exception:
                return f
        reduced_map = {_reduced(f): f for f in formula_to_donor}   # reduced → canonical
    except ImportError:
        _reduced       = lambda f: f
        reduced_map    = {}

    # ── Formula alias table for common CSV → JARVIS mismatches ─────────────
    # CSV uses notation like 'Hf0.5Zr0.5O2', 'HfAlO', 'HfLaO' while JARVIS
    # uses 'HfZrO4', 'HfAlO4', 'HfLaO4' etc.  Pymatgen reduced-formula
    # handles this automatically when installed, but may not be available.
    # This table maps common CSV material_system strings → canonical formulas
    # that exist in JARVIS, allowing imputation without pymatgen.
    FORMULA_ALIASES: Dict[str, List[str]] = {
        # HZO variants
        "Hf0.5Zr0.5O2":        ["HfZrO4", "Hf2Zr2O8", "HfO2"],
        "Hf0.5Zr0.5O2/HfO2":  ["HfZrO4", "HfO2"],
        "HfZrO2_LaSi":         ["HfZrO4", "HfO2"],
        "HfO2/ZrO2/HfO2":      ["HfO2"],
        # Hf-based ternaries — use HfO2 as structural proxy
        "HfAlO":               ["HfAlO4", "Al2HfO6", "HfO2"],
        "HfSiO":               ["HfSiO4", "Hf2Si2O8", "HfO2"],
        "HfLaO":               ["HfO2"],
        "HfYO":                ["HfO2"],
        "HfO2:La":             ["HfO2"],
        "HfO2:Al":             ["HfO2"],
        "HfO2:Y":              ["HfO2"],
        "HfO2:Gd":             ["HfO2"],
        "HfO2:N":              ["HfO2"],
        "HfO2:Sr":             ["HfO2"],
        # Multi-layer stacks — use primary layer formula
        "HfO2 on MoS2":        ["HfO2"],
        "HfO2/Al2O3_nanolaminate": ["HfO2"],
        "HfO2/Al2O3_multilayer":   ["HfO2"],
        # Zr-based
        "ZrAlO":               ["ZrO2"],
        # Other high-k oxides (map to closest available JARVIS formula)
        "SrTiO3":              ["SrTiO3", "TiO2"],
        "BaTiO3":              ["BaTiO3", "TiO2"],
        "La2O3":               ["La2O3", "LaAlO3"],
        "LaAlO3":              ["LaAlO3", "La2O3"],
        "LaAlO_solgel":        ["LaAlO3"],
        "La1-xAlxO3":          ["LaAlO3"], # non-stoichiometric -> use LaAl03 structure
        "Ta2O5":               ["Ta2O5"],
        "TiO2":                ["TiO2"],
        "Y2O3":                ["Y2O3"],
        "Nb2O5":               ["Nb2O5", "TiO2"],
    }

    def _find_donor(material: str) -> Optional[pd.Series]:
        """Return best donor Series for a given material string, or None."""
        m = str(material).strip()
        if not m or m.lower() in ("nan", "none", ""):
            return None
        # 1. Exact match
        if m in formula_to_donor:
            return formula_to_donor[m]
        # 2. Reduced formula via pymatgen
        m_red = _reduced(m)
        if m_red in reduced_map:
            return formula_to_donor[reduced_map[m_red]]
        # 3. Alias table lookup (handles Hf0.5Zr0.5O2, HfAlO, etc.)
        if m in FORMULA_ALIASES:
            for alias in FORMULA_ALIASES[m]:
                if alias in formula_to_donor:
                    return formula_to_donor[alias]
                alias_red = _reduced(alias)
                if alias_red in reduced_map:
                    return formula_to_donor[reduced_map[alias_red]]
        # 4. Substring match for inline annotations (e.g. "HfO2 (monoclinic)")
        for canon in formula_to_donor:
            if m.startswith(canon) or canon.startswith(m):
                return formula_to_donor[canon]
        # 5. Partial token match: first token of CSV formula vs JARVIS formulas
        #    Handles 'HfO2/Al2O3_nanolaminate' → token 'HfO2' → donor HfO2
        first_token = m.split("/")[0].split("_")[0].split(" ")[0].strip()
        if first_token != m and first_token in formula_to_donor:
            return formula_to_donor[first_token]
        first_red = _reduced(first_token)
        if first_red in reduced_map:
            return formula_to_donor[reduced_map[first_red]]
        # 6. Last-resort fallback: use HfO2 donor for any Hf-containing formula
        #    or the closest available donor for non-Hf high-k oxides.
        #    This guarantees every experimental row gets a structure so it can
        #    contribute process-param gradient even if the crystal structure is
        #    an approximation. Logged as WARNING so it is auditable.
        if "Hf" in m and "HfO2" in formula_to_donor:
            log.debug("_find_donor: using HfO2 as fallback for '%s'", m)
            return formula_to_donor["HfO2"]
        if "Zr" in m and "ZrO2" in formula_to_donor:
            log.debug("_find_donor: using ZrO2 as fallback for '%s'", m)
            return formula_to_donor["ZrO2"]
        return None

    # ── Impute ────────────────────────────────────────────────────────────────
    df_imp = df_process_only.copy()
    df_imp["imputed_structure"] = False
    df_imp["imputed_from"]      = ""

    matched_idx   = []
    unmatched_idx = []

    for idx, row in df_imp.iterrows():
        donor = _find_donor(row.get("material", ""))
        if donor is not None:
            df_imp.at[idx, "atoms_dict"]          = donor["atoms_dict"]
            df_imp.at[idx, "imputed_structure"]    = True
            df_imp.at[idx, "has_structure"]        = True   # FIX-HAS-STRUCT: required by
            # HighKGraphDataset.__getitem__ valid filter: has_structure.fillna(False).
            # Without this, imputed rows have has_structure=NaN → fillna(False) →
            # excluded from valid_idx → never sampled → proc_avail_pct=0% always.
            df_imp.at[idx, "imputed_from"]         = str(
                donor.get("jid", donor.get("mp_id", donor.get("row_hash", "?")))
            )
            # Carry donor spacegroup for record-keeping (does not affect training)
            if "spacegroup" in donor.index:
                df_imp.at[idx, "imputed_spacegroup"] = donor["spacegroup"]
            # functional_code: use OptB88vdW (0) for JARVIS donors, PBE (1) for MP
            source = str(donor.get("source", "JARVIS"))
            df_imp.at[idx, "functional_code"] = (
                FUNCTIONAL_CODE.get("OptB88vdW", 0)
                if source.startswith("JARVIS")
                else FUNCTIONAL_CODE.get("PBE", 1)
            )
            matched_idx.append(idx)
        else:
            unmatched_idx.append(idx)

    df_imputed   = df_imp.loc[matched_idx].copy()
    df_unmatched = df_imp.loc[unmatched_idx].copy()

    # ── Log results ───────────────────────────────────────────────────────────
    log.info(
        "FIX-T3-7 structure imputation: %d/%d rows matched  (%d unmatched)",
        len(df_imputed), len(df_process_only), len(df_unmatched),
    )
    if df_imputed is not None and len(df_imputed) > 0:
        by_formula = df_imputed.groupby(
            df_imputed.get("material", pd.Series(dtype=str))
        ).size()
        for formula, cnt in by_formula.items():
            sample_donor = df_imputed[
                df_imputed.get("material", pd.Series(dtype=str)) == formula
            ]["imputed_from"].iloc[0] if cnt > 0 else "?"
            log.info(
                "  %-20s  %3d rows  ← donor: %s", formula, cnt, sample_donor
            )
    if len(df_unmatched) > 0:
        # FIX-DONOR-LOG: use 'material' with fallback to 'material_system'
        # df_unmatched.get('material') returns default empty Series when the
        # column is absent — the [] in the warning was a misleading no-op.
        _mat_col = "material" if "material" in df_unmatched.columns else \
                   "material_system" if "material_system" in df_unmatched.columns else None
        unmatched_formulas = (
            df_unmatched[_mat_col].dropna().unique().tolist()
            if _mat_col else []
        )
        log.warning(
            "FIX-T3-7: %d rows unmatched — no JARVIS/MP donor for: %s.  "
            "Add these formulas to the HfO2-family filter or supply atoms_dict manually.",
            len(df_unmatched), unmatched_formulas,
        )

    return df_imputed, df_unmatched



def _clean_numeric_series(series: "pd.Series", col_name: str = "") -> "pd.Series":
    """
    FIX-T3-11: Convert literature-style numeric strings to float.

    Handles four patterns found in ALD process database CSVs:
      A. Approximation prefix  : '~20'  '≈25'  'ca.30'  '<5'  → 20.0 / 25.0 / 30.0 / 5.0
      B. Numeric range         : '50-82' '17.0-24.5' '10 to 30' → midpoint
      C. Value + unit suffix   : '3.3e-6 at 1MV/cm' '17.0 MV/cm' → leading float
      D. Qualitative strings   : 'negligible ...' → NaN  (logged at DEBUG)

    Returns a float64 Series. Existing NaN / None values pass through unchanged.
    """
    import re as _re

    _APPROX_PREFIXES = ('~', '≈', 'ca.', 'about ', '~<', '~>', '<', '>', '≤', '≥')
    # Suffix patterns to strip before numeric parse
    _UNIT_SUFFIX_RE  = _re.compile(
        r'\s*(MV/cm|V/cm|MV|kV|Torr|mTorr|Pa|MPa|GPa|nm|Å|at\s+\d[\de\.]*\s*\w*/\w*|at\s+\S+)',
        _re.IGNORECASE,
    )
    # Range separators
    _RANGE_SEP_RE = _re.compile(r'^(\d[\d\.e\+\-]*)\s*(?:–|-)\s*(\d[\d\.e\+\-]*)$')
    _TO_SEP_RE    = _re.compile(r'^(\d[\d\.e\+\-]*)\s+to\s+(\d[\d\.e\+\-]*)$', _re.IGNORECASE)

    def _parse_one(raw) -> float:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return float("nan")
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "n/a", "na", "-", ""):
            return float("nan")

        # Pattern C: strip trailing unit / field suffix first
        s = _UNIT_SUFFIX_RE.sub("", s).strip()

        # Pattern A: strip approximation prefix
        for pfx in _APPROX_PREFIXES:
            if s.startswith(pfx):
                s = s[len(pfx):].strip()
                break

        # Pattern B: range → midpoint ('lo to hi' takes priority over 'lo-hi'
        # to avoid treating negative numbers as ranges)
        m = _TO_SEP_RE.match(s)
        if m:
            try:
                return (float(m.group(1)) + float(m.group(2))) / 2.0
            except ValueError:
                pass

        m = _RANGE_SEP_RE.match(s)
        if m:
            try:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo <= hi:                        # sanity: lo must be ≤ hi
                    return (lo + hi) / 2.0
            except ValueError:
                pass

        # Direct float parse (covers scientific notation)
        try:
            return float(s)
        except ValueError:
            log.debug(
                "FIX-T3-11 _clean_numeric_series [%s]: "
                "qualitative/unmappable value '%s' → NaN", col_name, raw
            )
            return float("nan")

    cleaned = series.apply(_parse_one).astype(float)

    # Log recovery summary
    n_before = int(series.notna().sum()) if hasattr(series, "notna") else 0
    n_after  = int(cleaned.notna().sum())
    n_recovered = n_after - n_before
    if n_recovered > 0:
        log.info(
            "FIX-T3-11 numeric clean [%-28s]: "
            "%d non-blank → %d valid float  (+%d recovered from ~prefix/range/suffix)",
            col_name + "]", n_before, n_after, n_recovered,
        )
    elif n_before > 0:
        log.debug(
            "FIX-T3-11 numeric clean [%-28s]: "
            "%d non-blank → %d valid float  (0 additional recoveries)",
            col_name + "]", n_before, n_after,
        )
    return cleaned


def run_tier3_finetune(
    df_tier3: pd.DataFrame,
    pretrained_weights: Path,
    ablate_context: bool = False,
    df_tier2: Optional[pd.DataFrame] = None,
):
    """
    Tier 3 -- Project fine-tuning on HfO2-family (with process parameters).

    Loads Tier 2 pretrained weights.
    All layers unfrozen -- final adaptation to project-specific material space.
    Very low learning rate (1e-5, reduced from 5e-5 in FIX-T3-6) to preserve
    domain knowledge while adapting to project-specific k_measured distribution.

    Key difference from Tiers 1-2: Tier 3 dataset includes experimental
    entries with real ALD/anneal process parameters. The ALIGNN backbone
    handles the crystal structure branch; a separate MLP handles process
    parameters. Both outputs are concatenated before task heads.

    FIX-T3-PHASE-A: df_tier2 (all oxide dielectric rows) is used for a
    20-epoch band_gap consolidation Phase A before k_measured_log Phase B.
    This prevents the backbone from catastrophically forgetting band_gap
    representations when fine-tuned on the small HfO2-family k set.
    Pass df_tier2=None to skip Phase A (falls back gracefully with WARNING).
    """
    log.info("=" * 70)
    log.info(" TIER 3 -- Project Fine-tune (HfO2 Family)")
    log.info(" Rows: %d   Target: k_measured", len(df_tier3))
    log.info("=" * 70)

    cfg = TIER3_TRAIN_CONFIG

    # Separate structural vs process-only rows
    df_structural   = df_tier3[df_tier3["atoms_dict"].notna()].copy()
    df_process_only = df_tier3[df_tier3["atoms_dict"].isna()].copy()

    log.info(
        "  Structural rows (ALIGNN path): %d  |  "
        "Process-only rows (unused in v3): %d",
        len(df_structural), len(df_process_only)
    )

    # ── Structure imputation ──────────────────────────────
    # FIX-HAS-STRUCT / FIX-DONOR-POOL: build broad donor pool from df_tier2
    # for _impute_structures (all high-k families) separately from df_structural
    # (Hf/Zr-only training rows). This prevents the 1128-row df_structural issue
    # where expanded is_hfo2_family diluted proc signal to ~10% per batch.

    if df_tier2 is not None:
        df_donor_pool_wide = df_tier2[
            df_tier2["formula"].apply(
                lambda f: isinstance(f, str) and "O" in f and
                          any(el in f for el in HIGH_K_DONOR_ELEMENTS)
            )
        ].copy()
    else:
        df_donor_pool_wide = df_structural  # fallback: use existing structural rows

    df_imputed, df_unmatched = _impute_structures(df_process_only, df_donor_pool_wide)

    if len(df_imputed) > 0:
        df_structural = pd.concat(
            [df_structural, df_imputed], ignore_index=True, sort=False
        )
        log.info(
            "After imputation: df_structural=%d  "
            "(DFT-only=%d  imputed-experimental=%d  unmatched=%d)",
            len(df_structural),
            len(df_structural) - len(df_imputed),
            len(df_imputed),
            len(df_unmatched),
        )
    else:
        log.warning(
            "FIX-T3-7: No experimental rows could be imputed "
            "(donor pool empty or no formula matches).  "
            "Check that df_structural contains HfO2-family JARVIS entries "
            "and that process_db.csv 'material' column uses standard formulas "
            "(e.g. HfO2, Al2O3, ZrO2)."
        )

    if len(df_unmatched) > 0:
        log.warning(
            "Tier 3: %d process-only rows remain EXCLUDED — no matching "
            "crystal structure donor found.  "
            "See FIX-T3-7 warning above for unmatched formulas.",
            len(df_unmatched),
        )

    # Log-transform k (mirrors Tier 2 k_total_log approach).
    if cfg.get("log_transform", False):
        src_col = cfg["log_original_col"]   # "k_measured"
        log_col = cfg["target"]             # "k_measured_log"
        k_num   = pd.to_numeric(df_structural[src_col], errors="coerce")
        valid_k = k_num.notna()
        df_structural[log_col] = np.nan
        df_structural.loc[valid_k, log_col] = np.log(k_num[valid_k].clip(lower=0.1))
        log.info(
            "FIX-T3-9A log transform: %s → %s  "
            "(%d valid rows  range [%.3f, %.3f] log units = k [%.1f, %.1f])",
            src_col, log_col,
            int(valid_k.sum()),
            df_structural[log_col].dropna().min(),
            df_structural[log_col].dropna().max(),
            float(k_num[valid_k].min()), float(k_num[valid_k].max()),
        )

    # Log valid-row counts for each target BEFORE build_dataloader.
    log.info("─" * 65)
    log.info("Tier 3 target coverage in df_structural (%d rows):", len(df_structural))
    _diag_targets = [cfg["target"], "k_measured", "band_gap", "J_g_A_cm2", "E_BD_MV_cm"]
    for _tgt in _diag_targets:
        if _tgt not in df_structural.columns:
            if _tgt != "k_measured":   # k_measured_log may not exist yet if no log-transform
                log.warning("  %-22s  column ABSENT — check target_renames in load_experimental_process_db", _tgt)
            continue
        _total = int(df_structural[_tgt].notna().sum())
        if "source" in df_structural.columns:
            _dft = int(df_structural[df_structural["source"] != "Experimental"][_tgt].notna().sum())
            _exp = int(df_structural[df_structural["source"] == "Experimental"][_tgt].notna().sum())
            log.info("  %-22s  total=%-5d  (DFT=%-5d  experimental=%-5d)  coverage=%.1f%%",
                     _tgt + ":", _total, _dft, _exp,
                     100.0 * _total / max(len(df_structural), 1))
        else:
            log.info("  %-22s  total=%d  coverage=%.1f%%",
                     _tgt + ":", _total, 100.0 * _total / max(len(df_structural), 1))
    log.info("─" * 65)

    # FIX-T3-10B/C: validate valid k_measured_log count before building
    # the dataloader.
    #
    # Guard 1 (stale cache detection):
    #   valid_k < 50 almost certainly means tier3.h5 predates FIX-T3-8
    #   (dielectric_constant_k → k_measured rename runs inside build_tier3).
    #   Only ~33 JARVIS DFPT entries have k_measured in a stale cache.
    #
    # Guard 2 (adaptive min_epochs against catastrophic overfitting):
    #   min_epochs=60 on ≤70 training rows with 10M parameters causes
    #   the backbone to overfit k_measured_log and forget band_gap.
    #   Cap min_epochs at max(20, valid_k_count // 2) when data is scarce.
    _kmeas_col = cfg["target"]  # "k_total_log" after UNIFY-K (was "k_measured_log")
    valid_k_count = int(df_structural[_kmeas_col].notna().sum()) if _kmeas_col in df_structural.columns else 0

    if valid_k_count < 50:
        log.warning(
            "FIX-T3-10B STALE CACHE DETECTED: only %d valid '%s' rows "
            "(expected ≥ 100 after FIX-T3-8 column renames).  "
            "FIX-T3-8 (dielectric_constant_k → k_measured) runs inside "
            "build_tier3() which requires --rebuild_tier3 to execute.  "
            "Without rebuild: train≈23 rows → catastrophic overfitting → "
            "band_gap and aux heads will degrade severely.  "
            "Re-run with --rebuild_tier3 before continuing.",
            valid_k_count, _kmeas_col,
        )

    cfg_min_epochs = cfg.get("min_epochs", 40)
    if valid_k_count < 100:
        safe_min = max(20, valid_k_count // 2)
        if safe_min < cfg_min_epochs:
            n_train_est = max(1, int(valid_k_count * cfg["train_ratio"]))
            n_batches_est = max(1, n_train_est // cfg.get("batch_size", 8))
            log.warning(
                "FIX-T3-10C ADAPTIVE min_epochs: %d → %d  "
                "(valid_k=%d  train≈%d  batches/ep≈%d  "
                "full min=%d would force %d steps on %d examples → overfit).  "
                "Run --rebuild_tier3 to unlock min_epochs=%d.",
                cfg_min_epochs, safe_min,
                valid_k_count, n_train_est, n_batches_est,
                cfg_min_epochs, cfg_min_epochs * n_batches_est, n_train_est,
                cfg_min_epochs,
            )
            cfg = {**cfg, "min_epochs": safe_min}

    log.info(
        "FIX-T3-10 summary: valid_%s=%d  min_epochs=%d  patience=%d  "
        "worst-case-stop=ep%d",
        _kmeas_col, valid_k_count,
        cfg.get("min_epochs", 40), cfg.get("patience", 40),
        cfg.get("min_epochs", 40) + cfg.get("patience", 40),
    )


    # Filter to k-valid rows + reset_index for positional indexing.

    df_phase_b = (
        df_structural[df_structural[cfg["target"]].notna()]
        .copy()
        .reset_index(drop=True)
    )
    _n_phase_b = len(df_phase_b)
    _n_phase_b_train = int(_n_phase_b * cfg["train_ratio"])
    log.info(
        "Phase B dataloader built on k-valid subset  "
        "rows=%d  train≈%d  val≈%d  test≈%d  "
        "k-density=%.1f%%)",
        _n_phase_b,
        _n_phase_b_train,
        int(_n_phase_b * cfg["val_ratio"]),
        _n_phase_b - _n_phase_b_train - int(_n_phase_b * cfg["val_ratio"]),
        100.0 * _n_phase_b / max(len(df_structural), 1),
    )

    train_loader, val_loader, test_loader, train_sampler, _main_ds_t3 = build_dataloader(
        df         = df_phase_b,
        target_col = cfg["target"],
        aux_cols   = cfg["aux_targets"],
        train_frac = cfg["train_ratio"],
        val_frac   = cfg["val_ratio"],
        batch_size = cfg["batch_size"],
    )

    # Load model with Tier 2 weights -- no frozen layers for final fine-tune.
    # FIX-OBS1 collateral: explicit task_names from tier config for head-name
    # consistency with train_epoch targets_dict routing.
    #
    # Include "band_gap" so Phase A can train on it; Phase B/B2 are single-task
    # so band_gap head retains Phase A learned weights.
    task_names_t3 = [cfg["target"], "band_gap"]
    log.info("Tier 3 training with task heads: %s", task_names_t3)
    model = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names_t3)
    model.load_pretrained_weights(pretrained_weights, strict=False)
    model.configure_bounded_process_residual(
        enabled=cfg.get("bounded_process_residual", False),
        bound=cfg.get("process_delta_bound", 0.10),
        use_categorical=cfg.get("process_delta_use_categorical", False),
    )
    model.unfreeze_all()
    # v2.2: fit encoder stats AFTER weight load so Tier 3 ALD/stack data
    # overwrites the uninformative zeros from the Tier 2 checkpoint.
    # This is the FIRST tier where proc_avail_flag > 0 rows appear (experimental
    # entries from process_db_clean.csv). The encoders will receive real gradient
    # signal and alpha/beta will grow from their 1e-3 initial values.
    model.fit_encoder_stats(df_structural)

    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix="tier3",
                            ablate_context=ablate_context)

    def _eval_exact_k_for_diag(loader, split_name: str, stage_name: str) -> dict:
        """Evaluate log-space and exact linear-k metrics for Tier-3 diagnostics."""
        mae_log, rmse_log, preds_log, trues_log = trainer.evaluate(
            loader, cfg["target"], return_preds=True
        )
        out = {
            "stage": stage_name,
            "split": split_name,
            "mae_log": float(mae_log),
            "rmse_log": float(rmse_log),
            "mae_k": float("nan"),
            "rmse_k": float("nan"),
            "mad_k": float("nan"),
            "mad_mae_k": float("nan"),
            "n": 0,
        }
        if cfg.get("log_transform", False) and trues_log.numel() > 0:
            valid = ~torch.isnan(trues_log)
            out["n"] = int(valid.sum())
            if out["n"] > 0:
                p_lin = torch.exp(preds_log[valid])
                t_lin = torch.exp(trues_log[valid])
                out["mae_k"] = float((p_lin - t_lin).abs().mean().item())
                out["rmse_k"] = float(((p_lin - t_lin) ** 2).mean().sqrt().item())
                t_np = t_lin.cpu().numpy()
                out["mad_k"] = float(np.mean(np.abs(t_np - np.mean(t_np))))
                out["mad_mae_k"] = (
                    out["mad_k"] / out["mae_k"] if out["mae_k"] > 0 else float("nan")
                )
        log.info(
            "RESIDUAL-PROC-DIAG | %-18s | %-5s | "
            "log_MAE=%.4f log_RMSE=%.4f | "
            "linear_MAE=%.4f linear_RMSE=%.4f MAD=%.4f MAD:MAE=%.2f N=%d",
            stage_name, split_name, out["mae_log"], out["rmse_log"],
            out["mae_k"], out["rmse_k"], out["mad_k"],
            out["mad_mae_k"], out["n"],
        )
        return out

    log.info("─" * 68)
    log.info("RESIDUAL-PROC-DIAG: epoch-0 baseline before any Tier-3 gradient update")
    log.info(
        "Purpose: if epoch-0/epoch-1 already matches final performance, the gain is "
        "from the loaded Tier-2 DFT baseline rather than Tier-3 process-residual learning."
    )
    diag_epoch0_val = _eval_exact_k_for_diag(val_loader,  "VAL",  "epoch0_pretrain")
    diag_epoch0_test = _eval_exact_k_for_diag(test_loader, "TEST", "epoch0_pretrain")
    log.info("─" * 68)

    # ── PHASE-A: band_gap consolidation on Tier 3 structural data ──────────

    phase_a_steps = cfg["phase_a_steps"]

    if phase_a_steps > 0 and len(df_structural) > 0:
        df_pa = df_structural[df_structural["band_gap"].notna()].copy()
        log.info("─" * 68)
        log.info(
            "TIER 3 PHASE A: band_gap consolidation (Tier 3 data) "
            "rows=%d  budget=%d  lr=%.2e",
            len(df_pa), phase_a_steps,cfg["phase_a_lr"],
        )
        phase_b_steps = int(_n_phase_b * cfg["train_ratio"] / cfg["batch_size"]) * cfg.get("epochs", 100)
        log.info(
            "  Purpose: anchor backbone in HfO2 family representation subspace  "
            "before Phase B k_total_log fine-tuning"
            "Phase A:Phase B step ratio = %d:%d = %.2fx",
            phase_a_steps, phase_b_steps, phase_a_steps / max(phase_b_steps, 1)
        )

        pa_loader, _, _, pa_sampler, _ds = build_dataloader(
            df         = df_pa,
            target_col = "band_gap",
            aux_cols   = [],
            train_frac = 0.95,
            val_frac   = 0.05,
            batch_size = cfg["batch_size"],
        )
        pa_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg["phase_a_lr"],
            weight_decay = cfg["weight_decay"],
        )
        _saved_cfg        = trainer.cfg
        _saved_optimizer  = trainer.optimizer
        trainer.optimizer = pa_optimizer

        import itertools as _itools

        class _StepBudgetLoader:
            """Wraps a DataLoader and stops after `max_steps` batches."""
            def __init__(self, loader, max_steps):
                self._loader   = loader
                self._max      = max_steps
                self._done     = 0
            def __iter__(self):
                for batch in self._loader:
                    if self._done >= self._max:
                        return
                    yield batch
                    self._done += 1
            def __len__(self):
                return min(self._max, len(self._loader))

        _cycled_loader = _StepBudgetLoader(
            type('_CycledLoader', (), {
                '__iter__': lambda self: _itools.islice(
                    _itools.cycle(pa_loader), phase_a_steps
                ),
                '__len__': lambda self: phase_a_steps,
            })(),
            max_steps = phase_a_steps,
        )

        trainer.cfg = {**cfg, "epochs": 1, "early_stopping": False}
        sched_pa    = trainer.build_scheduler(phase_a_steps, 1)

        ep = trainer.train_epoch(_cycled_loader, sched_pa, "band_gap")
        log.info(
            "Tier 3 Phase A  %d steps complete  loss=%.4f  proc_avail=%.1f%%",
            phase_a_steps, ep["loss"], ep.get("proc_avail_pct", 0.0),
        )

        trainer.cfg       = _saved_cfg
        trainer.optimizer = _saved_optimizer
        log.info(
            "Tier 3 Phase A complete (%d steps) — starting Phase B k_total_log fine-tuning",
            phase_a_steps,
        )
        log.info("─" * 68)

    elif phase_a_steps > 0 and len(df_structural) == 0:
        log.warning(
            "Phase_ A: df_structural is empty. Skipping Phase A conslidation. ",
        )

    # ── Phase B: k_total_log fine-tuning  ───────────────────────────────────
    freeze_backbone = cfg.get("freeze_backbone", False)
    residual_proc_mode = bool(cfg.get("bounded_process_residual", False))
    if freeze_backbone:
        if residual_proc_mode:
            model.freeze_for_bounded_process_residual(
                train_base_head=cfg.get("process_delta_train_base_head", False)
            )
        else:
            model.freeze_backbone()
        log.info("─" * 68)
        log.info(
            "Tier 3 Phase B: %s  rows=%d  epochs=%d  lr≈%.2e  batch_size=%d",
            "DFT-only base + bounded process residual" if residual_proc_mode else "backbone-frozen single-task k_total_log fine-tuning",
            _n_phase_b, cfg["epochs"], cfg["learning_rate"], cfg["batch_size"]
        )
        if residual_proc_mode:
            log.info(
                " Frozen : ALIGNN backbone, functional conditioning, and base task head; "
                "training bounded process_delta_head only (unless train_base_head=True)."
            )
        else:
            log.info(" Frozen : ALIGNN backbone (all message passing layers)")
            log.info(" Training task_heads, context_proj, proc_encoder, stack_encoder, alpha, beta")

        # Build Phase B dataloader with single task targets (k_total_log only)
        pb_loader, pb_val_loader, _, pb_sampler, _ds = build_dataloader(
            df         = df_phase_b,
            target_col = cfg["target"],
            aux_cols   = [], # single-task: no aux heads
            train_frac = cfg["train_ratio"],
            val_frac   = cfg["val_ratio"],
            batch_size = cfg["batch_size"],
        )

        # Build optimizer with ONLY trainable (non-backbone) parameters
        pb_params = [p for p in model.parameters() if p.requires_grad]
        pb_optimizer = torch.optim.AdamW(
            pb_params,
            lr           = cfg["learning_rate"],
            weight_decay = cfg["weight_decay"],
        )
        _saved_optimizer  = trainer.optimizer
        trainer.optimizer = pb_optimizer

        pb_history = trainer.train(
            pb_loader, pb_val_loader,
            target_col = cfg["target"],
            train_sampler=pb_sampler,
        )

        _pb_best_epoch = trainer.best_epoch
        _pb_best_mae = trainer.best_val_mae
        trainer.optimizer = _saved_optimizer
        log.info(
            "Phase B complete: best val MAE=%.4f  at epoch=%d",
            _pb_best_mae, _pb_best_epoch
        )

        # RESIDUAL-PROC-DIAG: make the epoch-1 vs best-epoch behavior explicit.
        # If best_epoch=1 and epoch-0/epoch-1 ≈ final, Tier-3 training is not
        # adding value; it is mostly preserving the loaded Tier-2 DFT baseline.
        diag_epoch1_val = diag_epoch1_test = None
        if pb_history:
            h1 = pb_history[0]
            log.info(
                "RESIDUAL-PROC-DIAG | epoch1 history | VAL   | "
                "log_MAE=%.4f log_RMSE=%.4f  train_loss=%.4f  "
                "proc_avail=%.1f%% stack_avail=%.1f%%",
                h1.get("val_mae", float("nan")),
                h1.get("val_rmse", float("nan")),
                h1.get("train_loss", float("nan")),
                h1.get("proc_avail_pct", 0.0),
                h1.get("stack_avail_pct", 0.0),
            )

        epoch1_ckpt = CKPT_ROOT / "tier3_epoch1.pt"
        if epoch1_ckpt.exists():
            ckpt_e1 = safe_load_checkpoint(epoch1_ckpt)
            if ckpt_e1 is not None:
                state_e1 = _remap_state_dict(ckpt_e1["model_state_dict"])
                missing_e1, unexpected_e1 = trainer.model_core.load_state_dict(
                    state_e1, strict=False
                )
                log.info(
                    "RESIDUAL-PROC-DIAG: reloaded tier3_epoch1.pt for exact epoch-1 eval "
                    "(missing=%d unexpected=%d)",
                    len(missing_e1), len(unexpected_e1),
                )
                diag_epoch1_val = _eval_exact_k_for_diag(pb_val_loader, "VAL", "epoch1_ckpt")
                diag_epoch1_test = _eval_exact_k_for_diag(test_loader, "TEST", "epoch1_ckpt")
        else:
            log.warning(
                "RESIDUAL-PROC-DIAG: tier3_epoch1.pt not found. "
                "Set save_epoch1_checkpoint=True in TIER3_TRAIN_CONFIG to enable exact epoch-1 evaluation."
            )

        log.info("─" * 68)

        if residual_proc_mode:
            log.info(
                "RESIDUAL-PROC: skipping Phase B2 backbone unfreeze so the final "
                "model remains DFT-base + bounded process correction."
            )
        else:

                # Phase B2: unfreeze backbone, single-task (no aux heads)
            # Abandon multi-task for Tier3. Phase B2 is single-task
            # (k_total_log only) to avoid aux head gradient competition on 193 rows
            unfreeze_lr = cfg.get("unfreeze_backbone_lr", 1e-5)  # FIX-1: removed dead unfreeze_after variable (was read but never used)
            b2_epochs = max(0, cfg["epochs"] - _pb_best_epoch)
            if b2_epochs <= 0:
                b2_epochs = 30 # mininium if Phase B already converged

            log.info("─" * 68)
            log.info(
                "TIER 3 PHASE B2: Unfreeze backbone, Single-task"
                "epochs=%d  backbone_lr≈%.2e  batch_size=%d",
                b2_epochs, unfreeze_lr, cfg["batch_size"]
            )
            log.info("UnFreezing backbone at lr = %.2e (lower than Phase B lr=%.2e)", unfreeze_lr, cfg["learning_rate"])
            log.info(" Single-task: k_total_log only (no aux heads)")

            model.unfreeze_backbone(lr=unfreeze_lr)

            # Build Phase B2 dataloader --single-task (no aux targets)
            pb2_loader, pb2_val_loader, _, pb2_sampler, _ds = build_dataloader(
                df         = df_phase_b,
                target_col = cfg["target"],
                aux_cols   = [], # single-task: no aux heads
                train_frac = cfg["train_ratio"],
                val_frac   = cfg["val_ratio"],
                batch_size = cfg["batch_size"],
            )

            # Two tier LR: backbone at unfreeze_lr heads/context at higher LR
            # This prevents catastrophic forgetting while allowing task-specific adaption
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params =  [p for p in model.parameters()
                            if p not in set(backbone_params) and p.requires_grad]

            pb2_param_groups = [
                {"params": backbone_params, "lr": unfreeze_lr, "weight_decay": cfg["weight_decay"]},
                {"params": head_params, "lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]},
            ]
            log.info(
                "Phase B2 two tier LR: backbone=%.2e  heads/context=%.2e",
                unfreeze_lr, cfg["learning_rate"]
            )

            # Build optimizer with two tier param groups
            pb2_optimizer = torch.optim.AdamW(pb2_param_groups)
            _saved_optimizer  = trainer.optimizer
            trainer.optimizer = pb2_optimizer

            # update cfg for B2
            _saved_cfg = trainer.cfg
            trainer.cfg = {
                **cfg,
                "epochs": b2_epochs,
                "learning_rate": unfreeze_lr,
            }

            pb2_history = trainer.train(
                pb2_loader, pb2_val_loader,
                target_col=cfg["target"],
                train_sampler=pb2_sampler,
            )

            trainer.optimizer = _saved_optimizer
            trainer.cfg = _saved_cfg
            log.info(
                "Phase B2 complete: Best val_MAE=%.4f  at epoch %d"
                "(Phase B2 epoch %d)",
                trainer.best_val_mae, trainer.best_epoch,
                trainer.best_epoch - _pb_best_epoch,
            )
            log.info("─" * 68)

    # CKPT-RT2+RT3 FIX: reload best checkpoint with strict=False + key remapping.
    # Previous code: trainer.model_core.load_state_dict(ckpt["model_state_dict"])
    # Problems:
    #   (a) strict=True (default) raises RuntimeError if tier3_best.pt on disk
    #       is from a previous run with task_heads.k_measured_log.* while the
    #       current model has task_heads.k_total_log.* (CKPT-RT2).
    #   (b) torch.load called directly, bypassing safe_load_checkpoint integrity
    #       checks (size > 1024, model_state_dict key present) (CKPT-RT3).
    #   (c) No key remapping — old k_measured[_log] weights silently discarded.
    # Fix: use safe_load_checkpoint + _remap_state_dict + strict=False.
    best_ckpt = CKPT_ROOT / "tier3_best.pt"
    if best_ckpt.exists():
        ckpt = safe_load_checkpoint(best_ckpt)
        if ckpt is not None:
            state = _remap_state_dict(ckpt["model_state_dict"])
            missing, unexpected = trainer.model_core.load_state_dict(
                state, strict=False
            )
            log.info(
                "Reloaded tier3_best.pt for evaluation  "
                "(missing=%d  unexpected=%d)",
                len(missing), len(unexpected),
            )

    # FIX-T3-LINMAE: use return_preds=True for exact exp-space MAE.
    # Previous code: (exp(log_MAE)-1)*100 — only correct when all k equal k_mean.
    # Exact formula: mean|exp(pred_log) - exp(true_log)|.  Mirrors Tier 2 ~L4454.
    # Previous MAD:MAE=0.14 was a log-space artifact: MAD≈0 because all 5 test
    # samples were near-identical monoclinic HfO2 (log(k)≈log(22)≈3.09 for all).
    # New block computes MAD on linear-k and gives pass/fail vs benchmarks.

    test_mae_log, test_rmse_log, preds_log, trues_log = trainer.evaluate(
        test_loader, cfg["target"], return_preds=True
    )

    log.info("─" * 68)
    log.info("TIER 3 TEST RESULTS")
    log.info("─" * 68)

    if cfg.get("log_transform", False):
        # ── Diagnostic: log-space (training metric, not the benchmark) ────────
        log.info(
            "  Diagnostic log-space  MAE  = %.4f  [log(k) units]", test_mae_log
        )
        log.info("  Diagnostic log-space  RMSE = %.4f", test_rmse_log)
        log.info(
            "  exp(MAE)                  = %.4f×  → ±%.1f%% avg relative error",
            math.exp(test_mae_log),
            (math.exp(test_mae_log) - 1) * 100,
        )
        log.info("")

        # ── Primary: exact linear-k (publication / benchmark metric) ──────────
        valid_mask = ~torch.isnan(trues_log)
        n_valid    = int(valid_mask.sum())
        if n_valid > 0:
            p_lin = torch.exp(preds_log[valid_mask])
            t_lin = torch.exp(trues_log[valid_mask])
            mae_k_linear   = (p_lin - t_lin).abs().mean().item()
            rmse_k_linear  = ((p_lin - t_lin) ** 2).mean().sqrt().item()
            t_np           = t_lin.cpu().numpy()
            # Use mean absolute deviation from the mean, matching
            # evaluate_multitask() and the MAD:MAE convention used in the
            # earlier Tier-1/Tier-2/Tier-3 result tables. Median absolute
            # deviation is robust, but it is not directly comparable to those
            # reported MAD:MAE values.
            mad_k_linear   = float(np.mean(np.abs(t_np - np.mean(t_np))))
            mad_mae_linear = (
                mad_k_linear / mae_k_linear if mae_k_linear > 0 else float("nan")
            )
            log.info(
                "  PRIMARY (linear-k exact)  MAE  = %.4f  [dielectric units]",
                mae_k_linear,
            )
            log.info(
                "  PRIMARY (linear-k exact)  RMSE = %.4f  [dielectric units]",
                rmse_k_linear,
            )
            log.info(
                "  PRIMARY (linear-k exact)  MAD  = %.4f  "
                "→ MAD:MAE = %.2f  (N=%d)",
                mad_k_linear, mad_mae_linear, n_valid,
            )
            log.info("─" * 68)
            if mad_mae_linear >= 2.5:
                log.info(
                    "  STATUS: ✓ PROJECT GOAL MET  "
                    "(MAD:MAE %.2f ≥ 2.5)", mad_mae_linear
                )
            elif mad_mae_linear >= 1.63:
                log.info(
                    "  STATUS: ✓ PAPER BENCHMARK MET  "
                    "(MAD:MAE %.2f ≥ 1.63) — push toward 2.5 goal",
                    mad_mae_linear,
                )
            else:
                log.info(
                    "  STATUS: ✗ BELOW PAPER BENCHMARK  "
                    "(MAD:MAE %.2f < 1.63) — check data count "
                    "and process encoder activation",
                    mad_mae_linear,
                )
        else:
            log.warning(
                "  No valid k_measured predictions on test set (n_valid=0). "
                "Check --rebuild_tier3 log for k_measured_log coverage."
            )
            mae_k_linear = rmse_k_linear = mad_k_linear = mad_mae_linear = float("nan")
            n_valid = 0

    else:
        # Non-log-transform fallback — should not be reached in normal Tier 3
        test_mae_log, test_rmse_log = trainer.evaluate(test_loader, cfg["target"])
        log.info(
            "  k_total (linear)  MAE  = %.4f  [dielectric units]", test_mae_log   # UNIFY-K
        )
        log.info("  k_total (linear)  RMSE = %.4f  [dielectric units]", test_rmse_log)  # UNIFY-K
        mae_k_linear = test_mae_log
        rmse_k_linear = test_rmse_log
        mad_k_linear = mad_mae_linear = float("nan")
        n_valid = 0

    # ── Multi-task table (all heads) ─────────────────────────────────────────
    # band_gap is NOT a primary metric for Tier 3. The model was trained for
    # k_total_log in Phase B2, and the band_gap head was only trained during
    # Phase A (500 steps). Evaluating band_gap on the same 2,024 DFT rows
    # used in Phase A training is evaluating on the training distribution.
    # If band_gap prediction is needed, add it as an aux target during Phase B2.

    # ── Final summary (linear-k, matches Tier 2 reporting format) ────────────
    log.info("─" * 68)
    log.info("TIER 3 FINAL SUMMARY  (linear-k, exact exp-space)")
    log.info(
        "  k_total  MAE=%.4f  RMSE=%.4f  MAD=%.4f  MAD:MAE=%.2f  N=%d",  # UNIFY-K
        mae_k_linear, rmse_k_linear, mad_k_linear, mad_mae_linear, n_valid,
    )
    log.info(
        "  paper benchmark: MAD:MAE ≥ 1.63 (44K JARVIS, no transfer)  |  "
        "project goal: MAD:MAE ≥ 2.5 (three-tier + process conditioning)"
    )

    log.info("─" * 68)
    log.info("RESIDUAL-PROC-DIAG SUMMARY: epoch-0 vs epoch-1 vs best checkpoint")
    log.info(
        "  epoch0 VAL : log_MAE=%.4f  linear_MAE=%.4f  MAD:MAE=%.2f",
        diag_epoch0_val.get("mae_log", float("nan")),
        diag_epoch0_val.get("mae_k", float("nan")),
        diag_epoch0_val.get("mad_mae_k", float("nan")),
    )
    log.info(
        "  epoch0 TEST: log_MAE=%.4f  linear_MAE=%.4f  MAD:MAE=%.2f",
        diag_epoch0_test.get("mae_log", float("nan")),
        diag_epoch0_test.get("mae_k", float("nan")),
        diag_epoch0_test.get("mad_mae_k", float("nan")),
    )
    if 'diag_epoch1_val' in locals() and diag_epoch1_val is not None:
        log.info(
            "  epoch1 VAL : log_MAE=%.4f  linear_MAE=%.4f  MAD:MAE=%.2f",
            diag_epoch1_val.get("mae_log", float("nan")),
            diag_epoch1_val.get("mae_k", float("nan")),
            diag_epoch1_val.get("mad_mae_k", float("nan")),
        )
    if 'diag_epoch1_test' in locals() and diag_epoch1_test is not None:
        log.info(
            "  epoch1 TEST: log_MAE=%.4f  linear_MAE=%.4f  MAD:MAE=%.2f",
            diag_epoch1_test.get("mae_log", float("nan")),
            diag_epoch1_test.get("mae_k", float("nan")),
            diag_epoch1_test.get("mad_mae_k", float("nan")),
        )
    log.info(
        "  BEST TEST  : log_MAE=%.4f  linear_MAE=%.4f  MAD:MAE=%.2f  best_epoch=%d",
        test_mae_log, mae_k_linear, mad_mae_linear, trainer.best_epoch,
    )
    if trainer.best_epoch <= 1:
        log.warning(
            "RESIDUAL-PROC-DIAG INTERPRETATION: best_epoch=%d. "
            "If epoch0/epoch1 metrics are close to BEST, the Tier-3 process residual "
            "is not yet adding measurable value; the result is mostly the fixed Tier-2 DFT baseline.",
            trainer.best_epoch,
        )
    else:
        log.info(
            "RESIDUAL-PROC-DIAG INTERPRETATION: best_epoch=%d. "
            "Tier-3 training improved validation after the initial checkpoint baseline; "
            "compare epoch0/epoch1/BEST test metrics to quantify the gain.",
            trainer.best_epoch,
        )

    out_path = REPORT_ROOT / "tier3_test_results.json"
    primary_metrics = {
        # Log-space (training diagnostic)
        "mae_log_k":          test_mae_log,
        "rmse_log_k":         test_rmse_log,
        "exp_mae_multiplier": (
            math.exp(test_mae_log) if cfg.get("log_transform") else float("nan")
        ),
        # Linear-k exact (FIX-T3-LINMAE — publication / benchmark metric)
        "mae_k_exact":       mae_k_linear,
        "rmse_k_exact":      rmse_k_linear,
        "mad_k_exact":       mad_k_linear,
        "mad_mae_k_exact":   mad_mae_linear,
        "n_test":             n_valid,
        # Canonical aliases for downstream comparison scripts
        "mae":                mae_k_linear,
        "rmse":               rmse_k_linear,
        "benchmark_scale":    "linear",
        "diagnostic_scale":   "log",
    }
    primary_metrics["residual_proc_diag"] = {
        "epoch0_val": diag_epoch0_val,
        "epoch0_test": diag_epoch0_test,
        "epoch1_val": diag_epoch1_val if 'diag_epoch1_val' in locals() else None,
        "epoch1_test": diag_epoch1_test if 'diag_epoch1_test' in locals() else None,
        "best_epoch": int(trainer.best_epoch),
        "best_val_mae_log": float(trainer.best_val_mae),
    }

    # FIX-4: restore multitask section in tier3_test_results.json.
    # evaluate_multitask is called here (training path) with _main_ds_t3.df
    # so row_indices are aligned with the reset-indexed dataset (iloc==loc).
    # Note: band_gap head was trained only during Phase A (500 steps on
    # df_structural), not Phase B/B2. Its N_Valid and MAE are informational
    # only — k_total_log remains the sole primary benchmark metric.
    mt_results_t3 = trainer.evaluate_multitask(
        test_loader, cfg, df_full=_main_ds_t3.df
    )
    trainer.print_multitask_results(mt_results_t3, split_name="TEST", tier_name="TIER 3")

    with open(out_path, "w") as f:
        json.dump({"primary": primary_metrics, "multitask": mt_results_t3}, f, indent=2)
    log.info("Tier 3 test results saved → %s", out_path)

    return CKPT_ROOT / "tier3_best.pt"


# ==============================================================================
# SECTION 9 -- MAIN ENTRY POINT  (was Section 8)
# ==============================================================================

def _tier_col_summary(df: pd.DataFrame, col: str) -> str:
    """One-line stat string for a numeric column: count, %, mean, std."""
    if col not in df.columns:
        return "column absent"
    s = df[col].dropna()
    if len(s) == 0:
        return "0 rows  (0.0%)"
    return (
        f"{len(s):,} rows  ({100*len(s)/len(df):.1f}%)"
        f"  mean={s.mean():.3g}  std={s.std():.3g}"
        f"  min={s.min():.3g}  max={s.max():.3g}"
    )


def log_tier_summary(df: pd.DataFrame, tier_num: int, tier_name: str):
    """
    Emit a comprehensive per-tier summary to the logger.
    Called once per tier at the end of extract_only / after build_tier*.
    Covers all targets, source breakdown, k_total thresholds, and process coverage.
    """
    W = 70
    log.info("=" * W)
    log.info(" TIER %d  %s  |  %d total rows", tier_num, tier_name.upper(), len(df))
    log.info("=" * W)

    # -- Structural coverage ---------------------------------------------------
    n_struct = int(df["has_structure"].sum()) if "has_structure" in df.columns \
               else int(df["atoms_dict"].notna().sum()) if "atoms_dict" in df.columns else 0
    log.info("  Structural rows (ALIGNN-ready) : %d  (%.1f%%)",
             n_struct, 100 * n_struct / max(len(df), 1))
    if tier_num == 3:
        n_proc = len(df) - n_struct
        log.info("  Process-only rows (no struct)  : %d  (%.1f%%)",
                 n_proc, 100 * n_proc / max(len(df), 1))

    log.info("")

    # -- Dielectric target columns (THE critical check) -----------------------
    log.info("  ── Dielectric targets ──────────────────────────────────")
    for col in ["k_total", "k_measured", "k_ionic", "k_elec"]:
        log.info("  %-14s  %s", col + ":", _tier_col_summary(df, col))

    # k_total threshold breakdown (Tier 1/2) and k_measured (Tier 3)
    k_col = "k_total"   # UNIFY-K: was conditional "k_total if tier<3 else k_measured"
    if k_col in df.columns:
        k = df[k_col].dropna()
        if len(k) > 0:
            log.info("  %s thresholds:", k_col)
            for thresh, label in [(3.9, "k>3.9 (above SiO2)"),
                                   (10,  "k>10  (coarse high-k)"),
                                   (25,  "k>25  (practical high-k)"),
                                   (35,  "k>35  (advanced high-k)"),
                                   (100, "k>100 (ultra-high-k / ferroelec)")]:
                n = int((k > thresh).sum())
                log.info("    %-28s %d  (%.1f%% of k rows)", label, n,
                         100 * n / len(k))

    # Tier 2 critical line: how many rows are available for k_total training
    if tier_num == 2:
        n_trainable = int(df["k_total"].notna().sum()) if "k_total" in df.columns else 0
        log.info("")
        log.info("  ★ TIER 2 TRAINING DATASET SIZE (k_total.notna()): %d rows",
                 n_trainable)
        log.info("    (target ≥ 3,000 for robust fine-tuning)")
        if n_trainable < 1000:
            log.warning(
                "    !! BELOW MINIMUM -- k_total head will be undertrained."
                "    Check MP dielectric extraction (see FIX-DIELECTRIC-SCALAR)."
            )

    log.info("")

    # -- Other property targets ------------------------------------------------
    log.info("  ── Other property targets ──────────────────────────────")
    for col in ["band_gap", "formation_energy_per_atom", "e_above_hull",
                "J_g_A_cm2", "E_BD_MV_cm"]:
        if col in df.columns:
            log.info("  %-28s %s", col + ":", _tier_col_summary(df, col))

    log.info("")

    # -- Source / functional breakdown ----------------------------------------
    if "source" in df.columns:
        log.info("  ── Source breakdown ────────────────────────────────────")
        for src, cnt in df["source"].value_counts().items():
            # k_total coverage per source
            k_in_src = df.loc[df["source"] == src, "k_total"].notna().sum() \
                       if "k_total" in df.columns else 0
            log.info("  %-22s %6d rows   k_total non-null: %d  (%.1f%%)",
                     src, cnt, k_in_src, 100 * k_in_src / max(cnt, 1))
        log.info("")

    if "dft_functional" in df.columns:
        log.info("  ── DFT functional breakdown ────────────────────────────")
        for fn, cnt in df["dft_functional"].value_counts().items():
            log.info("  %-14s  %d  (%.1f%%)", fn, cnt, 100 * cnt / len(df))
        log.info("")

    # -- Process/stack context coverage (Tier 3) ------------------------------
    if tier_num == 3:
        log.info("  ── Process/stack context coverage ──────────────────────")
        proc_cols  = [c for c in PROCESS_PARAMS_FEATURES["numerical"]
                      + list(PROCESS_PARAMS_FEATURES["categorical"].keys())
                      if c in df.columns]
        stack_cols = [c for c in STACK_CONTEXT_FEATURES["numerical"]
                      + list(STACK_CONTEXT_FEATURES["categorical"].keys())
                      if c in df.columns]
        for col in proc_cols + stack_cols:
            n_filled = int(df[col].notna().sum())
            log.info("  %-28s %d  (%.1f%%)", col + ":", n_filled,
                     100 * n_filled / max(len(df), 1))
        log.info("")

    log.info("=" * W)


def log_full_dataset_summary(
    df_tier1: pd.DataFrame,
    df_tier2: pd.DataFrame,
    df_tier3: pd.DataFrame,
):
    """
    Print a comprehensive summary for all three tiers after dataset generation.
    Called at the end of extract_only and after Step 5 in the main pipeline.
    """
    bar = "█" * 70
    log.info("\n%s", bar)
    log.info("  FULL DATASET GENERATION SUMMARY  (end of Step 5)")
    log.info("%s\n", bar)

    tier_names = {1: "Foundation Pretrain", 2: "Domain Fine-tune", 3: "Project Fine-tune"}
    tiers      = {1: df_tier1, 2: df_tier2, 3: df_tier3}

    for t in [1, 2, 3]:
        df = tiers[t]
        if df is None or len(df) == 0:
            log.warning("  Tier %d (%s): EMPTY OR NOT BUILT", t, tier_names[t])
        else:
            log_tier_summary(df, t, tier_names[t])

    # -- Cross-tier k_total summary (the key diagnostic table) ----------------
    log.info("  ┌─────────────────────────────────────────────────────────────┐")
    log.info("  │  k_total AVAILABILITY ACROSS TIERS  (Tier 2 critical check) │")
    log.info("  ├──────────────┬────────────┬────────────┬────────────────────┤")
    log.info("  │ Tier         │ Total rows │ k_total    │ k_total > 10       │")
    log.info("  ├──────────────┼────────────┼────────────┼────────────────────┤")
    for t, df, nm in [(1, df_tier1, "Tier 1 Found."),
                      (2, df_tier2, "Tier 2 Domain"),
                      (3, df_tier3, "Tier 3 Proj.")]:
        if df is None or len(df) == 0:
            log.info("  │ %-12s │ %-10s │ %-10s │ %-18s │", nm, "N/A", "N/A", "N/A")
            continue
        n_total = len(df)
        k_col   = "k_total"   # UNIFY-K: was "k_total if t<3 else k_measured"
        k_avail = int(df[k_col].notna().sum()) if k_col in df.columns else 0
        k_gt10  = int((df[k_col].dropna() > 10).sum()) if k_col in df.columns else 0
        pct     = 100 * k_avail / max(n_total, 1)
        flag    = "  ← TARGET ≥3K" if t == 2 and k_gt10 < 3000 else \
                  "  ✓ sufficient" if t == 2 and k_gt10 >= 3000 else ""
        log.info("  │ %-12s │ %10d │ %10d │ %6d  (%4.1f%%)%s │",
                 nm, n_total, k_avail, k_gt10, pct, flag)
    log.info("  └──────────────┴────────────┴────────────┴────────────────────┘")
    log.info("")
    log.info("%s\n", bar)


def run_tier_evaluate(
    tier:             int,
    df:               pd.DataFrame,
    checkpoint_path:  Path,
    ablate_context:   bool = False,
):
    """
    Load a saved checkpoint and re-run the full test evaluation without
    retraining.  Useful when:
      - Training completed but the evaluation block crashed with an error
      - You fixed a bug and want to re-validate the best checkpoint
      - You want to compare multiple saved checkpoints head-to-head

    The test split is reproduced exactly because get_stratified_split uses
    a fixed seed (42) and the same DataFrame ordering from the HDF5 cache.
    As long as you haven't rebuilt the HDF5 the test rows are identical to
    those held out during training.

    Usage:
      python highk_alignn_train_v4_5.py \\
          --mode tier1_evaluate \\
          --weights checkpoints/tier1_best.pt

      python highk_alignn_train_v4_5.py \\
          --mode tier2_evaluate \\
          --weights checkpoints/tier2_best.pt

      python highk_alignn_train_v4_5.py \\
          --mode tier3_evaluate \\
          --weights checkpoints/tier3_best.pt
    """
    # FIX-T3-2: guard extended from (1,2) to (1,2,3)
    if tier not in (1, 2, 3):
        raise ValueError(f"run_tier_evaluate supports tiers 1, 2, 3 — got {tier}")

    tier_label = f"TIER {tier}"
    log.info("=" * 70)
    log.info(" %s -- Checkpoint Evaluation (no retraining)", tier_label)
    log.info(" Checkpoint : %s", checkpoint_path)
    log.info(" Rows in df : %d", len(df))
    log.info("=" * 70)

    # ── Preprocessing: identical to training ─────────────────────────────────
    if tier == 1:
        cfg = TIER1_TRAIN_CONFIG
        df_eval = df.copy()

        # Replicate Tier 1 log-transform aux preprocessing
        for log_col, src_col in cfg.get("log_transform_aux", {}).items():
            if src_col in df_eval.columns:
                src_numeric = pd.to_numeric(df_eval[src_col], errors="coerce")
                mask = src_numeric.notna()
                df_eval[log_col] = np.nan
                df_eval.loc[mask, log_col] = np.log(
                    src_numeric[mask].clip(lower=0.1)
                )
                log.info("Aux log transform: %s → %s  (%d non-null rows)",
                         src_col, log_col, int(mask.sum()))

        target_col = cfg["target"]             # formation_energy_per_atom
        task_names = [target_col] + cfg["aux_targets"]
        ckpt_prefix = "tier1"

    elif tier == 2:
        cfg = TIER2_TRAIN_CONFIG
        df_eval = df[pd.to_numeric(df["k_total"], errors="coerce").notna()].copy()
        log.info("Tier 2 rows with k_total: %d", len(df_eval))

        if cfg.get("log_transform", False):
            orig_col = cfg["log_original_col"]
            log_col  = cfg["target"]
            k_num    = pd.to_numeric(df_eval[orig_col], errors="coerce")
            df_eval[log_col] = np.log(k_num.clip(lower=0.1))
            log.info("Log transform: %s → %s  range=[%.3f, %.3f]",
                     orig_col, log_col,
                     df_eval[log_col].min(), df_eval[log_col].max())

        target_col = cfg["target"]             # k_total_log
        task_names = [target_col] + cfg["aux_targets"]
        ckpt_prefix = "tier2"

    else:  # tier == 3  (FIX-T3-2)
        cfg = TIER3_TRAIN_CONFIG
        df_eval = df.copy()

        # FIX: apply log-transform FIRST (mirrors run_tier3_finetune)
        # THEN filter by k_total_log.notna(). The HDF5 cache has k_total but
        # not k_total_log - the transform must be applied here
        if cfg.get("log_transform", False):
            src_col = cfg["log_original_col"]
            log_col = cfg["target"]
            k_num   = pd.to_numeric(df_eval[src_col], errors="coerce")
            valid_k = k_num.notna()
            df_eval[log_col] = np.nan
            df_eval.loc[valid_k, log_col] = np.log(k_num[valid_k].clip(lower=0.1))
            log.info("tier3_evaluate: log transform %s → %s  (%d rows)", src_col, log_col, int(valid_k.sum()))

        df_kvalid = df_eval[df_eval[cfg["target"]].notna()].copy()
        log.info("Tier 3 rows with %s : %d (matches training df_phase_b)", cfg["target"], len(df_kvalid))

        # FIX : Filter to structural rows BEFORE stratified split.
        # HighKGraphDataset silently drops rows without atoms_dict. If we split
        # on 193 rows but only 82 have structures, the test set ends up with ~13 rows
        # instead of ~30. we must split on the rows that will actually be used by the 
        # dataloader
        #
        # Note : In training, experimental rows have atoms_dict imputed from DFT
        # donors. If the HDF5 cache has these imputed values all 193 rows will pass.
        # If not, only the 82 DFT rows will pass - which is fine, we just need to split
        # on the correct set of rows
        if "atoms_dict" in df_kvalid.columns:
            df_eval = df_kvalid[df_kvalid["atoms_dict"].notna()].copy()
            n_dropped = len(df_kvalid) - len(df_eval)
            if n_dropped > 0:
                log.info(
                    "Tier 3 %d k-valid rows without atoms_dict dropped "
                    "(split will be on %d structural k-valid rows)",
                    n_dropped, len(df_eval),
                )
        else:
            df_eval = df_kvalid

        target_col  = cfg["target"]
        task_names  = [target_col] + cfg["aux_targets"]
        ckpt_prefix = "tier3"

    # ── Rebuild the exact same test split (seed=42, same ratios) ─────────────
    _, _, test_loader, _, test_dataset = build_dataloader(
        df         = df_eval,
        target_col = target_col,
        aux_cols   = [t for t in task_names if t != target_col],
        train_frac = cfg["train_ratio"],
        val_frac   = cfg["val_ratio"],
        batch_size = cfg["batch_size"],
    )
    log.info("Test loader: %d batches  (seed=42 → same split as training)",
             len(test_loader))

    # ── Build model and load checkpoint ──────────────────────────────────────
    model = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names)
    model.fit_encoder_stats(df_eval)

    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_epoch = ckpt.get("epoch", "?")
    saved_mae   = ckpt.get("val_mae", float("nan"))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    log.info("Loaded checkpoint: epoch=%s  saved_val_MAE=%.4f", saved_epoch, saved_mae)

    # ── Trainer (eval-only — no optimizer needed) ─────────────────────────────
    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix=ckpt_prefix,
                            ablate_context=ablate_context)

    # ── Primary metric ────────────────────────────────────────────────────────
    test_mae_log, test_rmse_log, preds_t, trues_t = trainer.evaluate(
        test_loader, target_col, return_preds=True
    )

    log.info("")
    log.info("─" * 68)
    log.info("  %s TEST RESULTS  (checkpoint: ep %s)", tier_label, saved_epoch)
    log.info("─" * 68)

    if tier == 2 and cfg.get("log_transform", False):
        valid = ~torch.isnan(trues_t)
        mae_k_exact  = (torch.exp(preds_t[valid]) -
                        torch.exp(trues_t[valid])).abs().mean().item()
        rmse_k_exact = ((torch.exp(preds_t[valid]) -
                         torch.exp(trues_t[valid])) ** 2).mean().sqrt().item()
        log.info("  Diagnostic log-space  MAE  = %.4f  [log(k) units]", test_mae_log)
        log.info("  Diagnostic log-space  RMSE = %.4f", test_rmse_log)
        log.info("  exp(MAE)                  = %.4f×  → ±%.1f%% average relative error",
                 math.exp(test_mae_log), (math.exp(test_mae_log) - 1) * 100)
        log.info("  PRIMARY benchmark (linear-k) MAE  = %.4f  [dielectric units, exact]", mae_k_exact)
        log.info("  PRIMARY benchmark (linear-k) RMSE = %.4f  [dielectric units]", rmse_k_exact)
        log.info("  Benchmark scale for publication: LINEAR-K")
        log.info("  ALIGNN paper    ≈ 0.81  [linear k, JARVIS DFPT]")
    elif tier == 3:
        # Tier 3 primary is k_measured in linear space
        valid = ~torch.isnan(trues_t)
        if valid.sum() > 0:
           mae_k_exact = (torch.exp(preds_t[valid]) -
                          torch.exp(trues_t[valid])).abs().mean().item()
           rmse_k_exact = ((torch.exp(preds_t[valid]) -
                            torch.exp(trues_t[valid]))  ** 2).mean().sqrt().item()
        else:
            mae_k_exact = rmse_k_exact = float("nan")
        log.info(" Diagnostic log-space MAE = %.4f [log(k) units]", test_mae_log)
        log.info(" Diagnostic log-space RMSE = %.4f ",test_rmse_log )
        log.info(" exp (MAE) = %.4fx -> %.1f%% avg relative error",
                 math.exp(test_mae_log) if not math.isnan(test_mae_log) else float("nan"),
                 ((math.exp(test_mae_log) -1) *100) if not math.isnan(test_mae_log) else float("nan"))
        if not math.isnan(mae_k_exact):
            log.info(" PRIMARY (linear - k exact) MAE = %.4f [dielectric units]",mae_k_exact)
            log.info(" PRIMARY (linear - k exact) RMSE = %.4f [dielectric units]",rmse_k_exact)
        else:
            log.info(" PRIMARY (linear - k exact) MAE = N/A (no valid samples)")
            log.info(" PRIMARY (linear - k exact) RMSE = N/A")
    else:
        log.info("  MAE  = %.4f", test_mae_log)
        log.info("  RMSE = %.4f", test_rmse_log)

    # ── Full multitask table ──────────────────────────────────────────────────
    mt_results = trainer.evaluate_multitask(
        test_loader, cfg, df_full=test_dataset.df
    )
    trainer.print_multitask_results(
        mt_results, split_name="TEST", tier_name=tier_label
    )

    # ── Tier 3 MAD:MAE summary block ─────────────────────────────────────────
    if tier == 3 and "k_measured" in mt_results and mt_results["k_measured"]["n"] > 0:
        km = mt_results["k_measured"]
        log.info(
            "k_measured  MAD=%.2f  MAE=%.4f  MAD:MAE=%.2f  "
            "(paper: 1.63 @ 44K no transfer | goal: ≥ 2.5 with three-tier transfer)",
            km["mad"], km["mae"], km["mad_mae_ratio"],
        )

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_path = REPORT_ROOT / f"tier{tier}_evaluate_results.json"
    primary  = {"mae": test_mae_log, "rmse": test_rmse_log,
                "mae_log_k": test_mae_log,
                "rmse_log_k": test_rmse_log,
                "checkpoint_epoch": saved_epoch,
                "checkpoint_val_mae": saved_mae,
                "benchmark_scale": "linear",
                "diagnostic_scale": "log"}
    if tier == 2 and cfg.get("log_transform", False):
        primary.update({"mae_linear_k": mae_k_exact,
                        "rmse_linear_k": rmse_k_exact,
                        "mae": mae_k_exact,
                        "rmse": rmse_k_exact,
                        "exp_mae_multiplier": math.exp(test_mae_log)})
    with open(out_path, "w") as f:
        json.dump({"primary": primary, "multitask": mt_results}, f, indent=2)
    log.info("Results saved → %s", out_path)
    log.info("─" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="High-k ALIGNN Three-Tier Scalable Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "extract_only", "tier1_pretrain",
                 "tier2_finetune", "tier3_finetune", "dataset_stats",
                 "tier1_evaluate", "tier2_evaluate", "tier3_evaluate"],
        default="full_pipeline",
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Re-download raw data even if cache exists")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Rebuild ALL tier datasets even if they exist. "
                             "WARNING: triggers the full 30-min Tier 1 cross-source dedup. "
                             "Only use when Tier 1 or Tier 2 source data has changed. "
                             "Use --rebuild_tier3 instead when only process_db.csv changed.")
    parser.add_argument("--rebuild_tier3", action="store_true",
                        help="FIX-T3-4: Force rebuild ONLY the Tier 3 HDF5 cache. "
                             "Tier 1 and Tier 2 are always loaded from their existing "
                             "caches -- skips the 30-min Tier 1 cross-source dedup. "
                             "Use this whenever process_db.csv has been updated.")
    parser.add_argument("--skip_cross_dedup", action="store_true",
                        help="FIX4: skip MP-JARVIS structural dedup (faster, less clean)")
    parser.add_argument(
        "--ablate_context", action="store_true",
        help=(
            "Obs 3 ablation: force proc_context=stack_context=None in all "
            "forward calls regardless of data.  Runs the full pipeline with "
            "identical data but zero context-branch contribution -- use to "
            "measure context branches' effect on Tier 3 k_measured MAE."
        ),
    )
    args = parser.parse_args()

    # Distributed init — must be first, before any CUDA calls
    init_dist()

    log.info("HighK ALIGNN Pipeline  mode=%s  rank=%d/%d",
             args.mode, _DIST["rank"], _DIST["world"])
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log.info("  GPU %d: %s  VRAM %.1f GB",
                     i,
                     torch.cuda.get_device_name(i),
                     torch.cuda.get_device_properties(i).total_memory / 1e9)
    else:
        log.info("  Device: CPU only")
    if args.ablate_context:
        log.info(
            "ABLATION MODE: context branches forced to None -- "
            "model behaves as pure ALIGNN regardless of tier."
        )

    # -- Initialise components -------------------------------------------------
    extractor = DatasetExtractor()
    builder   = TierDatasetBuilder()

    if args.mode == "dataset_stats":
        # Load from existing HDF5 files and print stats (no rebuild)
        dfs = {}
        for t, nm in [(1,"Foundation"),(2,"Domain"),(3,"Project")]:
            p = builder.TIER_PATHS[t]
            if p.exists():
                dfs[t] = pd.read_hdf(p, key="data")
                log.info("Loaded Tier %d from %s (%d rows)", t, p, len(dfs[t]))
            else:
                dfs[t] = pd.DataFrame()
                log.warning("Tier %d HDF5 not found at %s -- run extract_only first", t, p)
        log_full_dataset_summary(dfs.get(1, pd.DataFrame()),
                                 dfs.get(2, pd.DataFrame()),
                                 dfs.get(3, pd.DataFrame()))
        return

    # -- Evaluate-only modes (no retraining, load checkpoint + test) ----------
    if args.mode in ["tier1_evaluate", "tier2_evaluate", "tier3_evaluate"]:
        # FIX-T3-2: tier3_evaluate added alongside tier1/tier2 paths.
        tier_num     = {"tier1_evaluate": 1,
                        "tier2_evaluate": 2,
                        "tier3_evaluate": 3}[args.mode]
        ckpt_default = CKPT_ROOT / f"tier{tier_num}_best.pt"
        ckpt_path    = Path(args.weights) if args.weights else ckpt_default

        if not ckpt_path.exists():
            log.error(
                "Checkpoint not found: %s\n"
                "  Pass the path explicitly with --weights <path>", ckpt_path
            )
            return
        
        # FIX support --rebuild_tier3 for tier3_evaluate.
        # Rebuilds Tier3 dataset AND applies structure imputation before loading the HDF5 cache
        # This ensures experimental rows have atoms_dict for graph construction, matching
        # the training pipeline.
        if args.mode == "tier3_evaluate" and getattr(args, "rebuild_tier3", False):
            log.info("tier3_evaluate: rebuilding Tier 3 dataset ( --rebuild_tier3)")
            t1_hdf5 = builder.TIER_PATHS[1]
            t2_hdf5 = builder.TIER_PATHS[2]
            if not t1_hdf5.exists():
                log.error(" Tier1 HDF5 is not found at %s. Run tier1_pretrain first.",t1_hdf5)
                return
            if not t2_hdf5.exists():
                log.error(" Tier2 HDF5 is not found at %s. Run tier2_finetune first.",t2_hdf5)
                return
            df_tier1=pd.read_hdf(t1_hdf5, key="data")
            df_tier2=pd.read_hdf(t2_hdf5, key="data")
            df_exp = extractor.load_experimental_process_db()
            df_tier3=builder.build_tier3(df_tier2, df_exp, force_rebuild=True)

            # FIX : apply structure imputation (same as run_tier3_finetune)
            # The HDF5 cache doesnt have imputed atoms_dict - its done in-memory in run_tier3_finetune
            # we must replicate this here.
            # FIX: Use df_tier2 as donor pool (not df_structural)
            # The training code builds df_done_pool_wide from df_tier2 which
            # contains all high-k familes (Hf, Zr, Al, Ti, Ta, Sr, La, Y, Ba,
            # Nb, Ga, In, Sc, Ce, Pr, Nd oxides). Using df_structural
            # (Hfo2 family only) causes 95/1120 matched vs 120/120 in training
            df_structural = df_tier3[df_tier3["atoms_dict"].notna()].copy()
            df_process_only = df_tier3[df_tier3["atoms_dict"].isna()].copy()
            if len(df_process_only) > 0 and len(df_structural) > 0:
                df_donor_pool_wide = df_tier2[
                    df_tier2["formula"].apply(
                        lambda f: isinstance(f, str) and "O" in f and
                                any(el in f for el in HIGH_K_DONOR_ELEMENTS)
                    )
                    ].copy()
                log.info("Donor Pool from df_tier2: %d high-k oxide rows", len(df_donor_pool_wide))
                df_imputed, df_unmatched = _impute_structures(
                    df_process_only, df_donor_pool_wide
                )
                if len(df_imputed) > 0:
                    # Merge imputed rows back
                    df_tier3 = pd.concat(
                        [df_structural, df_imputed, df_unmatched], ignore_index=True
                    )

            log.info("Tier 3 rebuilt: %d rows (k_measured non-null : %d)",
                    len(df_tier3), int(df_tier3["k_measured"].notna().sum()))
            # use rebuilt data directly (with imputed atoms_dict) instead of
            # loading from the old HDF5 cache which lacks imputed structures.
            df_eval = df_tier3
            log.info("Using rebuilt Tier3 dataset (with structure imputation)")
          
        else:
            # Load HDF5 cache for the relevant tier
            tier_hdf5 = builder.TIER_PATHS[tier_num]
            if not tier_hdf5.exists():
                log.error(
                    "Tier %d HDF5 not found at %s.\n"
                    "  Run --mode tier%d_pretrain (or tier%d_finetune) first to build the cache.",
                    tier_num, tier_hdf5, tier_num, tier_num
                )
                return

            log.info("Loading Tier %d dataset from cache: %s", tier_num, tier_hdf5)
            df_eval = pd.read_hdf(tier_hdf5, key="data")
            log.info("  %d rows loaded", len(df_eval))

        run_tier_evaluate(
            tier            = tier_num,
            df              = df_eval,
            checkpoint_path = ckpt_path,
            ablate_context  = args.ablate_context,
        )
        return

    # -- Extract all raw datasets ----------------------------------------------
    if args.mode in ["full_pipeline", "extract_only",
                     "tier1_pretrain", "tier2_finetune", "tier3_finetune"]:

        log.info("-" * 60)
        log.info("Step 1/5: Extracting JARVIS-DFT full dataset (~55K entries)")
        df_jarvis = extractor.pull_jarvis_dft(force_refresh=args.force_refresh)

        log.info("-" * 60)
        log.info("Step 2/5: Extracting Materials Project (~60-70K oxide entries)")
        df_mp = extractor.pull_materials_project(force_refresh=args.force_refresh)

        log.info("-" * 60)
        log.info("Step 3/5: Extracting QM9 (~130K molecules)")
        df_qm9 = extractor.pull_qm9(force_refresh=args.force_refresh)

        log.info("-" * 60)
        log.info("Step 4/5: Loading experimental process database")
        df_exp = extractor.load_experimental_process_db()

    if args.mode == "extract_only":
        # Step 5: build all three tier datasets from the freshly extracted raw data.
        # force_rebuild=True is ALWAYS used here regardless of --force_rebuild flag.
        # Rationale: extract_only's sole purpose is to produce fresh tier datasets
        # from the just-extracted raw data.  Loading a stale HDF5 cache would
        # silently discard the fresh df_mp (including the corrected k_total values
        # from FIX-DIELECTRIC-SCALAR) and reproduce the original 755-row bug.
        log.info("-" * 60)
        log.info("Step 5/5: Building three-tier datasets (force_rebuild=True)")
        df_tier1 = builder.build_tier1(df_jarvis, df_mp, df_qm9,
                                        force_rebuild=True,            # always in extract_only
                                        skip_cross_dedup=args.skip_cross_dedup)
        log.info(
            "  Tier 1 built: %d rows | k_total non-null: %d (%.1f%%) "
            "| by source: JARVIS=%d MP=%d QM9=%d",
            len(df_tier1),
            int(df_tier1["k_total"].notna().sum()),
            100 * df_tier1["k_total"].notna().mean(),
            int((df_tier1["source"] == "JARVIS-DFT").sum()),
            int(df_tier1["source"].str.startswith("MaterialsProject").sum()),
            int((df_tier1["source"] == "QM9").sum()),
        )

        df_tier2 = builder.build_tier2(df_tier1, force_rebuild=True)  # always in extract_only
        log.info(
            "  Tier 2 built: %d rows | k_total non-null: %d (%.1f%%)"
            " | ★ TRAINING DATASET SIZE (k_total.notna()): %d",
            len(df_tier2),
            int(df_tier2["k_total"].notna().sum()),
            100 * df_tier2["k_total"].notna().mean(),
            int(df_tier2["k_total"].notna().sum()),
        )

        df_tier3 = builder.build_tier3(df_tier2, df_exp, force_rebuild=True)
        log.info(
            "  Tier 3 built: %d rows | k_measured non-null: %d (%.1f%%)",
            len(df_tier3),
            int(df_tier3["k_measured"].notna().sum()),
            100 * df_tier3["k_measured"].notna().mean(),
        )

        log_full_dataset_summary(df_tier1, df_tier2, df_tier3)
        return

    # -- Build three-tier dataset ----------------------------------------------
    # FIX-T3-4: Mode-aware rebuild routing.
    #
    # --force_rebuild is scoped to tiers that can legitimately have changed
    # given the current mode.  This prevents the 30-min Tier 1 cross-source
    # dedup from being triggered unnecessarily during tier3_finetune runs.
    #
    # --rebuild_tier3 is the fast path: forces ONLY Tier 3 to rebuild.
    # Tier 1 and Tier 2 always load from their existing HDF5 caches.
    #
    # Rebuild matrix:
    #   mode              Tier 1 rebuild     Tier 2 rebuild     Tier 3 rebuild
    #   full_pipeline     force_rebuild      force_rebuild      force_rebuild | rebuild_tier3
    #   tier1_pretrain    force_rebuild      force_rebuild      force_rebuild | rebuild_tier3
    #   tier2_finetune    NEVER              force_rebuild      force_rebuild | rebuild_tier3
    #   tier3_finetune    NEVER              NEVER              force_rebuild | rebuild_tier3
    #
    _mode = args.mode
    _t3_rebuild  = getattr(args, "rebuild_tier3", False)
    _t1_rebuild  = args.force_rebuild and _mode in [
        "full_pipeline", "tier1_pretrain",
    ]
    _t2_rebuild  = args.force_rebuild and _mode in [
        "full_pipeline", "tier1_pretrain", "tier2_finetune",
    ]
    _t3_rebuild_final = args.force_rebuild or _t3_rebuild

    log.info("-" * 60)
    log.info("Step 5/5: Building three-tier dataset")
    log.info(
        "  Rebuild flags  tier1=%s  tier2=%s  tier3=%s  "
        "(mode=%s  --force_rebuild=%s  --rebuild_tier3=%s)",
        _t1_rebuild, _t2_rebuild, _t3_rebuild_final,
        _mode, args.force_rebuild, _t3_rebuild,
    )

    df_tier1 = builder.build_tier1(df_jarvis, df_mp, df_qm9,
                                    force_rebuild=_t1_rebuild,
                                    skip_cross_dedup=args.skip_cross_dedup)
    df_tier2 = builder.build_tier2(df_tier1,
                                    force_rebuild=_t2_rebuild)
    df_tier3 = builder.build_tier3(df_tier2, df_exp,
                                    force_rebuild=_t3_rebuild_final)

    log_full_dataset_summary(df_tier1, df_tier2, df_tier3)

    # -- Run training ----------------------------------------------------------
    if args.mode in ["full_pipeline", "tier1_pretrain"]:
        t1_ckpt = run_tier1_pretrain(df_tier1,
                                     ablate_context=args.ablate_context)
        if args.mode == "tier1_pretrain":
            return

    if args.mode in ["full_pipeline", "tier2_finetune"]:
        t1_ckpt = (
            Path(args.weights) if args.weights
            else CKPT_ROOT / "tier1_best.pt"
        )
        if not t1_ckpt.exists():
            log.error("Tier 1 checkpoint not found at %s. "
                      "Run tier1_pretrain first.", t1_ckpt)
            return
        t2_ckpt = run_tier2_finetune(df_tier2, t1_ckpt,
                                     ablate_context=args.ablate_context)
        if args.mode == "tier2_finetune":
            return

    if args.mode in ["full_pipeline", "tier3_finetune"]:
        t2_ckpt = (
            Path(args.weights) if args.weights
            else CKPT_ROOT / "tier2_best.pt"
        )
        if not t2_ckpt.exists():
            log.error("Tier 2 checkpoint not found at %s. "
                      "Run tier2_finetune first.", t2_ckpt)
            return
        run_tier3_finetune(df_tier3, t2_ckpt,
                           ablate_context=args.ablate_context,
                           df_tier2=df_tier2)

    log.info("Pipeline complete. Final model: %s/tier3_best.pt", CKPT_ROOT)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_dist()
