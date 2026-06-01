# -*- coding: utf-8 -*-
"""
==============================================================================
 High-k Dielectric Discovery -- Three-Tier Scalable ALIGNN Training Pipeline
 Version 3.0  (Production -- frozen, with MP API hotfixes)
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
TIER2_CATIONS   = {                       # expanded high-k space
    "Hf", "Zr", "Ti", "La", "Ce", "Pr", "Nd", "Gd", "Dy", "Y", "Lu",
    "Al", "Ga", "In", "Si", "Ge", "Sn", "Nb", "Ta", "W", "Mo",
    "Ba", "Sr", "Ca", "Mg",
}

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
    scheduler      = "onecycle",
    loss           = "mse",
    target         = "formation_energy_per_atom",   # primary pretrain target
    aux_targets    = ["band_gap", "k_total"],        # auxiliary multi-task
    train_ratio    = 0.80,
    val_ratio      = 0.10,
    test_ratio     = 0.10,
)

TIER2_TRAIN_CONFIG = dict(
    epochs         = 150,
    batch_size     = 32,
    learning_rate  = 2e-4,
    weight_decay   = 1e-5,
    scheduler      = "cosine",
    loss           = "mse",
    target         = "k_total",
    aux_targets    = ["band_gap", "e_above_hull"],
    train_ratio    = 0.80,
    val_ratio      = 0.10,
    test_ratio     = 0.10,
    freeze_layers  = 2,    # freeze first 2 ALIGNN layers during early fine-tune
)

TIER3_TRAIN_CONFIG = dict(
    epochs         = 100,
    batch_size     = 16,
    learning_rate  = 5e-5,
    weight_decay   = 1e-5,
    scheduler      = "cosine",
    loss           = "mse",
    target         = "k_measured",
    aux_targets    = ["band_gap", "J_g_A_cm2", "E_BD_MV_cm"],
    train_ratio    = 0.70,
    val_ratio      = 0.15,
    test_ratio     = 0.15,
    freeze_layers  = 0,    # unfreeze all for final fine-tune
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
    "k_measured":                 1.63,   # JARVIS-DFT ε DFPT elec+ionic (Tier 3 head)
    "k_total":                    1.63,   # same benchmark -- DFT ionic+elec total (Tier 1/2 head)
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
        "anneal_ambient": ["N2", "O2", "forming_gas", "vacuum"],
        "precursor_type": ["TDMA-Hf", "HfCl4", "TEMAZ", "TDMAZ", "other"],
        "oxidant_type":   ["H2O", "O3", "O2_plasma", "other"],
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

        # Rename experimental columns to unified schema if needed
        col_renames = {
            "k_measured":   "k_measured",
            "band_gap_eV":  "band_gap",
            "J_g_A_cm2":    "J_g_A_cm2",
            "E_BD_MV_cm":   "E_BD_MV_cm",
        }
        df = df.rename(columns={k: v for k, v in col_renames.items() if k in df})

        # For experimental entries the measured value IS the total dielectric constant.
        # Populate k_total so Tier 2/3 configs that reference "k_total" work correctly.
        if "k_measured" in df.columns:
            df["k_total"] = df["k_measured"]
        else:
            df["k_total"] = np.nan

        # Row hash for dedup
        df["row_hash"] = df.apply(
            lambda r: hashlib.md5(
                f"EXP_{r.get('doi','?')}_{r.get('material','?')}_{r.get('ald_substrate_temp_C','?')}".encode()
            ).hexdigest()[:12],
            axis=1,
        )

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
        """Convert to float, returning None on failure."""
        if x is None or x == "na" or x == "":
            return None
        try:
            v = float(x)
            return None if (np.isnan(v) or np.isinf(v)) else v
        except Exception:
            return None


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

        # Must contain at least one Tier 2 high-k cation
        def has_tier2_cation(formula):
            if not isinstance(formula, str):
                return False
            return any(el in formula for el in TIER2_CATIONS)

        df_struct["_has_t2_cation"] = df_struct["formula"].apply(has_tier2_cation)
        df_cation = df_struct[df_struct["_has_t2_cation"]].copy()
        log.info("  k_total audit post-cation filter  | rows=%d  k_total=%d",
                 len(df_cation), int(df_cation["k_total"].notna().sum()))

        # Band gap > 1 eV (exclude metals)
        # Allow NaN (some entries don't have gap computed -- keep them)
        df_cation = df_cation[
            df_cation["band_gap"].isna() | (df_cation["band_gap"] > 1.0)
        ].copy()
        log.info("  k_total audit post-bandgap filter | rows=%d  k_total=%d",
                 len(df_cation), int(df_cation["k_total"].notna().sum()))

        # If k_total is present, must be > 10.
        has_k    = df_cation["k_total"].notna()
        valid_k  = df_cation["k_total"] > 10.0
        df_tier2 = df_cation[~has_k | valid_k].copy()
        log.info("  k_total audit post-k>10 filter    | rows=%d  k_total=%d  "
                 "(★ this is the Tier 2 training dataset size)",
                 len(df_tier2), int(df_tier2["k_total"].notna().sum()))

        df_tier2 = df_tier2.drop(columns=["_has_t2_cation"], errors="ignore")
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
        def is_hfo2_family(formula):
            if not isinstance(formula, str):
                return False
            has_hf = "Hf" in formula
            has_o  = "O" in formula
            # Include pure HfO2, HZO, HfSiO, HfAlO, HfLaO families
            return has_hf and has_o

        df_hf = df_tier2[df_tier2["formula"].apply(is_hfo2_family)].copy()
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

            df_tier3 = pd.concat(
                [df_hf, df_exp_aligned[df_hf.columns]],
                ignore_index=True, sort=False
            )
        else:
            df_tier3 = df_hf.copy()

        # Final dedup
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
        """Return row[col] if present and non-NaN, else None."""
        try:
            v = row[col]
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
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
        row     = self.df.iloc[row_idx]

        # -- Parse atoms ---------------------------------------------------
        try:
            atoms_dict = json.loads(row["atoms_dict"])
            j_atoms    = JAtoms.from_dict(atoms_dict)
        except Exception as e:
            log.debug("Graph construction failed for row %d: %s", row_idx, e)
            return None

        # -- Cutoff selection ----------------------------------------------
        is_mol  = bool(row.get("is_molecule", False))
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

        return {
            "graph":         graph,
            "line_graph":    line_graph,
            "target":        target,
            "aux_targets":   aux_targets,
            "proc_context":  proc_context,
            "stack_context": stack_context,
            "row_idx":       row_idx,
            "formula":       row.get("formula", ""),
            "source":        row.get("source", ""),
        }

    @staticmethod
    def collate_fn(batch):
        """FIX2: stacks aux_targets per-task for multi-task evaluation.
        v2.2: also stacks proc_context and stack_context tensors per batch.
        All aux_target values are NaN-tensors (not None) after __getitem__ fix,
        so torch.stack works safely across the whole batch.
        """
        import dgl
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        graphs      = dgl.batch([b["graph"]      for b in batch])
        line_graphs = dgl.batch([b["line_graph"] for b in batch])
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

        return {
            "graph":         graphs,
            "line_graph":    line_graphs,
            "target":        targets,
            "aux_targets":   aux_targets,
            "proc_context":  proc_context,
            "stack_context": stack_context,
            "formulas":      [b["formula"] for b in batch],
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

    For continous targets, bins are created based on value ranges.
    If any bin has < 2 samples, falls back to random split.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    df       = dataset.df.iloc[dataset.valid_idx].copy()

    # use the dataset's target column if not specified
    if target_col is None:
        target_col = dataset.target_col

    target_vals = df[target_col].values

    # Remove NaN values for binning
    valid_mask = ~np.isnan(target_vals.astype(float))
    if not valid_mask.any():
        log.warning("All target values are NaN. Falling back to random split.")
        return get_random_split(dataset, train_frac, val_frac, seed)
    
    # Create bins on percentiles to ensure balanced distribution
    try:
        # use quntile-based binning for better distribution
        n_bins = min(6, len(target_vals) // 10) # At least 10 samples per bin
        if n_bins < 2:
            log.warning("Not enough samples for stratification. Falling back to random split.")
            return get_random_split(dataset, train_frac, val_frac, seed)
        
        # create bins based on percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins_edges  = np.percentile(target_vals[valid_mask], percentiles)
        # Make bins unique by adding small epsilon
        bins_edges = np.unique(bins_edges)
        if len(bins_edges) < 2:
            log.warning("Not enough unique bins for stratification. Falling back to random split.")
            return get_random_split(dataset, train_frac, val_frac, seed)
        
        # Assign bins
        target_bins = np.digitize(target_vals, bins_edges[1:-1])
        
        # Check if any bin has <2 samples
        unique, counts = np.unique(target_bins, return_counts=True)
        if (counts < 2).any():
            log.warning("Some bins have less than 2 samples. Falling back to random split.")
            return get_random_split(dataset, train_frac, val_frac, seed)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=(1 - train_frac),
            random_state=seed
        )

        idx_all = np.arange(len(dataset))
        for train_idx, temp_idx in sss.split(idx_all, target_bins):
            pass

        # Further split temp -> val + test
        target_bins_temp = target_bins[temp_idx]
        test_ratio = (1 - train_frac - val_frac) / (1 - train_frac)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        for val_idx_local, test_idx_local in sss2.split(
            np.arange(len(temp_idx)), target_bins_temp
        ):
            pass

        val_idx  = temp_idx[val_idx_local]
        test_idx = temp_idx[test_idx_local]

    except Exception as e:
        log.warning("Stratification failed: %s. Falling back to random split.", e)
        return get_random_split(dataset, train_frac, val_frac, seed)

    log.info(
        "Split -- train: %d  val: %d  test: %d",
        len(train_idx), len(val_idx), len(test_idx)
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
        """Fit z-score normalisation stats from a training dataframe."""
        means, stds = [], []
        for col in self._num_cols:
            if col in df.columns:
                vals = df[col].dropna()
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
        """Fit z-score normalisation stats from a training dataframe."""
        means, stds = [], []
        for col in self._num_cols:
            if col in df.columns:
                vals = df[col].dropna()
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

        # Default task names if not specified
        if task_names is None:
            task_names = ["k_measured", "band_gap", "J_g_log", "E_BD"]
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

    def forward(
        self,
        graph,
        line_graph,
        task:          str           = "k_measured",
        proc_context:  Optional[dict] = None,
        stack_context: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass through ALIGNN backbone + context fusion + specified task head.
        """
        embedding = self.backbone((graph, line_graph, None))
        embedding = self.dropout(embedding)
        fused     = self._fuse(embedding, proc_context, stack_context)
        return self.task_heads[task](fused)

    def forward_all_tasks(
        self,
        graph,
        line_graph,
        proc_context:  Optional[dict] = None,
        stack_context: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for ALL task heads simultaneously (multi-task training)."""
        embedding = self.backbone((graph, line_graph, None))
        embedding = self.dropout(embedding)
        fused     = self._fuse(embedding, proc_context, stack_context)
        return {task: head(fused) for task, head in self.task_heads.items()}

    def load_pretrained_weights(
        self, checkpoint_path: Path, strict: bool = False
    ):
        """
        Load pretrained weights with flexible matching.
        strict=False allows loading Tier N weights into Tier N+1 model
        even if output heads differ.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        log.info(
            "Loaded weights from %s  "
            "(missing=%d, unexpected=%d)",
            checkpoint_path, len(missing), len(unexpected)
        )
        return self


# ==============================================================================
# SECTION 6 -- MULTI-TASK MASKED LOSS  (was Section 5)
# ==============================================================================

class MaskedMultiTaskLoss(nn.Module):
    """
    Multi-task MSE loss with:
    1. Masking for missing targets (NaN -> excluded from loss)
    2. Per-task loss weighting (high-k entries weighted more heavily)
    3. High-k upweighting: entries with k > 35 get weight multiplier

    This directly addresses the <1% class imbalance for k > 35 entries
    identified in the Week 3 EDA activity.
    """

    HIGH_K_THRESHOLD  = 35.0
    HIGH_K_MULTIPLIER = 5.0    # 5x weight for k > 35 entries

    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        upweight_high_k: bool = True,
    ):
        super().__init__()
        self.task_weights    = task_weights or {
            "k_measured":               2.0,    # Tier 3 primary -- measured experimental k
            "k_total":                  2.0,    # Tier 1/2 primary -- DFT ionic+elec total
            "band_gap":                 1.0,
            "formation_energy_per_atom": 1.0,   # Tier 1 primary
            "e_above_hull":             0.5,    # Tier 2/3 aux -- stability proxy
            "J_g_log":                  1.5,    # Tier 3 aux -- important for reliability
            "E_BD":                     1.0,
        }
        self.upweight_high_k = upweight_high_k
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        total_loss  = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        n_tasks_active = 0

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

            # High-k upweighting for dielectric prediction tasks (both head names)
            if task in ("k_measured", "k_total") and self.upweight_high_k:
                high_k_mask   = tgt_m > self.HIGH_K_THRESHOLD
                sample_weights = torch.ones_like(per_sample_loss)
                sample_weights[high_k_mask] = self.HIGH_K_MULTIPLIER
                per_sample_loss = per_sample_loss * sample_weights

            task_loss   = per_sample_loss.mean()
            total_loss += self.task_weights.get(task, 1.0) * task_loss
            n_tasks_active += 1

        return total_loss / max(n_tasks_active, 1)


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
        device:          str  = "cuda" if torch.cuda.is_available() else "cpu",
        ckpt_prefix:     str  = "tier",
        ablate_context:  bool = False,
    ):
        self.model          = model.to(device)
        self.cfg            = tier_cfg
        self.device         = device
        self.ckpt_prefix    = ckpt_prefix
        self.ablate_context = ablate_context   # Obs 3: force ctx=None in all fwd calls

        # Optimizer -- AdamW with decoupled weight decay (paper section Methods)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = tier_cfg["learning_rate"],
            weight_decay = tier_cfg["weight_decay"],
        )

        # Loss
        self.criterion = MaskedMultiTaskLoss(upweight_high_k=True)

        # Best metric tracking
        self.best_val_mae  = float("inf")
        self.best_epoch    = 0
        self.patience      = 30
        self.patience_ctr  = 0

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
                pct_start     = 0.3,
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

            self.optimizer.zero_grad()

            # Multi-task forward -- all heads produce predictions in one pass
            preds = self.model.forward_all_tasks(
                graph, line_graph,
                proc_context=proc_ctx,
                stack_context=stack_ctx,
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

            loss = self.criterion(preds, targets_dict)
            loss.backward()

            # Gradient clipping (important for stability on dielectric prediction)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
        loader:     DataLoader,
        target_col: str,
    ) -> Tuple[float, float]:
        """Evaluate MAE and RMSE on a validation/test split."""
        self.model.eval()
        preds_all   = []
        targets_all = []

        for batch in loader:
            if batch is None:
                continue
            graph      = batch["graph"].to(self.device)
            line_graph = batch["line_graph"].to(self.device)
            target     = batch["target"]

            # v2.2: context branches (None = zero contribution for Tier 1/2)
            proc_ctx  = batch.get("proc_context")
            stack_ctx = batch.get("stack_context")
            if proc_ctx is not None:
                proc_ctx  = {k: v.to(self.device) for k, v in proc_ctx.items()}
                stack_ctx = {k: v.to(self.device) for k, v in stack_ctx.items()}
            if self.ablate_context:
                proc_ctx = stack_ctx = None

            pred = self.model(
                graph, line_graph, task=target_col,
                proc_context=proc_ctx, stack_context=stack_ctx,
            )
            preds_all.append(pred.cpu())
            targets_all.append(target)

        if not preds_all:
            return float("inf"), float("inf")

        preds   = torch.cat(preds_all).squeeze()
        targets = torch.cat(targets_all).squeeze()

        valid   = ~torch.isnan(targets)
        mae     = (preds[valid] - targets[valid]).abs().mean().item()
        rmse    = ((preds[valid] - targets[valid]) ** 2).mean().sqrt().item()
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
        for col in aux_cols:
            for head_name, col_name in TASK_TO_COLUMN.items():
                if col_name == col:
                    task_to_batch_key[head_name] = col
                    break
            else:
                task_to_batch_key[col] = col

        task_preds   = {h: [] for h in self.model.task_heads}
        task_targets = {h: [] for h in self.model.task_heads}
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

            all_preds  = self.model.forward_all_tasks(
                graph, line_graph,
                proc_context=proc_ctx,
                stack_context=stack_ctx,
            )
            n_batches += 1

            for head_name, pred_t in all_preds.items():
                task_preds[head_name].append(pred_t.cpu())
                bkey = task_to_batch_key.get(head_name, "__none__")
                if bkey == "__primary__" or head_name == primary_col:
                    tgt = batch["target"]
                elif bkey != "__none__":
                    tgt = batch.get("aux_targets", {}).get(bkey)
                    if tgt is None:
                        tgt = torch.full(batch["target"].shape, float("nan"))
                else:
                    tgt = torch.full(batch["target"].shape, float("nan"))
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

        for head_name in self.model.task_heads:
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
        path = CKPT_ROOT / f"{self.ckpt_prefix}_{tag}.pt"
        torch.save({
            "epoch":           epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_mae":         val_mae,
            "config":          self.cfg,
        }, path)
        log.info("Checkpoint saved -> %s  (epoch=%d, val_mae=%.4f)",
                 path, epoch, val_mae)

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        target_col:   str,
    ):
        """Full training loop with early stopping and checkpoint saving."""
        n_epochs   = self.cfg["epochs"]
        scheduler  = self.build_scheduler(len(train_loader), n_epochs)
        history    = []

        log.info("Starting training: %d epochs, target='%s', device=%s",
                 n_epochs, target_col, self.device)

        for epoch in range(1, n_epochs + 1):
            t_start    = time.time()
            epoch_stats = self.train_epoch(train_loader, scheduler, target_col)
            train_loss  = epoch_stats["loss"]
            val_mae, val_rmse = self.evaluate(val_loader, target_col)

            if not isinstance(scheduler,
                              torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Track best
            improved = val_mae < self.best_val_mae
            if improved:
                self.best_val_mae = val_mae
                self.best_epoch   = epoch
                self.patience_ctr = 0
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
                self.model.alpha.item(), self.model.beta.item(),
                epoch_stats["proc_avail_pct"], epoch_stats["stack_avail_pct"],
                "✓" if improved else "",
            )

            history.append({
                "epoch":           epoch,
                "train_loss":      train_loss,
                "val_mae":         val_mae,
                "val_rmse":        val_rmse,
                "alpha":           self.model.alpha.item(),
                "beta":            self.model.beta.item(),
                "proc_avail_pct":  epoch_stats["proc_avail_pct"],
                "stack_avail_pct": epoch_stats["stack_avail_pct"],
            })

            # Save every 50 epochs as a safety checkpoint
            if epoch % 50 == 0:
                self.save_checkpoint(epoch, val_mae, tag=f"ep{epoch}")

            # Early stopping
            if self.patience_ctr >= self.patience:
                log.info(
                    "Early stopping at epoch %d "
                    "(no improvement for %d epochs)",
                    epoch, self.patience
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

def build_dataloader(
    df: pd.DataFrame,
    target_col: str,
    aux_cols:   List[str],
    train_frac: float,
    val_frac:   float,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with stratified splitting."""
    dataset = HighKGraphDataset(df, target_col=target_col, aux_cols=aux_cols)

    train_ds, val_ds, test_ds = get_stratified_split(
        dataset, train_frac=train_frac, val_frac=val_frac
    )

    collate = HighKGraphDataset.collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate
    )

    return train_loader, val_loader, test_loader


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

    task_names = [cfg["target"]] + cfg["aux_targets"]
    log.info("Tier 1 training with targets: %s",task_names)

    train_loader, val_loader, test_loader = build_dataloader(
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
    history = trainer.train(train_loader, val_loader, target_col=cfg["target"])

    # FIX2+MAD:MAE: load best checkpoint then evaluate all task heads
    best_ckpt = CKPT_ROOT / "tier1_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state_dict"])

    test_mae, test_rmse = trainer.evaluate(test_loader, cfg["target"])
    log.info("Tier 1 TEST  MAE=%.4f  RMSE=%.4f  (target: %s)",
             test_mae, test_rmse, cfg["target"])

    mt_results = trainer.evaluate_multitask(test_loader, cfg, df_full=df_tier1)
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

    # Only use rows with k_total for Tier 2 (k_total = DFT ionic + elec total).
    # FIX-OBS2: was filtering on k_measured (correct column) but then passing
    # target_col="k_total" to the dataset -- k_total now exists in the dataframe.
    df_t2_k = df_tier2[df_tier2["k_total"].notna()].copy()
    log.info("Tier 2 rows with k_total: %d", len(df_t2_k))

    train_loader, val_loader, test_loader = build_dataloader(
        df         = df_t2_k,
        target_col = cfg["target"],
        aux_cols   = cfg["aux_targets"],
        train_frac = cfg["train_ratio"],
        val_frac   = cfg["val_ratio"],
        batch_size = cfg["batch_size"],
    )

    # Load model with Tier 1 weights.
    # FIX-OBS1 collateral: use explicit task_names derived from tier config so
    # head names match the column names flowing through train_epoch's targets_dict.
    # Previously n_output_tasks=4 silently fell back to default heads
    # ["k_measured","band_gap","J_g_log","E_BD"] which are wrong for Tier 2.
    task_names = [cfg["target"]] + cfg["aux_targets"]
    log.info("Tier 2 training with task heads: %s", task_names)
    model = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names)
    model.load_pretrained_weights(pretrained_weights, strict=False)
    # v2.2: fit encoder stats AFTER weight load (overwrites Tier 1 zeros).
    # Tier 2 df has no proc/stack columns → stats stay mean=0 std=1 (correct).
    model.fit_encoder_stats(df_t2_k)

    # Freeze first 2 ALIGNN layers for early fine-tuning stability
    model.freeze_alignn_layers(cfg["freeze_layers"])

    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix="tier2",
                            ablate_context=ablate_context)

    # Phase 1: frozen lower layers (50 epochs)
    log.info("  Phase 1: lower %d ALIGNN layers frozen (50 epochs)",
             cfg["freeze_layers"])
    cfg_phase1 = {**cfg, "epochs": 50}
    trainer.cfg = cfg_phase1
    scheduler1  = trainer.build_scheduler(len(train_loader), 50)
    for epoch in range(1, 51):
        trainer.train_epoch(train_loader, scheduler1, cfg["target"])

    # Phase 2: unfreeze all, continue at lower lr
    log.info("  Phase 2: all layers unfrozen (100 epochs)")
    model.unfreeze_all()
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"] * 0.5,
        weight_decay=cfg["weight_decay"],
    )
    cfg_phase2 = {**cfg, "epochs": 100}
    trainer.cfg = cfg_phase2
    history = trainer.train(train_loader, val_loader, target_col=cfg["target"])

    # FIX2+MAD:MAE: load best checkpoint then evaluate all task heads
    best_ckpt = CKPT_ROOT / "tier2_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state_dict"])

    test_mae, test_rmse = trainer.evaluate(test_loader, cfg["target"])
    log.info("Tier 2 TEST  MAE=%.4f  RMSE=%.4f  (target: k_total)",
             test_mae, test_rmse)

    mt_results = trainer.evaluate_multitask(test_loader, cfg, df_full=df_t2_k)
    trainer.print_multitask_results(mt_results, split_name="TEST", tier_name="TIER 2")

    out_path = REPORT_ROOT / "tier2_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"primary_mae": test_mae, "primary_rmse": test_rmse,
                   "multitask": mt_results}, f, indent=2)
    log.info("Tier 2 test results saved -> %s", out_path)

    return CKPT_ROOT / "tier2_best.pt"


def run_tier3_finetune(
    df_tier3: pd.DataFrame,
    pretrained_weights: Path,
    ablate_context: bool = False,
):
    """
    Tier 3 -- Project fine-tuning on HfO2-family (with process parameters).

    Loads Tier 2 pretrained weights.
    All layers unfrozen -- final adaptation to project-specific material space.
    Very low learning rate (5e-5) to preserve domain knowledge.

    Key difference from Tiers 1-2: Tier 3 dataset includes experimental
    entries with real ALD/anneal process parameters. The ALIGNN backbone
    handles the crystal structure branch; a separate MLP handles process
    parameters. Both outputs are concatenated before task heads.
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

    # ── TODO(v4): process-only rows are silently excluded from training ──────
    #
    # CURRENT GAP: df_process_only contains experimental entries that have ALD
    # process parameters and measured k/J_g/E_BD values but NO matched crystal
    # structure (atoms_dict is NaN).  These rows are counted and logged above
    # but never passed to build_dataloader, fit_encoder_stats, or the trainer.
    #
    # WHY IT MATTERS: these rows represent the purest signal for the
    # process/stack context branches (ProcessParamsEncoder + StackContextEncoder).
    # Excluding them means the encoders train exclusively on structural rows
    # where process data is a secondary annotation, not the primary learning
    # signal.  Any claim of "full process-aware Tier 3 learning" requires
    # this gap to be closed.
    #
    # WHY IT IS NOT FIXED IN v3: ALIGNN requires a DGL crystal graph, which
    # requires atoms_dict.  A process-only row has no graph to feed the
    # backbone, so it cannot flow through the current HighKALIGNN.forward()
    # path at all -- passing it to build_dataloader would cause a None-graph
    # error in collate_fn.
    #
    # IMPLEMENTATION PATHS FOR v4 (choose one before claiming process-aware
    # Tier 3):
    #
    #   A) Structure imputation (simplest)
    #      Match each process-only row to its nearest structural neighbour by
    #      composition (e.g. all HfO2 process entries → monoclinic P21/c HfO2
    #      from JARVIS jid="JVASP-815").  Imputed structure feeds the backbone;
    #      the crystal embedding is a stand-in but the process branch receives
    #      real gradients.  Risk: imputed crystal embedding adds noise when the
    #      actual film phase differs from the matched structure.
    #
    #   B) Composition-only graph (medium)
    #      Build a minimal one-node DGL graph from chemical composition alone
    #      (atom-type embeddings, no bond geometry).  The backbone produces a
    #      composition-level embedding; the process branch corrects it toward
    #      the measured k.  Requires a light backbone variant or a separate
    #      composition encoder path in HighKALIGNN.
    #
    #   C) Separate process-only MLP head (cleanest separation)
    #      Train a standalone MLP(ProcessParamsEncoder → k_measured) on
    #      df_process_only after the ALIGNN model is trained on df_structural.
    #      Ensemble predictions at inference: if atoms_dict is present use
    #      ALIGNN path; if absent use MLP path.  Clean boundary, no risk of
    #      the process-only signal degrading the structural backbone.
    #
    # PREREQUISITE ANALYSIS before choosing a path:
    #   - Log len(df_process_only) across real Tier 3 runs; if < 50 rows the
    #     signal may be too weak to justify the architecture change.
    #   - Check whether process-only rows cluster in k_measured space (if they
    #     are all high-k, option C is attractive; if mixed, option A is safer).
    #   - Run the --ablate_context flag on the structural rows first to confirm
    #     context branches add value before investing in process-only support.
    #
    if len(df_process_only) > 0:
        log.warning(
            "Tier 3: %d process-only experimental rows (no crystal structure) "
            "are EXCLUDED from training.  These rows have ALD process params "
            "and measured k/J_g/E_BD but no atoms_dict.  "
            "See TODO(v4) in run_tier3_finetune for implementation paths.  "
            "Full process-aware Tier 3 learning requires addressing this gap.",
            len(df_process_only),
        )

    train_loader, val_loader, test_loader = build_dataloader(
        df         = df_structural,
        target_col = cfg["target"],
        aux_cols   = cfg["aux_targets"],
        train_frac = cfg["train_ratio"],
        val_frac   = cfg["val_ratio"],
        batch_size = cfg["batch_size"],
    )

    # Load model with Tier 2 weights -- no frozen layers for final fine-tune.
    # FIX-OBS1 collateral: explicit task_names from tier config for head-name
    # consistency with train_epoch targets_dict routing.
    task_names_t3 = [cfg["target"]] + cfg["aux_targets"]
    log.info("Tier 3 training with task heads: %s", task_names_t3)
    model = HighKALIGNN(config=ALIGNN_BASE_CONFIG, task_names=task_names_t3)
    model.load_pretrained_weights(pretrained_weights, strict=False)
    model.unfreeze_all()
    # v2.2: fit encoder stats AFTER weight load so Tier 3 ALD/stack data
    # overwrites the uninformative zeros from the Tier 2 checkpoint.
    # This is the FIRST tier where proc_avail_flag > 0 rows appear (experimental
    # entries from process_db_clean.csv). The encoders will receive real gradient
    # signal and alpha/beta will grow from their 1e-3 initial values.
    model.fit_encoder_stats(df_structural)

    trainer = ALIGNNTrainer(model, cfg, ckpt_prefix="tier3",
                            ablate_context=ablate_context)
    history = trainer.train(train_loader, val_loader, target_col=cfg["target"])

    # FIX2+MAD:MAE: load best checkpoint then evaluate all task heads
    best_ckpt = CKPT_ROOT / "tier3_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state_dict"])

    test_mae, test_rmse = trainer.evaluate(test_loader, cfg["target"])
    log.info("Tier 3 TEST  MAE=%.4f  RMSE=%.4f  (target: k_measured)",
             test_mae, test_rmse)

    mt_results = trainer.evaluate_multitask(test_loader, cfg, df_full=df_structural)
    trainer.print_multitask_results(mt_results, split_name="TEST", tier_name="TIER 3")

    # Project-specific MAD:MAE summary for k_measured
    if "k_measured" in mt_results and mt_results["k_measured"]["n"] > 0:
        km = mt_results["k_measured"]
        log.info(
            "k_measured MAD=%.2f  MAE=%.4f  MAD:MAE=%.2f  "
            "(paper: 1.63 @ 44K no transfer | goal: >= 2.5 with three-tier transfer)",
            km["mad"], km["mae"], km["mad_mae_ratio"],
        )

    out_path = REPORT_ROOT / "tier3_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"primary_mae": test_mae, "primary_rmse": test_rmse,
                   "multitask": mt_results}, f, indent=2)
    log.info("Tier 3 test results saved -> %s", out_path)

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
    k_col = "k_total" if tier_num < 3 else "k_measured"
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
        k_col   = "k_total" if t < 3 else "k_measured"
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


def main():
    parser = argparse.ArgumentParser(
        description="High-k ALIGNN Three-Tier Scalable Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "extract_only", "tier1_pretrain",
                 "tier2_finetune", "tier3_finetune", "dataset_stats"],
        default="full_pipeline",
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Re-download raw data even if cache exists")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Rebuild tier datasets even if they exist")
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

    log.info("HighK ALIGNN Pipeline  mode=%s", args.mode)
    log.info("Device: %s", "GPU OK" if torch.cuda.is_available() else "CPU only")
    if torch.cuda.is_available():
        log.info("GPU: %s  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
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
    log.info("-" * 60)
    log.info("Step 5/5: Building three-tier dataset")

    df_tier1 = builder.build_tier1(df_jarvis, df_mp, df_qm9,
                                    force_rebuild=args.force_rebuild,
                                    skip_cross_dedup=args.skip_cross_dedup)
    df_tier2 = builder.build_tier2(df_tier1,
                                    force_rebuild=args.force_rebuild)
    df_tier3 = builder.build_tier3(df_tier2, df_exp,
                                    force_rebuild=args.force_rebuild)

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
                           ablate_context=args.ablate_context)

    log.info("Pipeline complete. Final model: %s/tier3_best.pt", CKPT_ROOT)


if __name__ == "__main__":
    main()
