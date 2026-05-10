#!/usr/bin/env python3
"""
================================================================================
  JARVIS + ALIGNN Multi-Task Pipeline
  Tasks:
    1. Band Gap Regression       (eV)   — OPT/mBJ level DFT targets
    2. High-k Classification     (ε>10) — Binary: high-k dielectric vs normal
    3. Dielectric Constant Regression   — ε_static auxiliary head

  Backbone  : ALIGNN (Atomistic Line Graph Neural Network)
              Choudhary & DeCost, npj Comput. Mater. 7, 185 (2021)
              DOI: 10.1038/s41524-021-00650-1

  Dataset   : JARVIS-DFT 3D (~75K materials, NIST)
              https://jarvis.nist.gov / jarvis-tools

  Author    : Generated for semiconductor high-k discovery pipeline
================================================================================

Install dependencies:
  pip install jarvis-tools alignn dgl torch torchvision torchaudio \
              torch-geometric scikit-learn matplotlib seaborn tqdm

  # DGL with CUDA (adjust to your CUDA version):
  pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import pickle
import logging
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import dgl
from dgl.nn import AvgPooling, SumPooling

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_cache_dir:     str   = "./cache/graphs"
    checkpoint_dir:     str   = "./checkpoints"
    results_dir:        str   = "./results"

    # High-k dielectric threshold (ε > 10 is industry standard vs SiO₂ @ 3.9)
    highk_threshold:    float = 10.0

    # Band gap filter: only semiconductors/insulators for high-k relevance
    bandgap_min:        float = 0.1    # eV — exclude metals
    bandgap_max:        float = 15.0   # eV — exclude extreme insulators

    # ── Graph Construction ────────────────────────────────────────────────────
    cutoff:             float = 8.0    # Å — bond cutoff radius
    max_neighbors:      int   = 12     # max bonds per atom
    atom_features:      str   = "cgcnn"  # 92-dim elemental features

    # RBF edge encoding
    rbf_min:            float = 0.5
    rbf_max:            float = 8.0
    num_rbf:            int   = 80     # bond distance RBF bins
    num_triplet_rbf:    int   = 40     # bond angle RBF bins (line graph)

    # ── ALIGNN Backbone ───────────────────────────────────────────────────────
    # Pre-trained model name from ALIGNN figshare
    pretrained_bandgap: str   = "jv_optb88vdw_bandgap_alignn"
    freeze_backbone:    bool  = False   # False = full fine-tune
    freeze_epochs:      int   = 5       # epochs to keep backbone frozen if True

    # Architecture
    alignn_layers:      int   = 4       # line-graph ALIGNN conv layers
    gcn_layers:         int   = 4       # atom graph conv layers
    atom_input_features: int  = 92      # CGCNN feature size
    edge_input_features: int  = 80      # RBF edge features
    triplet_input_features: int = 40    # RBF angle features
    embedding_features: int   = 64
    hidden_features:    int   = 256
    output_features:    int   = 256     # backbone output dim before heads

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:         int   = 64
    num_epochs:         int   = 100
    learning_rate:      float = 1e-3
    weight_decay:       float = 1e-5
    lr_scheduler:       str   = "cosine"   # "cosine" | "step" | "plateau"
    warmup_epochs:      int   = 5
    grad_clip:          float = 5.0
    early_stop_patience: int  = 15

    # Multi-task loss weights (learnable uncertainty weighting active by default)
    use_uncertainty_weighting: bool = True
    lambda_bandgap:     float = 1.0    # manual weight if not using uncertainty
    lambda_highk:       float = 1.5    # upweight classification (imbalanced)
    lambda_eps:         float = 0.5    # auxiliary dielectric regression

    # Train/val/test split
    train_ratio:        float = 0.80
    val_ratio:          float = 0.10
    test_ratio:         float = 0.10
    random_seed:        int   = 42

    # ── Hardware ──────────────────────────────────────────────────────────────
    num_workers:        int   = 4
    device:             str   = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision:    bool  = True

    def __post_init__(self):
        for d in [self.data_cache_dir, self.checkpoint_dir, self.results_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = PipelineConfig()


# ──────────────────────────────────────────────────────────────────────────────
# JARVIS DATA LOADING & FILTERING
# ──────────────────────────────────────────────────────────────────────────────
def load_jarvis_data(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load JARVIS-DFT 3D dataset and filter for materials with valid:
      - Band gap (OPT level)
      - Static dielectric constant ε_static
    Returns a cleaned DataFrame ready for graph construction.
    """
    from jarvis.db.figshare import data as jdata

    cache_path = Path(cfg.data_cache_dir) / "jarvis_filtered.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached JARVIS data from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} materials from cache")
        return df

    logger.info("Downloading JARVIS-DFT 3D dataset (~75K materials)...")
    raw = jdata("dft_3d")
    df  = pd.DataFrame(raw)

    logger.info(f"Raw JARVIS entries: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    # ── Property filtering ────────────────────────────────────────────────────
    def _is_valid_float(val, min_v=None, max_v=None):
        try:
            v = float(val)
            if not np.isfinite(v): return False
            if min_v is not None and v < min_v: return False
            if max_v is not None and v > max_v: return False
            return True
        except (TypeError, ValueError):
            return False

    # Filter: valid bandgap
    mask_bg = df["optb88vdw_bandgap"].apply(
        lambda x: _is_valid_float(x, cfg.bandgap_min, cfg.bandgap_max)
    )
    # Filter: valid static dielectric
    mask_eps = df["eps_static"].apply(
        lambda x: _is_valid_float(x, 0.5, 1000.0)
    )
    # Filter: valid atoms dict
    mask_atoms = df["atoms"].apply(
        lambda x: isinstance(x, dict) and "elements" in x and len(x["elements"]) > 0
    )

    df = df[mask_bg & mask_eps & mask_atoms].copy()
    logger.info(f"After filtering: {len(df)} materials")

    # ── Feature extraction ────────────────────────────────────────────────────
    df["bandgap"]   = df["optb88vdw_bandgap"].astype(float)
    df["eps_static_val"] = df["eps_static"].astype(float)

    # Use mBJ band gap where available (more accurate for semiconductors)
    def _best_bandgap(row):
        try:
            mbj = float(row["mbj_bandgap"])
            if np.isfinite(mbj) and cfg.bandgap_min < mbj < cfg.bandgap_max:
                return mbj
        except (TypeError, ValueError):
            pass
        return float(row["optb88vdw_bandgap"])

    df["bandgap_best"] = df.apply(_best_bandgap, axis=1)

    # ── High-k label ──────────────────────────────────────────────────────────
    df["is_high_k"] = (df["eps_static_val"] > cfg.highk_threshold).astype(int)

    # ── Auxiliary features ────────────────────────────────────────────────────
    def _safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    df["formation_energy"] = df.get("formation_energy_peratom", pd.Series(0.0, index=df.index)).apply(_safe_float)
    df["n_atoms"]          = df["atoms"].apply(lambda x: len(x["elements"]) if isinstance(x, dict) else 0)

    # Keep only needed columns
    keep = ["jid", "atoms", "bandgap", "bandgap_best",
            "eps_static_val", "is_high_k", "formation_energy", "n_atoms"]
    df = df[keep].reset_index(drop=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total materials     : {len(df)}")
    logger.info(f"  High-k (ε > {cfg.highk_threshold})  : {df['is_high_k'].sum()} ({df['is_high_k'].mean()*100:.1f}%)")
    logger.info(f"  Band gap (eV) mean  : {df['bandgap_best'].mean():.3f} ± {df['bandgap_best'].std():.3f}")
    logger.info(f"  ε_static mean       : {df['eps_static_val'].mean():.2f} ± {df['eps_static_val'].std():.2f}")
    logger.info(f"{'='*50}\n")

    df.to_parquet(cache_path)
    logger.info(f"Cached filtered dataset to {cache_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION  (ALIGNN-style: atom graph + line graph)
# ──────────────────────────────────────────────────────────────────────────────
def rbf_expansion(distances: torch.Tensor, d_min: float, d_max: float, num_rbf: int) -> torch.Tensor:
    """Gaussian Radial Basis Function expansion."""
    centers = torch.linspace(d_min, d_max, num_rbf, device=distances.device)
    width   = (d_max - d_min) / num_rbf
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


def cosine_rbf(angles: torch.Tensor, num_rbf: int = 40) -> torch.Tensor:
    """RBF expansion over cosine of bond angles ∈ [-1, 1]."""
    centers = torch.linspace(-1.0, 1.0, num_rbf, device=angles.device)
    width   = 2.0 / num_rbf
    return torch.exp(-((angles.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


def build_alignn_graphs(atoms_dict: dict, cfg: PipelineConfig):
    """
    Build atom graph g and line graph lg from a JARVIS Atoms dict.

    Atom graph  g  : nodes = atoms,  edges = bonds within cutoff
    Line graph  lg : nodes = bonds,  edges = bond pairs sharing an atom
                     (encodes 3-body angular interactions)

    Returns:
        g   : DGLGraph  — atom graph with node/edge features
        lg  : DGLGraph  — line graph with bond-angle features
        N   : int       — number of atoms
    """
    from jarvis.core.atoms import Atoms
    from jarvis.core.graphs import Graph as JGraph

    atoms = Atoms.from_dict(atoms_dict)

    # Build DGL multi-graph with periodic boundary conditions
    g_dgl, lg_dgl = JGraph.atom_dgl_multigraph(
        atoms,
        cutoff=cfg.cutoff,
        atom_features=cfg.atom_features,   # 92-dim CGCNN features
        max_neighbors=cfg.max_neighbors,
        compute_line_graph=True,
        use_canonize=True                  # canonical edge ordering
    )

    # ── Node features: 92-dim CGCNN elemental descriptors ─────────────────────
    x = torch.tensor(g_dgl.ndata["atom_features"].numpy(), dtype=torch.float32)

    # ── Edge features: RBF-encoded bond distances ─────────────────────────────
    bond_vecs  = g_dgl.edata["r"]                              # [E, 3] displacement
    bond_dists = torch.norm(torch.tensor(bond_vecs.numpy(), dtype=torch.float32), dim=1)  # [E]
    edge_feats = rbf_expansion(bond_dists, cfg.rbf_min, cfg.rbf_max, cfg.num_rbf)         # [E, 80]

    g_dgl.ndata["x"]            = x
    g_dgl.edata["edge_attr"]    = edge_feats
    g_dgl.edata["bond_dist"]    = bond_dists.unsqueeze(-1)

    # ── Line graph features: RBF-encoded bond angles ───────────────────────────
    # lg_dgl.ndata["r"] = bond displacement vectors (same as g.edata["r"])
    if "r" in lg_dgl.ndata:
        lg_bond_vecs = torch.tensor(lg_dgl.ndata["r"].numpy(), dtype=torch.float32)  # [E, 3]

        # Compute cosine of angle between consecutive bonds sharing an atom
        # Each edge in lg = (b_i, b_j) where b_i, b_j are bonds in g
        # We need the angle between them
        src_lg, dst_lg = lg_dgl.edges()
        v1 = lg_bond_vecs[src_lg]   # [E_lg, 3]
        v2 = lg_bond_vecs[dst_lg]   # [E_lg, 3]

        cos_angles = F.cosine_similarity(v1, v2, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)  # [E_lg]
        triplet_feats = cosine_rbf(cos_angles, cfg.num_triplet_rbf)                  # [E_lg, 40]
        lg_dgl.edata["triplet_attr"] = triplet_feats

        # Line graph node features = bond distance RBF (bond becomes node in lg)
        lg_bond_dists = torch.norm(lg_bond_vecs, dim=1)
        lg_node_feats = rbf_expansion(lg_bond_dists, cfg.rbf_min, cfg.rbf_max, cfg.num_rbf)
        lg_dgl.ndata["x"] = lg_node_feats
    else:
        # Fallback: zero features
        n_lg = lg_dgl.num_nodes()
        e_lg = lg_dgl.num_edges()
        lg_dgl.ndata["x"] = torch.zeros(n_lg, cfg.num_rbf)
        if e_lg > 0:
            lg_dgl.edata["triplet_attr"] = torch.zeros(e_lg, cfg.num_triplet_rbf)

    return g_dgl, lg_dgl, len(atoms.elements)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────
class JARVISHighKDataset(Dataset):
    """
    PyTorch Dataset wrapping JARVIS-DFT materials as ALIGNN-compatible
    (atom graph, line graph) pairs with multi-task labels.
    """

    def __init__(self, df: pd.DataFrame, cfg: PipelineConfig, split: str = "train"):
        self.cfg    = cfg
        self.split  = split
        self.data   = []

        cache_dir = Path(cfg.data_cache_dir) / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building {split} graphs for {len(df)} materials...")
        failed = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split}] graph build"):
            jid        = row["jid"]
            cache_file = cache_dir / f"{jid}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    item = pickle.load(f)
                self.data.append(item)
                continue

            try:
                g, lg, n_atoms = build_alignn_graphs(row["atoms"], cfg)

                item = {
                    "jid":      jid,
                    "g":        g,
                    "lg":       lg,
                    "bandgap":  torch.tensor([row["bandgap_best"]], dtype=torch.float32),
                    "eps":      torch.tensor([row["eps_static_val"]], dtype=torch.float32),
                    "is_high_k": torch.tensor(row["is_high_k"], dtype=torch.long),
                    "n_atoms":  n_atoms,
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(item, f)
                self.data.append(item)

            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.warning(f"  Failed graph for {jid}: {e}")

        logger.info(f"  [{split}] Built: {len(self.data)} | Failed: {failed}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced high-k labels."""
        labels = torch.stack([item["is_high_k"] for item in self.data])
        counts = torch.bincount(labels, minlength=2).float()
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum()
        return weights * 2   # scale for loss

    def get_sampler_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler (address class imbalance)."""
        class_w = self.get_class_weights().numpy()
        return [class_w[item["is_high_k"].item()] for item in self.data]


def collate_alignn(batch: List[dict]):
    """Custom collate: batch DGL graphs using dgl.batch()."""
    jids      = [b["jid"] for b in batch]
    g_list    = [b["g"] for b in batch]
    lg_list   = [b["lg"] for b in batch]

    bg  = dgl.batch(g_list)   # batched atom graph
    blg = dgl.batch(lg_list)  # batched line graph

    bandgap  = torch.cat([b["bandgap"]  for b in batch], dim=0)   # [B, 1]
    eps      = torch.cat([b["eps"]      for b in batch], dim=0)   # [B, 1]
    is_high_k = torch.stack([b["is_high_k"] for b in batch])       # [B]

    return {
        "jids":     jids,
        "g":        bg,
        "lg":       blg,
        "bandgap":  bandgap,
        "eps":      eps,
        "is_high_k": is_high_k
    }


# ──────────────────────────────────────────────────────────────────────────────
# ALIGNN BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────
class EdgeGatedGraphConv(nn.Module):
    """
    Edge-gated graph convolution (atom-graph update step).
    Ref: Eq. 3–4 in Choudhary & DeCost 2021
    """
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.src_gate  = nn.Linear(node_dim, edge_dim)
        self.dst_gate  = nn.Linear(node_dim, edge_dim)
        self.edge_gate = nn.Linear(edge_dim, edge_dim)

        self.src_msg   = nn.Linear(node_dim, node_dim)
        self.dst_msg   = nn.Linear(node_dim, node_dim)
        self.edge_msg  = nn.Linear(edge_dim, node_dim)

        self.node_bn   = nn.LayerNorm(node_dim)
        self.edge_bn   = nn.LayerNorm(edge_dim)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        with g.local_scope():
            g.ndata["h"] = node_feats
            g.edata["e"] = edge_feats

            # Edge gate: σ(W_s h_src + W_d h_dst + W_e e)
            g.apply_edges(lambda edges: {
                "gate": torch.sigmoid(
                    self.src_gate(edges.src["h"]) +
                    self.dst_gate(edges.dst["h"]) +
                    self.edge_gate(edges.data["e"])
                )
            })

            # Updated edge features
            new_e = self.edge_bn(
                edge_feats + F.silu(
                    self.src_gate(g.ndata["h"])[g.edges()[0]] +
                    self.dst_gate(g.ndata["h"])[g.edges()[1]] +
                    self.edge_gate(edge_feats)
                )
            )

            # Node message: gated aggregation
            g.edata["gate"] = g.edata["gate"]
            g.edata["msg"]  = g.edata["gate"] * self.edge_msg(new_e)
            g.update_all(
                dgl.function.copy_e("msg", "m"),
                dgl.function.sum("m", "agg")
            )
            new_h = self.node_bn(
                node_feats + F.silu(
                    self.dst_msg(node_feats) + g.ndata["agg"]
                )
            )
            return new_h, new_e


class ALIGNNLayer(nn.Module):
    """
    Full ALIGNN layer:
      1. Line-graph conv: updates bond features using angle triplets
      2. Atom-graph conv: updates atom/bond features using updated bonds
    """
    def __init__(self, node_dim: int, edge_dim: int, triplet_dim: int):
        super().__init__()
        # Line graph operates on edges→nodes of lg (bonds) and edges of lg (angles)
        self.line_graph_conv = EdgeGatedGraphConv(edge_dim, triplet_dim)
        self.atom_graph_conv = EdgeGatedGraphConv(node_dim, edge_dim)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph,
                node_feats: torch.Tensor, edge_feats: torch.Tensor,
                triplet_feats: torch.Tensor):
        # Step 1: Update edge (bond) features via line graph
        edge_feats, _ = self.line_graph_conv(lg, edge_feats, triplet_feats)

        # Sync updated edge features back to atom graph
        # (lg nodes correspond to g edges in order)
        g_num_edges = g.num_edges()
        if edge_feats.shape[0] == g_num_edges:
            pass  # already aligned
        else:
            edge_feats = edge_feats[:g_num_edges]

        # Step 2: Update atom/edge features via atom graph
        node_feats, edge_feats = self.atom_graph_conv(g, node_feats, edge_feats)

        return node_feats, edge_feats, triplet_feats


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-TASK ALIGNN MODEL
# ──────────────────────────────────────────────────────────────────────────────
class MultiTaskALIGNN(nn.Module):
    """
    ALIGNN backbone with three task-specific output heads:
      - Head 1: Band gap regression        → scalar (eV)
      - Head 2: High-k binary classifier   → 2 logits (normal-k / high-k)
      - Head 3: Dielectric ε regression    → scalar (auxiliary)

    Optionally loads pre-trained weights from ALIGNN figshare for the backbone.
    """

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg

        H  = cfg.hidden_features      # 256 — hidden/node dim throughout
        E  = cfg.edge_input_features   # 80  — RBF bond distance dim
        T  = cfg.triplet_input_features # 40 — RBF angle dim

        # ── Input projections ─────────────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(cfg.atom_input_features, H),
            nn.LayerNorm(H),
            nn.SiLU()
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(E, H),
            nn.LayerNorm(H),
            nn.SiLU()
        )
        self.triplet_embedding = nn.Sequential(
            nn.Linear(T, H // 2),
            nn.LayerNorm(H // 2),
            nn.SiLU()
        )

        # ── ALIGNN layers (line-graph + atom-graph) ───────────────────────────
        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(H, H, H // 2)
            for _ in range(cfg.alignn_layers)
        ])

        # ── GCN-only layers (atom graph, no line graph) ───────────────────────
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConv(H, H)
            for _ in range(cfg.gcn_layers)
        ])

        # ── Global pooling ────────────────────────────────────────────────────
        self.avg_pool = AvgPooling()
        self.sum_pool = SumPooling()

        # Combine mean + sum pooling for richer graph representation
        self.pool_proj = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.LayerNorm(H),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # ── Task Heads ────────────────────────────────────────────────────────

        def _mlp(in_d, hidden_d, out_d, dropout=0.1):
            return nn.Sequential(
                nn.Linear(in_d, hidden_d),
                nn.LayerNorm(hidden_d),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_d, hidden_d // 2),
                nn.SiLU(),
                nn.Linear(hidden_d // 2, out_d)
            )

        # Head 1: Band gap (eV) — regression with softplus for non-negativity
        self.bandgap_head = _mlp(H, H // 2, 1)
        self.bandgap_act  = nn.Softplus()   # ensures predicted Eg ≥ 0

        # Head 2: High-k binary classifier
        self.highk_head = _mlp(H, H // 2, 2)

        # Head 3: Dielectric constant (auxiliary regression)
        self.eps_head = _mlp(H, H // 2, 1)
        self.eps_act  = nn.Softplus()        # ε must be > 0

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained_backbone(self, model_name: str = None):
        """
        Load pre-trained ALIGNN weights from JARVIS figshare.
        Transfers weights for shared backbone layers only (not task heads).
        """
        try:
            from alignn.pretrained import get_figshare_model
            model_name = model_name or self.cfg.pretrained_bandgap
            logger.info(f"Loading ALIGNN pretrained backbone: '{model_name}'")
            pretrained, _ = get_figshare_model(model_name)

            # Transfer compatible weights
            pretrained_sd = pretrained.state_dict()
            model_sd      = self.state_dict()
            transferred   = 0

            for k, v in pretrained_sd.items():
                # Map ALIGNN weight names to our naming convention
                mapped_key = self._map_pretrained_key(k)
                if mapped_key in model_sd and model_sd[mapped_key].shape == v.shape:
                    model_sd[mapped_key].copy_(v)
                    transferred += 1

            logger.info(f"  Transferred {transferred} / {len(pretrained_sd)} weight tensors")

        except Exception as e:
            logger.warning(f"  Could not load pretrained weights: {e}")
            logger.warning("  Proceeding with random initialization.")

    def _map_pretrained_key(self, key: str) -> str:
        """Map ALIGNN package weight names → our model's attribute names."""
        # ALIGNN package uses 'layers' for ALIGNN layers, 'gcn_layers' for GCN
        mappings = {
            "atom_embedding.0.": "atom_embedding.0.",
            "edge_embedding.0.": "edge_embedding.0.",
            "alignn_layers.":    "alignn_layers.",
            "gcn_layers.":       "gcn_layers.",
        }
        for src, dst in mappings.items():
            if key.startswith(src):
                return key.replace(src, dst, 1)
        return key

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone for stage-wise fine-tuning."""
        backbone_modules = [
            self.atom_embedding, self.edge_embedding,
            self.triplet_embedding, self.alignn_layers, self.gcn_layers,
            self.pool_proj
        ]
        for mod in backbone_modules:
            for p in mod.parameters():
                p.requires_grad = not freeze

        status = "FROZEN" if freeze else "UNFROZEN"
        logger.info(f"  Backbone parameters: {status}")

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict with keys 'g' (batched atom DGL graph), 'lg' (batched line graph)

        Returns:
            bandgap_pred : [B, 1]  — predicted band gap (eV, ≥ 0)
            highk_logits : [B, 2]  — high-k classification logits
            eps_pred     : [B, 1]  — predicted ε_static (≥ 0)
        """
        g  = batch["g"]
        lg = batch["lg"]

        # ── Initial embeddings ────────────────────────────────────────────────
        h  = self.atom_embedding(g.ndata["x"])          # [N_atoms, H]
        e  = self.edge_embedding(g.edata["edge_attr"])  # [N_bonds, H]

        # Triplet features from line graph edges
        if "triplet_attr" in lg.edata:
            t = self.triplet_embedding(lg.edata["triplet_attr"])  # [N_triplets, H//2]
        else:
            t = torch.zeros(lg.num_edges(), self.cfg.hidden_features // 2,
                            device=h.device)

        # Line graph node features (bond → node in lg) need same dim as e
        if lg.num_nodes() > 0:
            lg.ndata["x"] = e[:lg.num_nodes()]  # sync bond feats to lg nodes

        # ── ALIGNN message-passing ────────────────────────────────────────────
        for alignn_layer in self.alignn_layers:
            h, e, t = alignn_layer(g, lg, h, e, t)

        # ── GCN-only layers ───────────────────────────────────────────────────
        for gcn_layer in self.gcn_layers:
            h, e = gcn_layer(g, h, e)

        # ── Global pooling ────────────────────────────────────────────────────
        h_avg  = self.avg_pool(g, h)          # [B, H]
        h_sum  = self.sum_pool(g, h)          # [B, H]
        h_graph = self.pool_proj(torch.cat([h_avg, h_sum], dim=-1))  # [B, H]

        # ── Task predictions ──────────────────────────────────────────────────
        bandgap_pred = self.bandgap_act(self.bandgap_head(h_graph))   # [B, 1] ≥ 0
        highk_logits = self.highk_head(h_graph)                       # [B, 2]
        eps_pred     = self.eps_act(self.eps_head(h_graph))           # [B, 1] ≥ 0

        return bandgap_pred, highk_logits, eps_pred


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-TASK LOSS
# ──────────────────────────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., NeurIPS 2018).
    Learns task-weighting via log-variance parameters σ² per task.
    L_total = Σ_t [ 1/(2σ_t²) * L_t + log(σ_t) ]
    """

    def __init__(self, cfg: PipelineConfig, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.use_uncertainty = cfg.use_uncertainty_weighting

        # Learnable log-variance per task
        if self.use_uncertainty:
            self.log_var_bg  = nn.Parameter(torch.zeros(1))  # band gap
            self.log_var_hk  = nn.Parameter(torch.zeros(1))  # high-k classif
            self.log_var_eps = nn.Parameter(torch.zeros(1))  # dielectric

        # Classification loss with class weights
        cw = class_weights if class_weights is not None else None
        self.ce_loss  = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        bandgap_pred:  torch.Tensor, bandgap_true:  torch.Tensor,
        highk_logits:  torch.Tensor, highk_true:    torch.Tensor,
        eps_pred:      torch.Tensor, eps_true:       torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Per-task losses
        L_bg  = self.mse_loss(bandgap_pred.squeeze(-1), bandgap_true.squeeze(-1))
        L_hk  = self.ce_loss(highk_logits, highk_true)
        L_eps = self.mae_loss(eps_pred.squeeze(-1), eps_true.squeeze(-1))

        if self.use_uncertainty:
            # Kendall-style uncertainty weighting
            total = (
                torch.exp(-self.log_var_bg)  * L_bg  + self.log_var_bg  +
                torch.exp(-self.log_var_hk)  * L_hk  + self.log_var_hk  +
                torch.exp(-self.log_var_eps) * L_eps + self.log_var_eps
            )
            w_bg  = torch.exp(-self.log_var_bg.item()).item()
            w_hk  = torch.exp(-self.log_var_hk.item()).item()
            w_eps = torch.exp(-self.log_var_eps.item()).item()
        else:
            total = (
                self.cfg.lambda_bandgap * L_bg +
                self.cfg.lambda_highk   * L_hk +
                self.cfg.lambda_eps     * L_eps
            )
            w_bg, w_hk, w_eps = self.cfg.lambda_bandgap, self.cfg.lambda_highk, self.cfg.lambda_eps

        details = {
            "loss_bandgap": L_bg.item(),
            "loss_highk":   L_hk.item(),
            "loss_eps":     L_eps.item(),
            "w_bandgap":    w_bg,
            "w_highk":      w_hk,
            "w_eps":        w_eps,
        }
        return total, details


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: nn.Module, cfg: PipelineConfig):
        self.model  = model.to(cfg.device)
        self.cfg    = cfg
        self.device = cfg.device
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and cfg.device == "cuda")

        # Best model tracking
        self.best_val_auc    = 0.0
        self.best_val_mae_bg = float("inf")
        self.patience_counter = 0

        # History
        self.history = {k: [] for k in
                        ["train_loss", "val_loss", "val_mae_bg",
                         "val_auc_hk", "val_mae_eps", "lr"]}

    def configure_optimizers(self, loss_fn: MultiTaskLoss, train_dataset: JARVISHighKDataset):
        # Separate param groups: backbone (lower LR) vs heads (higher LR)
        backbone_params = list(self.model.atom_embedding.parameters()) + \
                          list(self.model.edge_embedding.parameters()) + \
                          list(self.model.triplet_embedding.parameters()) + \
                          list(self.model.alignn_layers.parameters()) + \
                          list(self.model.gcn_layers.parameters()) + \
                          list(self.model.pool_proj.parameters())

        head_params = list(self.model.bandgap_head.parameters()) + \
                      list(self.model.highk_head.parameters()) + \
                      list(self.model.eps_head.parameters())

        loss_params = list(loss_fn.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.cfg.learning_rate * 0.1, "name": "backbone"},
            {"params": head_params,     "lr": self.cfg.learning_rate,       "name": "heads"},
            {"params": loss_params,     "lr": self.cfg.learning_rate * 0.01,"name": "loss_weights"},
        ], weight_decay=self.cfg.weight_decay)

        total_steps = self.cfg.num_epochs * max(1, len(train_dataset) // self.cfg.batch_size)
        warmup_steps = self.cfg.warmup_epochs * max(1, len(train_dataset) // self.cfg.batch_size)

        if self.cfg.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
        elif self.cfg.lr_scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )

        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        self.warmup_steps    = warmup_steps
        self.current_step    = 0

    def _step_scheduler(self, val_metric=None):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_metric is not None:
                self.scheduler.step(val_metric)
        else:
            self.scheduler.step()

    def train_epoch(self, loader: DataLoader, loss_fn: MultiTaskLoss) -> Dict:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc="  train", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision and self.device == "cuda"):
                bg_pred, hk_logits, eps_pred = self.model(batch)
                loss, details = loss_fn(
                    bg_pred,   batch["bandgap"],
                    hk_logits, batch["is_high_k"],
                    eps_pred,  batch["eps"]
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._step_scheduler()
            total_loss += loss.item()
            n_batches  += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, loss_fn: MultiTaskLoss, desc: str = "val") -> Dict:
        self.model.eval()
        all_bg_pred, all_bg_true = [], []
        all_hk_prob, all_hk_true = [], []
        all_eps_pred, all_eps_true = [], []
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"  {desc}", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision and self.device == "cuda"):
                bg_pred, hk_logits, eps_pred = self.model(batch)
                loss, _ = loss_fn(
                    bg_pred, batch["bandgap"],
                    hk_logits, batch["is_high_k"],
                    eps_pred, batch["eps"]
                )

            total_loss    += loss.item()
            n_batches     += 1

            all_bg_pred.extend(bg_pred.squeeze(-1).cpu().numpy())
            all_bg_true.extend(batch["bandgap"].squeeze(-1).cpu().numpy())
            all_hk_prob.extend(F.softmax(hk_logits, dim=1)[:, 1].cpu().numpy())
            all_hk_true.extend(batch["is_high_k"].cpu().numpy())
            all_eps_pred.extend(eps_pred.squeeze(-1).cpu().numpy())
            all_eps_true.extend(batch["eps"].squeeze(-1).cpu().numpy())

        bg_mae  = mean_absolute_error(all_bg_true, all_bg_pred)
        bg_r2   = r2_score(all_bg_true, all_bg_pred)
        eps_mae = mean_absolute_error(all_eps_true, all_eps_pred)

        hk_labels = (np.array(all_hk_prob) > 0.5).astype(int)
        auc = roc_auc_score(all_hk_true, all_hk_prob) if len(np.unique(all_hk_true)) > 1 else 0.0
        acc = (hk_labels == np.array(all_hk_true)).mean()

        return {
            f"{desc}_loss":   total_loss / max(n_batches, 1),
            f"{desc}_mae_bg": bg_mae,
            f"{desc}_r2_bg":  bg_r2,
            f"{desc}_auc_hk": auc,
            f"{desc}_acc_hk": acc,
            f"{desc}_mae_eps": eps_mae,
            "_bg_pred": all_bg_pred, "_bg_true": all_bg_true,
            "_hk_prob": all_hk_prob, "_hk_true": all_hk_true,
            "_eps_pred": all_eps_pred, "_eps_true": all_eps_true,
        }

    def fit(self, train_loader, val_loader, loss_fn: MultiTaskLoss):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training | Device: {self.device} | Epochs: {self.cfg.num_epochs}")
        logger.info(f"{'='*60}")

        # Stage 1: Freeze backbone (if configured)
        if self.cfg.freeze_backbone:
            self.model.freeze_backbone(freeze=True)
            logger.info(f"  Stage 1: Backbone frozen for {self.cfg.freeze_epochs} epochs")

        for epoch in range(1, self.cfg.num_epochs + 1):

            # Unfreeze backbone after warmup stage
            if self.cfg.freeze_backbone and epoch == self.cfg.freeze_epochs + 1:
                self.model.freeze_backbone(freeze=False)
                logger.info(f"  Stage 2: Backbone unfrozen at epoch {epoch}")

            train_metrics = self.train_epoch(train_loader, loss_fn)
            val_metrics   = self.evaluate(val_loader, loss_fn, desc="val")
            lr_now        = self.optimizer.param_groups[0]["lr"]

            # Scheduler step (plateau uses val AUC)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._step_scheduler(val_metrics["val_auc_hk"])

            # Log
            logger.info(
                f"  Epoch {epoch:3d}/{self.cfg.num_epochs} | "
                f"Loss {train_metrics['train_loss']:.4f} | "
                f"Val Loss {val_metrics['val_loss']:.4f} | "
                f"Eg MAE {val_metrics['val_mae_bg']:.3f} eV | "
                f"AUC {val_metrics['val_auc_hk']:.4f} | "
                f"ε MAE {val_metrics['val_mae_eps']:.2f} | "
                f"LR {lr_now:.2e}"
            )

            # History
            for k, v in {**train_metrics, **val_metrics}.items():
                if not k.startswith("_") and k in self.history:
                    self.history[k].append(v)
            self.history["lr"].append(lr_now)

            # Checkpoint: save best by composite score (AUC + R² of band gap)
            composite = val_metrics["val_auc_hk"] + val_metrics["val_r2_bg"]
            if composite > self.best_val_auc + self.best_val_mae_bg:
                self.best_val_auc    = val_metrics["val_auc_hk"]
                self.best_val_mae_bg = val_metrics["val_mae_bg"]
                self.patience_counter = 0
                ckpt_path = Path(self.cfg.checkpoint_dir) / "best_model.pt"
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config":      self.cfg,
                }, ckpt_path)
                logger.info(f"    ✓ Best model saved (AUC={val_metrics['val_auc_hk']:.4f}, "
                            f"Eg MAE={val_metrics['val_mae_bg']:.3f} eV)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        logger.info(f"\n  Best Val AUC: {self.best_val_auc:.4f}  |  Best Eg MAE: {self.best_val_mae_bg:.3f} eV")
        return self.history


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def run_test_evaluation(model: nn.Module, test_loader: DataLoader,
                        loss_fn: MultiTaskLoss, cfg: PipelineConfig):
    """Full test set evaluation with detailed metrics and plots."""
    device = cfg.device
    trainer = Trainer(model, cfg)
    metrics = trainer.evaluate(test_loader, loss_fn, desc="test")

    bg_pred  = np.array(metrics["_bg_pred"])
    bg_true  = np.array(metrics["_bg_true"])
    hk_prob  = np.array(metrics["_hk_prob"])
    hk_true  = np.array(metrics["_hk_true"])
    eps_pred = np.array(metrics["_eps_pred"])
    eps_true = np.array(metrics["_eps_true"])
    hk_pred  = (hk_prob > 0.5).astype(int)

    # Print classification report
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"Band Gap → MAE: {metrics['test_mae_bg']:.3f} eV | R²: {metrics['test_r2_bg']:.4f}")
    logger.info(f"High-k   → AUC: {metrics['test_auc_hk']:.4f} | Acc: {metrics['test_acc_hk']:.4f}")
    logger.info(f"ε_static → MAE: {metrics['test_mae_eps']:.3f}")
    logger.info("\nHigh-k Classification Report:")
    logger.info(classification_report(hk_true, hk_pred, target_names=["Normal-k", "High-k"]))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("JARVIS + ALIGNN Multi-Task: Band Gap + High-k Prediction", fontsize=14, fontweight="bold")

    # 1. Band gap parity plot
    ax = axes[0, 0]
    ax.scatter(bg_true, bg_pred, alpha=0.4, s=12, c=hk_true, cmap="coolwarm", rasterized=True)
    lim = [min(bg_true.min(), bg_pred.min()) - 0.2, max(bg_true.max(), bg_pred.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("DFT Band Gap (eV)")
    ax.set_ylabel("Predicted Band Gap (eV)")
    ax.set_title(f"Band Gap Parity Plot\nMAE={metrics['test_mae_bg']:.3f} eV, R²={metrics['test_r2_bg']:.4f}")
    ax.legend(fontsize=8); ax.set_xlim(lim); ax.set_ylim(lim)

    # 2. ε_static parity plot
    ax = axes[0, 1]
    sc = ax.scatter(eps_true, eps_pred, alpha=0.4, s=12, c=hk_true, cmap="coolwarm", rasterized=True)
    plt.colorbar(sc, ax=ax, label="High-k label")
    lim2 = [0, min(eps_true.max(), eps_pred.max()) * 1.05]
    ax.plot(lim2, lim2, "k--", lw=1.5)
    ax.axvline(x=cfg.highk_threshold, color="r", lw=1, linestyle=":", label=f"ε={cfg.highk_threshold}")
    ax.set_xlabel("DFT ε_static")
    ax.set_ylabel("Predicted ε_static")
    ax.set_title(f"Dielectric Parity Plot\nMAE={metrics['test_mae_eps']:.2f}")
    ax.legend(fontsize=8)

    # 3. High-k ROC curve
    from sklearn.metrics import roc_curve
    ax = axes[0, 2]
    fpr, tpr, _ = roc_curve(hk_true, hk_prob)
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {metrics['test_auc_hk']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("High-k ROC Curve"); ax.legend()

    # 4. Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(hk_true, hk_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal-k", "High-k"],
                yticklabels=["Normal-k", "High-k"])
    ax.set_title("Confusion Matrix"); ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    # 5. High-k probability distribution
    ax = axes[1, 1]
    ax.hist(hk_prob[hk_true == 0], bins=50, alpha=0.6, label="Normal-k", density=True, color="steelblue")
    ax.hist(hk_prob[hk_true == 1], bins=50, alpha=0.6, label="High-k",   density=True, color="tomato")
    ax.axvline(x=0.5, color="k", lw=1.5, linestyle="--")
    ax.set_xlabel("High-k Probability"); ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution"); ax.legend()

    # 6. Band gap vs ε_static scatter (colored by prediction accuracy)
    ax = axes[1, 2]
    err_bg = np.abs(bg_pred - bg_true)
    sc2 = ax.scatter(bg_true, eps_true, c=err_bg, s=10, alpha=0.5,
                     cmap="viridis", rasterized=True)
    plt.colorbar(sc2, ax=ax, label="|ΔEg| (eV)")
    ax.axhline(y=cfg.highk_threshold, color="r", lw=1, linestyle=":", label=f"ε={cfg.highk_threshold}")
    ax.set_xlabel("DFT Band Gap (eV)"); ax.set_ylabel("ε_static")
    ax.set_title("Band Gap vs ε_static (colored by Eg error)"); ax.legend()

    plt.tight_layout()
    plot_path = Path(cfg.results_dir) / "test_evaluation.pdf"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"\nPlots saved to {plot_path}")
    plt.show()

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "bg_true": bg_true, "bg_pred": bg_pred,
        "eps_true": eps_true, "eps_pred": eps_pred,
        "hk_true": hk_true, "hk_pred": hk_pred,
        "hk_prob": hk_prob,
    })
    pred_csv = Path(cfg.results_dir) / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info(f"Predictions saved to {pred_csv}")

    return metrics, pred_df


def plot_training_history(history: dict, cfg: PipelineConfig):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_mae_bg"], color="steelblue", label="Eg MAE (eV)")
    axes[1].set_title("Band Gap MAE"); axes[1].legend()
    ax2 = axes[1].twinx()
    ax2.plot(history["val_auc_hk"], color="tomato", linestyle="--", label="High-k AUC")
    ax2.set_ylabel("AUC"); ax2.legend(loc="lower right")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["lr"])
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(Path(cfg.results_dir) / "training_curves.pdf", dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE UTILITY
# ──────────────────────────────────────────────────────────────────────────────
def predict_single_material(atoms_dict: dict, checkpoint_path: str, cfg: PipelineConfig) -> dict:
    """
    Run inference on a single material given its JARVIS Atoms dict.

    Returns:
        dict with keys: bandgap_eV, is_high_k, highk_probability, eps_static
    """
    model = MultiTaskALIGNN(cfg).to(cfg.device)
    ckpt  = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    g, lg, _ = build_alignn_graphs(atoms_dict, cfg)
    batch = {
        "g":  dgl.batch([g]).to(cfg.device),
        "lg": dgl.batch([lg]).to(cfg.device),
    }

    with torch.no_grad():
        bg_pred, hk_logits, eps_pred = model(batch)

    hk_prob = F.softmax(hk_logits, dim=1)[0, 1].item()
    return {
        "bandgap_eV":       bg_pred.item(),
        "is_high_k":        hk_prob > 0.5,
        "highk_probability": hk_prob,
        "eps_static_pred":  eps_pred.item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="JARVIS + ALIGNN Multi-Task Training")
    parser.add_argument("--epochs",         type=int,   default=CFG.num_epochs)
    parser.add_argument("--batch_size",     type=int,   default=CFG.batch_size)
    parser.add_argument("--lr",             type=float, default=CFG.learning_rate)
    parser.add_argument("--highk_threshold",type=float, default=CFG.highk_threshold)
    parser.add_argument("--no_pretrained",  action="store_true", help="Skip pretrained weight loading")
    parser.add_argument("--freeze",         action="store_true", help="Freeze backbone initially")
    parser.add_argument("--test_only",      type=str,   default=None, help="Path to checkpoint for test-only eval")
    args = parser.parse_args()

    # Apply CLI overrides
    CFG.num_epochs       = args.epochs
    CFG.batch_size       = args.batch_size
    CFG.learning_rate    = args.lr
    CFG.highk_threshold  = args.highk_threshold
    CFG.freeze_backbone  = args.freeze

    torch.manual_seed(CFG.random_seed)
    np.random.seed(CFG.random_seed)

    logger.info(f"Device: {CFG.device}")
    logger.info(f"High-k threshold: ε > {CFG.highk_threshold}")

    # ── 1. Load & filter JARVIS data ──────────────────────────────────────────
    df = load_jarvis_data(CFG)

    # ── 2. Train/Val/Test split ───────────────────────────────────────────────
    df_train, df_temp = train_test_split(df, test_size=CFG.val_ratio + CFG.test_ratio,
                                          random_state=CFG.random_seed, stratify=df["is_high_k"])
    df_val, df_test = train_test_split(df_temp, test_size=CFG.test_ratio / (CFG.val_ratio + CFG.test_ratio),
                                        random_state=CFG.random_seed, stratify=df_temp["is_high_k"])

    logger.info(f"Split sizes — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── 3. Build datasets ─────────────────────────────────────────────────────
    train_dataset = JARVISHighKDataset(df_train, CFG, split="train")
    val_dataset   = JARVISHighKDataset(df_val,   CFG, split="val")
    test_dataset  = JARVISHighKDataset(df_test,  CFG, split="test")

    # Weighted sampler for class imbalance
    sample_weights  = train_dataset.get_sampler_weights()
    sampler         = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    class_weights_t = train_dataset.get_class_weights().to(CFG.device)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=sampler,
                               num_workers=CFG.num_workers, collate_fn=collate_alignn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG.batch_size * 2, shuffle=False,
                               num_workers=CFG.num_workers, collate_fn=collate_alignn, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CFG.batch_size * 2, shuffle=False,
                               num_workers=CFG.num_workers, collate_fn=collate_alignn, pin_memory=True)

    # ── 4. Build model ────────────────────────────────────────────────────────
    model = MultiTaskALIGNN(CFG)

    if not args.no_pretrained:
        model.load_pretrained_backbone(CFG.pretrained_bandgap)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # ── 5. Loss function ──────────────────────────────────────────────────────
    loss_fn = MultiTaskLoss(CFG, class_weights=class_weights_t).to(CFG.device)

    # ── 6. Test-only mode ─────────────────────────────────────────────────────
    if args.test_only:
        ckpt = torch.load(args.test_only, map_location=CFG.device)
        model.load_state_dict(ckpt["model_state"])
        run_test_evaluation(model, test_loader, loss_fn, CFG)
        return

    # ── 7. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, CFG)
    trainer.configure_optimizers(loss_fn, train_dataset)
    history = trainer.fit(train_loader, val_loader, loss_fn)

    # ── 8. Plot training curves ───────────────────────────────────────────────
    plot_training_history(history, CFG)

    # ── 9. Final test evaluation ──────────────────────────────────────────────
    best_ckpt = Path(CFG.checkpoint_dir) / "best_model.pt"
    best_state = torch.load(best_ckpt, map_location=CFG.device)
    model.load_state_dict(best_state["model_state"])
    metrics, pred_df = run_test_evaluation(model, test_loader, loss_fn, CFG)

    # ── 10. Save config snapshot ──────────────────────────────────────────────
    config_path = Path(CFG.results_dir) / "run_config.json"
    import dataclasses
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(CFG), f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Best Eg MAE  : {metrics['test_mae_bg']:.3f} eV")
    logger.info(f"  High-k AUC  : {metrics['test_auc_hk']:.4f}")
    logger.info(f"  ε MAE       : {metrics['test_mae_eps']:.3f}")
    logger.info(f"  Checkpoints : {CFG.checkpoint_dir}")
    logger.info(f"  Results     : {CFG.results_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
