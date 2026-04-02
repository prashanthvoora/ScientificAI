#!/usr/bin/env python3
"""
Enhanced molecular GNN benchmark with MD17 hyperparameter sweep support.

What is new in this version
---------------------------
1. Keeps the original QM9 / MD17 training flow.
2. Adds DimeNet++ support.
3. Adds MD17 hyperparameter sweep mode for DimeNet and DimeNet++.
4. Saves all sweep runs into one CSV.
5. Automatically plots Force MAE vs Training Time.

Recommended use
---------------
For MD17 sweeps, use:
    --dataset md17 --use-forces --sweep-md17

Example:
python train_compare_molecular_gnns_sweep.py \
    --dataset md17 \
    --dataset-root ./data \
    --output-dir ./outputs_md17_sweep \
    --md17-molecule "revised aspirin" \
    --use-forces \
    --epochs 200 \
    --patience 9999 \
    --sweep-md17 \
    --models dimenet,dimenetpp \
    --sweep-lr 1e-4,3e-4 \
    --sweep-batch-size 2,4 \
    --sweep-cutoff 4.0,5.0 \
    --sweep-force-weight 1.0,10.0 \
    --sweep-hidden-channels 64,128 \
    --sweep-num-blocks 3,4 \
    --sweep-num-bilinear 4,8 \
    --sweep-num-spherical 4,7 \
    --sweep-num-radial 6,8 \
    --sweep-max-num-neighbors 32,64 \
    --sweep-int-emb-size 32,64 \
    --sweep-basis-emb-size 8,16 \
    --sweep-out-emb-channels 128,256 \
    --max-sweep-runs 64
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import QM9, MD17
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    CGConv,
    DimeNet,
    DimeNetPlusPlus,
    SchNet,
    global_mean_pool,
    radius_graph,
)


QM9_TARGET_NAMES = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
    12: "U0_atom",
    13: "U_atom",
    14: "H_atom",
    15: "G_atom",
    16: "A",
    17: "B",
    18: "C",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_csv_list(raw: str, cast_fn):
    return [cast_fn(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass
class RunConfig:
    dataset: str
    dataset_root: str
    output_dir: str
    seed: int
    device: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    num_workers: int
    max_samples: Optional[int]
    train_ratio: float
    val_ratio: float
    cutoff: float
    use_forces: bool
    force_weight: float
    qm9_target_index: int
    md17_molecule: str
    models: List[str]
    sweep_md17: bool
    max_sweep_runs: Optional[int]


def split_list(data_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data_list))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_set = [data_list[i] for i in train_idx]
    val_set = [data_list[i] for i in val_idx]
    test_set = [data_list[i] for i in test_idx]
    return train_set, val_set, test_set


def compute_scalar_stats(dataset, attr="y"):
    vals = []
    for d in dataset:
        vals.append(getattr(d, attr).view(-1).float())
    vals = torch.cat(vals, dim=0)
    mean = vals.mean().item()
    std = vals.std().item()
    if std < 1e-12:
        std = 1.0
    return mean, std


def standardize_attr(dataset, attr: str, mean: float, std: float):
    out = []
    for d in dataset:
        d2 = copy.copy(d)
        setattr(d2, attr, (getattr(d2, attr).float() - mean) / std)
        out.append(d2)
    return out


def prepare_qm9(cfg: RunConfig):
    raw = QM9(root=cfg.dataset_root)
    data_list = []
    for d in raw:
        d2 = copy.copy(d)
        d2.y = d.y[:, cfg.qm9_target_index].view(-1)
        data_list.append(d2)

    if cfg.max_samples is not None:
        data_list = data_list[:cfg.max_samples]

    train_set, val_set, test_set = split_list(
        data_list, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed
    )

    y_mean, y_std = compute_scalar_stats(train_set, attr="y")
    train_set = standardize_attr(train_set, "y", y_mean, y_std)
    val_set = standardize_attr(val_set, "y", y_mean, y_std)
    test_set = standardize_attr(test_set, "y", y_mean, y_std)

    stats = {"y_mean": y_mean, "y_std": y_std, "f_mean": None, "f_std": None}
    meta = {
        "dataset_name": "QM9",
        "target_name": QM9_TARGET_NAMES.get(cfg.qm9_target_index, str(cfg.qm9_target_index)),
        "target_type": "scalar_property",
    }
    return train_set, val_set, test_set, stats, meta


def prepare_md17(cfg: RunConfig):
    raw = MD17(root=cfg.dataset_root, name=cfg.md17_molecule)
    data_list = [copy.copy(d) for d in raw]

    if cfg.max_samples is not None:
        data_list = data_list[:cfg.max_samples]

    train_set, val_set, test_set = split_list(
        data_list, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed
    )

    y_mean, y_std = compute_scalar_stats(train_set, attr="energy")
    train_set = standardize_attr(train_set, "energy", y_mean, y_std)
    val_set = standardize_attr(val_set, "energy", y_mean, y_std)
    test_set = standardize_attr(test_set, "energy", y_mean, y_std)

    if cfg.use_forces:
        all_f = torch.cat([d.force.view(-1) for d in train_set], dim=0).float()
        f_mean = all_f.mean().item()
        f_std = all_f.std().item()
        if f_std < 1e-12:
            f_std = 1.0

        def standardize_force(ds):
            out = []
            for d in ds:
                d2 = copy.copy(d)
                d2.force = (d2.force.float() - f_mean) / f_std
                out.append(d2)
            return out

        train_set = standardize_force(train_set)
        val_set = standardize_force(val_set)
        test_set = standardize_force(test_set)
    else:
        f_mean, f_std = None, None

    stats = {"y_mean": y_mean, "y_std": y_std, "f_mean": f_mean, "f_std": f_std}
    meta = {
        "dataset_name": "MD17",
        "target_name": cfg.md17_molecule,
        "target_type": "energy" if not cfg.use_forces else "energy_and_forces",
    }
    return train_set, val_set, test_set, stats, meta


class CGCNNStyle(nn.Module):
    def __init__(self, emb_dim=128, num_layers=4, cutoff=5.0, dropout=0.0, max_num_neighbors=64):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.embedding = nn.Embedding(100, emb_dim)
        self.convs = nn.ModuleList([
            CGConv(channels=emb_dim, dim=1, batch_norm=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, z, pos, batch):
        x = self.embedding(z)
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, dist)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


class SchNetWrapper(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32,
    ):
        super().__init__()
        self.model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout="add",
            dipole=False,
        )

    def forward(self, z, pos, batch):
        return self.model(z=z, pos=pos, batch=batch).view(-1)


class DimeNetWrapper(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_blocks=4,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    ):
        super().__init__()
        self.model = DimeNet(
            hidden_channels=hidden_channels,
            out_channels=1,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, z, pos, batch):
        return self.model(z=z, pos=pos, batch=batch).view(-1)


class DimeNetPPWrapper(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    ):
        super().__init__()
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=1,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, z, pos, batch):
        return self.model(z=z, pos=pos, batch=batch).view(-1)


def build_model(name: str, cfg: RunConfig, hparams: Optional[Dict[str, Any]] = None):
    p = dict(hparams or {})
    if name == "cgcnn":
        return CGCNNStyle(
            emb_dim=p.get("emb_dim", 128),
            num_layers=p.get("num_layers", 4),
            cutoff=p.get("cutoff", cfg.cutoff),
            dropout=p.get("dropout", 0.0),
            max_num_neighbors=p.get("max_num_neighbors", 64),
        )
    if name == "schnet":
        return SchNetWrapper(
            hidden_channels=p.get("hidden_channels", 128),
            num_filters=p.get("num_filters", p.get("hidden_channels", 128)),
            num_interactions=p.get("num_interactions", 6),
            num_gaussians=p.get("num_gaussians", 50),
            cutoff=max(p.get("cutoff", cfg.cutoff), 1e-6),
            max_num_neighbors=p.get("max_num_neighbors", 32),
        )
    if name == "dimenet":
        return DimeNetWrapper(
            hidden_channels=p.get("hidden_channels", 128),
            num_blocks=p.get("num_blocks", 4),
            num_bilinear=p.get("num_bilinear", 8),
            num_spherical=p.get("num_spherical", 7),
            num_radial=p.get("num_radial", 6),
            cutoff=p.get("cutoff", cfg.cutoff),
            max_num_neighbors=p.get("max_num_neighbors", 32),
            envelope_exponent=p.get("envelope_exponent", 5),
            num_before_skip=p.get("num_before_skip", 1),
            num_after_skip=p.get("num_after_skip", 2),
            num_output_layers=p.get("num_output_layers", 3),
        )
    if name in {"dimenetpp", "dimenet++"}:
        return DimeNetPPWrapper(
            hidden_channels=p.get("hidden_channels", 128),
            num_blocks=p.get("num_blocks", 4),
            int_emb_size=p.get("int_emb_size", 64),
            basis_emb_size=p.get("basis_emb_size", 8),
            out_emb_channels=p.get("out_emb_channels", 256),
            num_spherical=p.get("num_spherical", 7),
            num_radial=p.get("num_radial", 6),
            cutoff=p.get("cutoff", cfg.cutoff),
            max_num_neighbors=p.get("max_num_neighbors", 32),
            envelope_exponent=p.get("envelope_exponent", 5),
            num_before_skip=p.get("num_before_skip", 1),
            num_after_skip=p.get("num_after_skip", 2),
            num_output_layers=p.get("num_output_layers", 3),
        )
    raise ValueError(f"Unknown model: {name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": safe_rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def forward_energy_and_forces(model, batch, use_forces: bool):
    if use_forces:
        pos = batch.pos.clone().detach().requires_grad_(True)
    else:
        pos = batch.pos

    energy = model(batch.z, pos, batch.batch)

    if use_forces:
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=pos,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        force_pred = -grad
        return energy, force_pred

    return energy, None


def train_one_epoch_qm9(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.z, batch.pos, batch.batch)
        target = batch.y.view(-1)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        n = target.numel()
        total_loss += loss.item() * n
        total_items += n

    return total_loss / max(total_items, 1)


def train_one_epoch_md17(model, loader, optimizer, device, use_forces: bool, force_weight: float):
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        energy_pred, force_pred = forward_energy_and_forces(model, batch, use_forces=use_forces)
        energy_true = batch.energy.view(-1)

        loss_e = F.mse_loss(energy_pred, energy_true)
        if use_forces:
            loss_f = F.mse_loss(force_pred, batch.force)
            loss = loss_e + force_weight * loss_f
        else:
            loss = loss_e

        loss.backward()
        optimizer.step()

        n = energy_true.numel()
        total_loss += loss.item() * n
        total_items += n

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_qm9(model, loader, device, y_mean, y_std):
    model.eval()
    all_pred, all_true = [], []
    num_graphs = 0
    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.z, batch.pos, batch.batch)
        target = batch.y.view(-1)

        pred_real = pred * y_std + y_mean
        target_real = target * y_std + y_mean

        all_pred.append(pred_real.cpu())
        all_true.append(target_real.cpu())
        num_graphs += batch.num_graphs

    elapsed = time.perf_counter() - start
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    m = regression_metrics(y_true, y_pred)
    m["latency_ms_per_graph"] = (elapsed / max(num_graphs, 1)) * 1000.0
    return m


def evaluate_md17(model, loader, device, y_mean, y_std, use_forces, f_mean, f_std):
    model.eval()
    all_e_pred, all_e_true = [], []
    all_f_pred, all_f_true = [], []
    num_graphs = 0
    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            energy_pred, force_pred = forward_energy_and_forces(model, batch, use_forces=use_forces)

        energy_true = batch.energy.view(-1)
        energy_pred_real = energy_pred * y_std + y_mean
        energy_true_real = energy_true * y_std + y_mean

        all_e_pred.append(energy_pred_real.detach().cpu())
        all_e_true.append(energy_true_real.detach().cpu())

        if use_forces:
            force_pred_real = force_pred * f_std + f_mean
            force_true_real = batch.force * f_std + f_mean
            all_f_pred.append(force_pred_real.detach().cpu().view(-1))
            all_f_true.append(force_true_real.detach().cpu().view(-1))

        num_graphs += batch.num_graphs

    elapsed = time.perf_counter() - start

    e_pred = torch.cat(all_e_pred).numpy()
    e_true = torch.cat(all_e_true).numpy()
    m_e = regression_metrics(e_true, e_pred)
    m_e["latency_ms_per_graph"] = (elapsed / max(num_graphs, 1)) * 1000.0

    if use_forces:
        f_pred = torch.cat(all_f_pred).numpy()
        f_true = torch.cat(all_f_true).numpy()
        m_f = regression_metrics(f_true, f_pred)
        return {
            "energy_mae": m_e["mae"],
            "energy_rmse": m_e["rmse"],
            "energy_r2": m_e["r2"],
            "force_mae": m_f["mae"],
            "force_rmse": m_f["rmse"],
            "force_r2": m_f["r2"],
            "latency_ms_per_graph": m_e["latency_ms_per_graph"],
        }

    return {
        "energy_mae": m_e["mae"],
        "energy_rmse": m_e["rmse"],
        "energy_r2": m_e["r2"],
        "latency_ms_per_graph": m_e["latency_ms_per_graph"],
    }


def train_model(
    cfg: RunConfig,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    stats,
    model_hparams: Optional[Dict[str, Any]] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    force_weight_override: Optional[float] = None,
):
    device = infer_device(cfg.device)
    model = build_model(model_name, cfg, model_hparams).to(device)

    lr = cfg.lr if lr_override is None else lr_override
    force_weight = cfg.force_weight if force_weight_override is None else force_weight_override

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    best_state = None
    best_key = float("inf")
    bad_epochs = 0
    history = []

    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        if cfg.dataset == "qm9":
            train_loss = train_one_epoch_qm9(model, train_loader, optimizer, device)
            val_metrics = evaluate_qm9(model, val_loader, device, stats["y_mean"], stats["y_std"])
            test_metrics = evaluate_qm9(model, test_loader, device, stats["y_mean"], stats["y_std"])
            val_score = val_metrics["mae"]
        else:
            train_loss = train_one_epoch_md17(
                model,
                train_loader,
                optimizer,
                device,
                use_forces=cfg.use_forces,
                force_weight=force_weight,
            )
            val_metrics = evaluate_md17(
                model,
                val_loader,
                device,
                stats["y_mean"],
                stats["y_std"],
                cfg.use_forces,
                stats["f_mean"],
                stats["f_std"],
            )
            test_metrics = evaluate_md17(
                model,
                test_loader,
                device,
                stats["y_mean"],
                stats["y_std"],
                cfg.use_forces,
                stats["f_mean"],
                stats["f_std"],
            )
            val_score = val_metrics["force_mae"] if cfg.use_forces and "force_mae" in val_metrics else val_metrics["energy_mae"]

        scheduler.step(val_score)

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        if model_hparams:
            row.update({f"hp_{k}": v for k, v in model_hparams.items()})
        history.append(row)

        if val_score < best_key:
            best_key = val_score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"[{cfg.dataset}/{model_name}] Epoch {epoch:03d} | TrainLoss={train_loss:.5f} | BestValKey={best_key:.5f}")

        if bad_epochs >= cfg.patience:
            print(f"[{cfg.dataset}/{model_name}] Early stopping at epoch {epoch}")
            break

    train_time_sec = time.perf_counter() - t0
    model.load_state_dict(best_state)

    if cfg.dataset == "qm9":
        final_val = evaluate_qm9(model, val_loader, device, stats["y_mean"], stats["y_std"])
        final_test = evaluate_qm9(model, test_loader, device, stats["y_mean"], stats["y_std"])
        summary = {
            "model": model_name,
            "num_parameters": count_parameters(model),
            "train_time_sec": train_time_sec,
            "best_val_mae": final_val["mae"],
            "test_mae": final_test["mae"],
            "test_rmse": final_test["rmse"],
            "test_r2": final_test["r2"],
            "latency_ms_per_graph": final_test["latency_ms_per_graph"],
            "used_lr": lr,
            "used_batch_size": batch_size_override if batch_size_override is not None else cfg.batch_size,
        }
    else:
        final_val = evaluate_md17(
            model,
            val_loader,
            device,
            stats["y_mean"],
            stats["y_std"],
            cfg.use_forces,
            stats["f_mean"],
            stats["f_std"],
        )
        final_test = evaluate_md17(
            model,
            test_loader,
            device,
            stats["y_mean"],
            stats["y_std"],
            cfg.use_forces,
            stats["f_mean"],
            stats["f_std"],
        )
        summary = {
            "model": model_name,
            "num_parameters": count_parameters(model),
            "train_time_sec": train_time_sec,
            "best_val_energy_mae": final_val["energy_mae"],
            "test_energy_mae": final_test["energy_mae"],
            "test_energy_rmse": final_test["energy_rmse"],
            "test_energy_r2": final_test["energy_r2"],
            "latency_ms_per_graph": final_test["latency_ms_per_graph"],
            "used_lr": lr,
            "used_batch_size": batch_size_override if batch_size_override is not None else cfg.batch_size,
            "used_force_weight": force_weight,
        }
        if cfg.use_forces:
            summary["best_val_force_mae"] = final_val["force_mae"]
            summary["test_force_mae"] = final_test["force_mae"]
            summary["test_force_rmse"] = final_test["force_rmse"]
            summary["test_force_r2"] = final_test["force_r2"]

    if model_hparams:
        summary.update({f"hp_{k}": v for k, v in model_hparams.items()})

    return model, history, summary


def plot_history(history_df: pd.DataFrame, cfg: RunConfig, out_dir: Path):
    plt.figure(figsize=(10, 6))
    if cfg.dataset == "qm9":
        for name, sub in history_df.groupby("model"):
            plt.plot(sub["epoch"], sub["val_mae"], label=f"{name} val")
            plt.plot(sub["epoch"], sub["test_mae"], linestyle="--", label=f"{name} test")
        plt.ylabel("MAE")
        plt.title("QM9 learning curves")
        filename = "learning_curve_mae.png"
    else:
        metric = "val_force_mae" if cfg.use_forces and "val_force_mae" in history_df.columns else "val_energy_mae"
        test_metric = "test_force_mae" if cfg.use_forces and "test_force_mae" in history_df.columns else "test_energy_mae"
        for name, sub in history_df.groupby("model"):
            plt.plot(sub["epoch"], sub[metric], label=f"{name} val")
            plt.plot(sub["epoch"], sub[test_metric], linestyle="--", label=f"{name} test")
        plt.ylabel(metric)
        plt.title("MD17 learning curves")
        filename = "learning_curve_md17.png"

    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_summary(summary_df: pd.DataFrame, cfg: RunConfig, out_dir: Path):
    if cfg.dataset == "qm9":
        metrics = ["test_mae", "test_rmse", "test_r2", "train_time_sec", "latency_ms_per_graph"]
    else:
        metrics = ["test_energy_mae", "test_energy_rmse", "test_energy_r2", "train_time_sec", "latency_ms_per_graph"]
        if cfg.use_forces and "test_force_mae" in summary_df.columns:
            metrics += ["test_force_mae", "test_force_rmse", "test_force_r2"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(summary_df["model"], summary_df[metric])
        plt.ylabel(metric)
        plt.title(f"Model comparison: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=200)
        plt.close()


def plot_force_mae_vs_train_time(summary_df: pd.DataFrame, out_dir: Path, title: str):
    if "test_force_mae" not in summary_df.columns:
        return

    plt.figure(figsize=(10, 6))
    models = summary_df["model"].astype(str).tolist()

    unique_models = []
    for m in models:
        if m not in unique_models:
            unique_models.append(m)

    for model_name in unique_models:
        sub = summary_df[summary_df["model"] == model_name]
        plt.scatter(sub["train_time_sec"], sub["test_force_mae"], s=60, label=model_name)

        for _, row in sub.iterrows():
            label_parts = [str(row["model"])]
            for hp_key in [
                "hp_hidden_channels",
                "hp_num_blocks",
                "hp_cutoff",
                "used_lr",
                "used_batch_size",
                "used_force_weight",
            ]:
                if hp_key in row and pd.notna(row[hp_key]):
                    short = hp_key.replace("hp_", "")
                    label_parts.append(f"{short}={row[hp_key]}")
            label = "\n".join(label_parts[:4])
            plt.annotate(
                label,
                (row["train_time_sec"], row["test_force_mae"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

    plt.xlabel("Training Time (sec)")
    plt.ylabel("Test Force MAE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "force_mae_vs_train_time.png", dpi=200)
    plt.close()


def build_md17_sweep_configs(args) -> List[Dict[str, Any]]:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_runs: List[Dict[str, Any]] = []

    common_grid = {
        "lr": parse_csv_list(args.sweep_lr, float),
        "batch_size": parse_csv_list(args.sweep_batch_size, int),
        "cutoff": parse_csv_list(args.sweep_cutoff, float),
        "force_weight": parse_csv_list(args.sweep_force_weight, float),
        "hidden_channels": parse_csv_list(args.sweep_hidden_channels, int),
        "num_blocks": parse_csv_list(args.sweep_num_blocks, int),
        "num_spherical": parse_csv_list(args.sweep_num_spherical, int),
        "num_radial": parse_csv_list(args.sweep_num_radial, int),
        "max_num_neighbors": parse_csv_list(args.sweep_max_num_neighbors, int),
        "envelope_exponent": parse_csv_list(args.sweep_envelope_exponent, int),
        "num_before_skip": parse_csv_list(args.sweep_num_before_skip, int),
        "num_after_skip": parse_csv_list(args.sweep_num_after_skip, int),
        "num_output_layers": parse_csv_list(args.sweep_num_output_layers, int),
    }

    dimenet_specific = {
        "num_bilinear": parse_csv_list(args.sweep_num_bilinear, int),
    }

    dimenetpp_specific = {
        "int_emb_size": parse_csv_list(args.sweep_int_emb_size, int),
        "basis_emb_size": parse_csv_list(args.sweep_basis_emb_size, int),
        "out_emb_channels": parse_csv_list(args.sweep_out_emb_channels, int),
    }

    for model_name in models:
        model_grid = dict(common_grid)
        if model_name == "dimenet":
            model_grid.update(dimenet_specific)
        elif model_name in {"dimenetpp", "dimenet++"}:
            model_grid.update(dimenetpp_specific)
        else:
            raise ValueError(f"Sweep mode currently supports only dimenet and dimenetpp, got: {model_name}")

        keys = list(model_grid.keys())
        vals = [model_grid[k] for k in keys]
        for combo in itertools.product(*vals):
            row = {"model": model_name}
            for k, v in zip(keys, combo):
                row[k] = v
            all_runs.append(row)

    if args.max_sweep_runs is not None and len(all_runs) > args.max_sweep_runs:
        rng = random.Random(args.seed)
        all_runs = rng.sample(all_runs, args.max_sweep_runs)

    return all_runs


def run_standard_experiment(cfg: RunConfig, args):
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dataset == "qm9":
        train_set, val_set, test_set, stats, meta = prepare_qm9(cfg)
    else:
        train_set, val_set, test_set, stats, meta = prepare_md17(cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    histories = []
    summaries = []

    for model_name in cfg.models:
        model, history, summary = train_model(cfg, model_name, train_loader, val_loader, test_loader, stats)
        histories.extend(history)
        summaries.append(summary)
        torch.save(model.state_dict(), out_dir / f"{cfg.dataset}_{model_name}_best.pt")

    history_df = pd.DataFrame(histories)
    summary_df = pd.DataFrame(summaries)

    sort_col = "test_mae" if cfg.dataset == "qm9" else ("test_force_mae" if cfg.use_forces else "test_energy_mae")
    summary_df = summary_df.sort_values(sort_col, ascending=True)

    history_df.to_csv(out_dir / "epoch_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    plot_history(history_df, cfg, out_dir)
    plot_summary(summary_df, cfg, out_dir)
    if cfg.dataset == "md17" and cfg.use_forces:
        plot_force_mae_vs_train_time(summary_df, out_dir, "MD17 Force MAE vs Training Time")

    print("\n=== DATASET META ===")
    print(json.dumps(meta, indent=2))
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nArtifacts saved to: {out_dir.resolve()}")


def run_md17_sweep(cfg: RunConfig, args):
    if cfg.dataset != "md17":
        raise ValueError("--sweep-md17 currently requires --dataset md17")
    if not cfg.use_forces:
        raise ValueError("--sweep-md17 is intended for force-aware training. Add --use-forces.")

    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, stats, meta = prepare_md17(cfg)
    sweep_runs = build_md17_sweep_configs(args)

    all_summaries = []
    all_histories = []

    for run_idx, sweep_cfg in enumerate(sweep_runs, start=1):
        model_name = sweep_cfg["model"]
        lr = sweep_cfg["lr"]
        batch_size = sweep_cfg["batch_size"]
        force_weight = sweep_cfg["force_weight"]

        model_hparams = {
            k: v for k, v in sweep_cfg.items()
            if k not in {"model", "lr", "batch_size", "force_weight"}
        }

        print(f"\n=== Sweep run {run_idx}/{len(sweep_runs)} ===")
        print(json.dumps(sweep_cfg, indent=2))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

        model, history, summary = train_model(
            cfg=cfg,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            stats=stats,
            model_hparams=model_hparams,
            batch_size_override=batch_size,
            lr_override=lr,
            force_weight_override=force_weight,
        )

        summary["run_id"] = run_idx
        for row in history:
            row["run_id"] = run_idx
        all_summaries.append(summary)
        all_histories.extend(history)

        ckpt_name = f"md17_{model_name}_run{run_idx}.pt"
        torch.save(model.state_dict(), out_dir / ckpt_name)

    summary_df = pd.DataFrame(all_summaries)
    history_df = pd.DataFrame(all_histories)

    if "best_val_force_mae" in summary_df.columns:
        summary_df = summary_df.sort_values(["best_val_force_mae", "test_force_mae", "train_time_sec"], ascending=[True, True, True])
    else:
        summary_df = summary_df.sort_values(["test_energy_mae", "train_time_sec"], ascending=[True, True])

    summary_df.to_csv(out_dir / "md17_sweep_all_runs.csv", index=False)
    history_df.to_csv(out_dir / "md17_sweep_epoch_metrics.csv", index=False)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "md17_sweep_search_space.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": args.models,
            "sweep_lr": args.sweep_lr,
            "sweep_batch_size": args.sweep_batch_size,
            "sweep_cutoff": args.sweep_cutoff,
            "sweep_force_weight": args.sweep_force_weight,
            "sweep_hidden_channels": args.sweep_hidden_channels,
            "sweep_num_blocks": args.sweep_num_blocks,
            "sweep_num_bilinear": args.sweep_num_bilinear,
            "sweep_num_spherical": args.sweep_num_spherical,
            "sweep_num_radial": args.sweep_num_radial,
            "sweep_max_num_neighbors": args.sweep_max_num_neighbors,
            "sweep_int_emb_size": args.sweep_int_emb_size,
            "sweep_basis_emb_size": args.sweep_basis_emb_size,
            "sweep_out_emb_channels": args.sweep_out_emb_channels,
            "sweep_envelope_exponent": args.sweep_envelope_exponent,
            "sweep_num_before_skip": args.sweep_num_before_skip,
            "sweep_num_after_skip": args.sweep_num_after_skip,
            "sweep_num_output_layers": args.sweep_num_output_layers,
            "max_sweep_runs": args.max_sweep_runs,
        }, f, indent=2)

    plot_force_mae_vs_train_time(summary_df, out_dir, "MD17 Sweep: Force MAE vs Training Time")

    plt.figure(figsize=(10, 6))
    for model_name, sub in summary_df.groupby("model"):
        plt.scatter(sub["num_parameters"], sub["test_force_mae"], s=60, label=model_name)
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Force MAE")
    plt.title("MD17 Sweep: Force MAE vs Model Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "force_mae_vs_num_parameters.png", dpi=200)
    plt.close()

    print("\n=== DATASET META ===")
    print(json.dumps(meta, indent=2))
    print("\n=== TOP 10 SWEEP RUNS ===")
    print(summary_df.head(10).to_string(index=False))
    print(f"\nSweep artifacts saved to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["qm9", "md17"], default="qm9")
    parser.add_argument("--dataset-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_molecular_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--use-forces", action="store_true", help="Only relevant for MD17")
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--qm9-target-index", type=int, default=4, help="4 = HOMO-LUMO gap")
    parser.add_argument("--md17-molecule", type=str, default="revised aspirin")
    parser.add_argument("--models", type=str, default="cgcnn,schnet,dimenet", help="Comma-separated model list")

    # Sweep mode
    parser.add_argument("--sweep-md17", action="store_true", help="Run MD17 hyperparameter sweep for DimeNet/DimeNet++")
    parser.add_argument("--max-sweep-runs", type=int, default=None, help="Optional cap on total sweep runs")

    # Sweep search-space args
    parser.add_argument("--sweep-lr", type=str, default="1e-4,3e-4,1e-3")
    parser.add_argument("--sweep-batch-size", type=str, default="2,4,8")
    parser.add_argument("--sweep-cutoff", type=str, default="4.0,5.0,6.0")
    parser.add_argument("--sweep-force-weight", type=str, default="1.0,10.0,50.0")
    parser.add_argument("--sweep-hidden-channels", type=str, default="64,128")
    parser.add_argument("--sweep-num-blocks", type=str, default="3,4")
    parser.add_argument("--sweep-num-bilinear", type=str, default="4,8")
    parser.add_argument("--sweep-num-spherical", type=str, default="4,7")
    parser.add_argument("--sweep-num-radial", type=str, default="6,8")
    parser.add_argument("--sweep-max-num-neighbors", type=str, default="32,64")
    parser.add_argument("--sweep-int-emb-size", type=str, default="32,64")
    parser.add_argument("--sweep-basis-emb-size", type=str, default="8,16")
    parser.add_argument("--sweep-out-emb-channels", type=str, default="128,256")
    parser.add_argument("--sweep-envelope-exponent", type=str, default="5")
    parser.add_argument("--sweep-num-before-skip", type=str, default="1")
    parser.add_argument("--sweep-num-after-skip", type=str, default="2")
    parser.add_argument("--sweep-num-output-layers", type=str, default="3")

    args = parser.parse_args()

    cfg = RunConfig(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        cutoff=args.cutoff,
        use_forces=args.use_forces,
        force_weight=args.force_weight,
        qm9_target_index=args.qm9_target_index,
        md17_molecule=args.md17_molecule,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        sweep_md17=args.sweep_md17,
        max_sweep_runs=args.max_sweep_runs,
    )

    if cfg.sweep_md17:
        run_md17_sweep(cfg, args)
    else:
        run_standard_experiment(cfg, args)


if __name__ == "__main__":
    main()
