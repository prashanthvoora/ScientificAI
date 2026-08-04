#!/usr/bin/env python3
"""Build and freeze a Tier-2 external dielectric benchmark.

Primary source:
  Takahashi et al. oxide dielectric database
  https://github.com/takahashi-akira-36m/oxi_diel_db

Harmonization follows the ALIGNN/JARVIS DFPT scalar target convention:
  k_total = spherical_average(electronic tensor)
          + spherical_average(ionic tensor)

The source database exposes epsilon_electronic_avg and
 epsilon_ionic_avg inside the dielectric object.  If absent, this script falls back to trace/3 of the
 corresponding 3x3 tensors.

Outputs:
  tier2_external_selected.csv
  tier2_external_manifest.csv
  tier2_external_manifest.csv.metadata.json
  tier2_external_rejections.csv
  tier2_external_build_metadata.json
  structures/*.cif

No training data or model predictions are used during benchmark creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

REPO_ZIP_URL = "https://codeload.github.com/takahashi-akira-36m/oxi_diel_db/zip/refs/heads/master"
SOURCE_DOI = "10.1103/PhysRevMaterials.4.103801"
DATASET_DOI = "10.17632/m5jhkc3p9d.1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def tensor_avg(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        arr = np.asarray(x, dtype=float)
        if arr.shape == (3, 3):
            return float(np.trace(arr) / 3.0)
        if arr.ndim == 1 and arr.size == 3:
            return float(np.mean(arr))
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return finite_float(x)


def find_value(d: Dict[str, Any], paths: Iterable[Tuple[str, ...]]) -> Any:
    for path in paths:
        cur: Any = d
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # Most source files are one material per JSON.  Support wrapped lists too.
        for key in ("data", "records", "materials", "entries"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return [obj]
    return []


def download_and_extract(out: Path, force: bool) -> Path:
    repo_dir = out / "source_repo"
    if repo_dir.exists() and not force:
        return repo_dir
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "oxi_diel_db_master.zip"
    print(f"Downloading {REPO_ZIP_URL}")
    with requests.get(REPO_ZIP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out / ".extract_tmp")
    roots = list((out / ".extract_tmp").glob("oxi_diel_db-*"))
    if not roots:
        raise RuntimeError("Downloaded archive did not contain oxi_diel_db-* root")
    shutil.move(str(roots[0]), str(repo_dir))
    shutil.rmtree(out / ".extract_tmp", ignore_errors=True)
    return repo_dir


def parse_record(rec: Dict[str, Any], source_file: str) -> Tuple[Optional[Dict[str, Any]], str]:
    struct_dict = find_value(rec, [("structure",), ("optimized_structure",), ("final_structure",)])
    if not isinstance(struct_dict, dict):
        return None, "missing_structure"
    try:
        structure = Structure.from_dict(struct_dict)
    except Exception:
        return None, "invalid_structure"
    if not structure.is_ordered:
        return None, "disordered_structure"

    diel = rec.get("dielectric", {}) if isinstance(rec.get("dielectric"), dict) else {}
    elec = finite_float(find_value(rec, [
        ("dielectric", "epsilon_electronic_avg"),
        ("dielectric", "dielectric_electronic_avg"),
        ("epsilon_electronic_avg",),
        ("dielectric_electronic_avg",),
    ]))
    ionic = finite_float(find_value(rec, [
        ("dielectric", "epsilon_ionic_avg"),
        ("dielectric", "dielectric_ionic_avg"),
        ("epsilon_ionic_avg",),
        ("dielectric_ionic_avg",),
    ]))
    if not math.isfinite(elec):
        elec = tensor_avg(find_value(rec, [
            ("dielectric", "epsilon_electronic"),
            ("dielectric", "dielectric_electronic"),
            ("epsilon_electronic",),
            ("dielectric_electronic",),
        ]))
    if not math.isfinite(ionic):
        ionic = tensor_avg(find_value(rec, [
            ("dielectric", "epsilon_ionic"),
            ("dielectric", "dielectric_ionic"),
            ("epsilon_ionic",),
            ("dielectric_ionic",),
        ]))
    if not (math.isfinite(elec) and math.isfinite(ionic)):
        return None, "missing_dielectric_component"
    k_total = elec + ionic
    if not math.isfinite(k_total) or k_total <= 0:
        return None, "invalid_k_total"

    mp_id = str(rec.get("mp_id", rec.get("material_id", rec.get("id", Path(source_file).stem))))
    band_gap = finite_float(rec.get("band_gap"))
    lowest_freq = finite_float(find_value(rec, [("phonon", "lowest_freq"), ("lowest_freq",)]))
    formula = structure.composition.reduced_formula
    sg_raw = rec.get("spacegroup", "")
    if isinstance(sg_raw, dict):
        sg_value = str(sg_raw.get("symbol", sg_raw.get("number", "")))
    else:
        sg_value = str(sg_raw)
    return {
        "source_material_id": mp_id,
        "formula": formula,
        "structure": structure,
        "n_atoms": int(len(structure)),
        "n_elements": int(len(structure.composition.elements)),
        "space_group": sg_value,
        "band_gap": band_gap,
        "k_electronic": elec,
        "k_ionic": ionic,
        "k_total": k_total,
        "k_total_log": float(np.log(k_total)),
        "lowest_phonon_frequency_THz": lowest_freq,
        "source_json": source_file,
    }, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Existing oxi_diel_db repository/data directory; otherwise download GitHub archive")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--min-k", type=float, default=3.9,
                    help="Tier-2 domain floor; default matches the v4.60.3 Tier-2 training filter")
    ap.add_argument("--max-k", type=float, default=500.0)
    ap.add_argument("--min-band-gap", type=float, default=1.0,
                    help="Default matches Tier-2 insulating-material filter")
    ap.add_argument("--allow-missing-band-gap", action="store_true")
    ap.add_argument("--min-atoms", type=int, default=2)
    ap.add_argument("--max-atoms", type=int, default=100)
    ap.add_argument("--require-dynamic-stability", action="store_true")
    ap.add_argument("--min-phonon-frequency", type=float, default=-0.3,
                    help="THz threshold when --require-dynamic-stability is enabled")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 keeps all accepted rows; positive value uses deterministic stratified selection")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--deduplicate-structures", action="store_true",
                    help="More expensive within-source StructureMatcher deduplication")
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    structures_dir = out / "structures"
    structures_dir.mkdir(exist_ok=True)

    source_root = args.source_dir.resolve() if args.source_dir else download_and_extract(out, args.force_download)
    json_files = sorted(source_root.rglob("*.json"))
    # Prefer database data, avoid model configs when a data subtree exists.
    data_json = [p for p in json_files if "/data/" in p.as_posix() or p.parent.name == "data"]
    if data_json:
        json_files = data_json
    if not json_files:
        raise FileNotFoundError(f"No JSON database files found under {source_root}")

    accepted: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    seen_ids = set()
    for path in tqdm(json_files, desc="Parsing dielectric database JSON"):
        try:
            records = load_records(path)
        except Exception as exc:
            rejects.append({"source_json": str(path), "reason": "json_parse_error", "detail": str(exc)[:160]})
            continue
        for rec in records:
            row, reason = parse_record(rec, str(path.relative_to(source_root)))
            if row is None:
                rejects.append({"source_json": str(path.relative_to(source_root)), "reason": reason})
                continue
            rid = row["source_material_id"]
            if rid in seen_ids:
                rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "duplicate_source_id"})
                continue
            seen_ids.add(rid)
            if not (args.min_atoms <= row["n_atoms"] <= args.max_atoms):
                rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "atom_count"})
                continue
            if not (args.min_k <= row["k_total"] <= args.max_k):
                rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "k_domain_filter"})
                continue
            if math.isfinite(row["band_gap"]):
                if row["band_gap"] < args.min_band_gap:
                    rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "band_gap_domain_filter"})
                    continue
            elif not args.allow_missing_band_gap:
                rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "missing_band_gap"})
                continue
            if args.require_dynamic_stability:
                lf = row["lowest_phonon_frequency_THz"]
                if not math.isfinite(lf) or lf < args.min_phonon_frequency:
                    rejects.append({"source_json": row["source_json"], "source_material_id": rid, "reason": "dynamic_stability"})
                    continue
            accepted.append(row)

    if args.deduplicate_structures and accepted:
        matcher = StructureMatcher(ltol=0.1, stol=0.15, angle_tol=3,
                                   primitive_cell=True, scale=True, attempt_supercell=False)
        unique: List[Dict[str, Any]] = []
        by_formula: Dict[str, List[Dict[str, Any]]] = {}
        for row in tqdm(accepted, desc="Within-source structural dedup"):
            dup = False
            for prev in by_formula.get(row["formula"], []):
                try:
                    if matcher.fit(row["structure"], prev["structure"]):
                        dup = True
                        break
                except Exception:
                    continue
            if dup:
                rejects.append({"source_json": row["source_json"], "source_material_id": row["source_material_id"], "reason": "within_source_structure_duplicate"})
            else:
                unique.append(row)
                by_formula.setdefault(row["formula"], []).append(row)
        accepted = unique

    # Deterministic stratified down-selection, if requested.
    if args.max_rows and len(accepted) > args.max_rows:
        frame = pd.DataFrame([{k: v for k, v in r.items() if k != "structure"} for r in accepted])
        frame["k_bin"] = pd.qcut(frame["k_total"], q=min(8, frame["k_total"].nunique()), duplicates="drop", labels=False)
        frame["gap_bin"] = pd.qcut(frame["band_gap"], q=min(6, frame["band_gap"].nunique()), duplicates="drop", labels=False)
        frame["stratum"] = frame["n_elements"].astype(str) + "_" + frame["k_bin"].astype(str) + "_" + frame["gap_bin"].astype(str)
        rng = np.random.default_rng(args.seed)
        chosen = []
        groups = list(frame.groupby("stratum").groups.values())
        # Round-robin randomized within each stratum preserves coverage.
        shuffled = [list(rng.permutation(list(g))) for g in groups]
        while len(chosen) < args.max_rows and any(shuffled):
            for g in shuffled:
                if g and len(chosen) < args.max_rows:
                    chosen.append(g.pop())
        accepted = [accepted[int(i)] for i in chosen]

    selected_rows = []
    for i, row in enumerate(tqdm(accepted, desc="Writing benchmark CIFs")):
        source_id = row["source_material_id"]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_id)
        cif_rel = Path("structures") / f"{safe_id}.cif"
        row["structure"].to(filename=str(out / cif_rel))
        selected_rows.append({
            "benchmark_row_id": f"T2OXI-{i:05d}",
            "source_database": "Takahashi_Oxide_Dielectric_DB",
            "source_material_id": source_id,
            "structure_path": cif_rel.as_posix(),
            "formula": row["formula"],
            "space_group": row["space_group"],
            "n_atoms": row["n_atoms"],
            "n_elements": row["n_elements"],
            "k_electronic": row["k_electronic"],
            "k_ionic": row["k_ionic"],
            "k_total": row["k_total"],
            "k_total_log": row["k_total_log"],
            "band_gap": row["band_gap"],
            "formation_energy_per_atom": np.nan,
            "lowest_phonon_frequency_THz": row["lowest_phonon_frequency_THz"],
            "dft_functional": "PBE_VASP_DFPT",
            "functional_code": 1,
            "target_harmonization": "k_total=dielectric_electronic_avg+dielectric_ionic_avg",
            "source_reference_doi": SOURCE_DOI,
            "dataset_doi": DATASET_DOI,
            "source_json": row["source_json"],
        })

    selected_df = pd.DataFrame(selected_rows)
    rejection_counts = Counter(r.get("reason", "unknown") for r in rejects)
    if selected_df.empty:
        pd.DataFrame(rejects).to_csv(out / "tier2_external_rejections.csv", index=False)
        raise RuntimeError(
            "Tier-2 benchmark selection produced zero rows. "
            f"Top rejection reasons: {dict(rejection_counts.most_common(10))}. "
            "Inspect tier2_external_rejections.csv and source schema."
        )
    selected_path = out / "tier2_external_selected.csv"
    manifest_path = out / "tier2_external_manifest.csv"
    selected_df.to_csv(selected_path, index=False)
    selected_df.to_csv(manifest_path, index=False)
    pd.DataFrame(rejects).to_csv(out / "tier2_external_rejections.csv", index=False)

    manifest_hash = sha256_file(manifest_path)
    metadata = {
        "benchmark_version": "tier2_external_oxide_dielectric_v1",
        "source_repository": "https://github.com/takahashi-akira-36m/oxi_diel_db",
        "source_reference_doi": SOURCE_DOI,
        "dataset_doi": DATASET_DOI,
        "source_root": str(source_root),
        "json_files_scanned": len(json_files),
        "selected_rows": int(len(selected_df)),
        "target_counts": {
            "k_total": int(selected_df["k_total"].notna().sum()) if len(selected_df) else 0,
            "band_gap": int(selected_df["band_gap"].notna().sum()) if len(selected_df) else 0,
            "formation_energy_per_atom": 0,
        },
        "quality_rules": vars(args) | {
            "output_dir": str(args.output_dir),
            "source_dir": str(args.source_dir) if args.source_dir else None,
        },
        "rejection_counts": dict(rejection_counts),
        "harmonization": {
            "electronic_scalar": "source dielectric_electronic_avg; fallback trace(tensor)/3",
            "ionic_scalar": "source dielectric_ionic_avg; fallback trace(tensor)/3",
            "k_total": "k_electronic + k_ionic",
            "training_target": "natural_log(k_total)",
            "headline_evaluation": "linear k_total MAE/RMSE/MAD::MAE",
        },
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
    }
    (out / "tier2_external_manifest.csv.metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (out / "tier2_external_build_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "selected_rows": len(selected_df),
        "k_total_range": [float(selected_df.k_total.min()), float(selected_df.k_total.max())] if len(selected_df) else None,
        "band_gap_range": [float(selected_df.band_gap.min()), float(selected_df.band_gap.max())] if len(selected_df) else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
