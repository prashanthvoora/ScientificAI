#!/usr/bin/env python3
"""Build a reproducible OQMD Tier-1 external benchmark input.

The script queries OQMD's credential-free OPTIMADE structures endpoint,
creates local CIF files, performs conservative quality filtering and exact
within-snapshot deduplication, and writes ``oqmd_selected.csv`` in the schema
accepted by ``prepare_tier1_external_manifest.py``.

It does NOT compare against the model's training data. Structural overlap with
JARVIS/Materials Project is intentionally handled later by
``tier1_external_eval --tier1_external_overlap_policy exclude|fail``.

Dependencies
------------
    pip install requests pandas numpy pymatgen tqdm

Typical use
-----------
    python oqmd_benchmark.py \
      --output-dir external_raw/oqmd_tier1_v1 \
      --max-candidates 50000 \
      --max-rows 5000 \
      --seed 20260729 \
      --require-formation-energy

For an oxide-focused benchmark add:
    --filter 'elements HAS "O"'

The complete API pages used to build the candidate pool are stored as JSONL,
and a metadata JSON records the endpoint, filter, hashes and selection rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    from pymatgen.core import Lattice, Structure
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pymatgen is required. Install with: pip install pymatgen requests pandas numpy tqdm"
    ) from exc

LOG = logging.getLogger("oqmd_benchmark")
DEFAULT_ENDPOINT = "https://oqmd.org/optimade/structures"
DEFAULT_FIELDS = ",".join(
    [
        "id",
        "chemical_formula_reduced",
        "chemical_formula_descriptive",
        "elements",
        "nelements",
        "nsites",
        "lattice_vectors",
        "cartesian_site_positions",
        "species_at_sites",
        "species",
        "_oqmd_delta_e",
        "_oqmd_band_gap",
        "_oqmd_entry_id",
        "_oqmd_calculation_id",
        "_oqmd_icsd_id",
        "_oqmd_prototype",
        "_oqmd_spacegroup",
        "_oqmd_stability",
        "_oqmd_volume",
    ]
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def request_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]],
    timeout: float,
    retries: int,
    backoff: float,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}: {response.text[:200]}",
                    response=response,
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "data" not in payload:
                raise ValueError("OQMD response is not an OPTIMADE data document")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = backoff * (2**attempt)
            LOG.warning("Request failed (%s); retrying in %.1f s", exc, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"OQMD request failed after {retries + 1} attempts: {last_error}")


def iter_optimade_pages(
    endpoint: str,
    query_filter: str,
    page_limit: int,
    max_candidates: int,
    timeout: float,
    retries: int,
    backoff: float,
    user_agent: str,
) -> Iterator[Tuple[Dict[str, Any], str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/vnd.api+json"})
    params: Optional[Dict[str, Any]] = {
        "page_limit": page_limit,
        "response_fields": DEFAULT_FIELDS,
    }
    if query_filter:
        params["filter"] = query_filter

    next_url: Optional[str] = endpoint
    seen_urls: set[str] = set()
    yielded = 0

    while next_url and yielded < max_candidates:
        if next_url in seen_urls:
            raise RuntimeError(f"Pagination loop detected at {next_url}")
        seen_urls.add(next_url)
        payload = request_json(session, next_url, params, timeout, retries, backoff)
        resolved_url = payload.get("links", {}).get("self") or next_url
        yield payload, str(resolved_url)
        yielded += len(payload.get("data", []))

        nxt = payload.get("links", {}).get("next")
        if isinstance(nxt, dict):
            nxt = nxt.get("href")
        if not nxt:
            break
        next_url = urljoin(next_url, str(nxt))
        params = None  # next link already contains pagination and filter parameters


def resolve_species(attrs: Dict[str, Any], reject_disorder: bool) -> Optional[List[str]]:
    site_names = attrs.get("species_at_sites")
    if not isinstance(site_names, list) or not site_names:
        return None

    definitions = {}
    for item in attrs.get("species") or []:
        if isinstance(item, dict) and item.get("name"):
            definitions[str(item["name"])] = item

    resolved: List[str] = []
    for site_name in site_names:
        name = str(site_name)
        spec = definitions.get(name)
        if spec is None:
            # Most ordered OPTIMADE providers use the element symbol directly.
            resolved.append(name)
            continue
        symbols = list(spec.get("chemical_symbols") or [])
        concentrations = list(spec.get("concentration") or [])
        if len(symbols) == 1 and (not concentrations or abs(float(concentrations[0]) - 1.0) < 1e-8):
            resolved.append(str(symbols[0]))
            continue
        if reject_disorder:
            return None
        # Pymatgen accepts an occupancy dictionary for disordered structures,
        # but this builder deliberately emits only ordered structures by default.
        return None
    return resolved


def structure_from_attributes(attrs: Dict[str, Any], reject_disorder: bool) -> Optional[Structure]:
    lattice = attrs.get("lattice_vectors")
    positions = attrs.get("cartesian_site_positions")
    species = resolve_species(attrs, reject_disorder=reject_disorder)
    if not isinstance(lattice, list) or len(lattice) != 3:
        return None
    if not isinstance(positions, list) or not positions or species is None:
        return None
    if len(positions) != len(species):
        return None
    try:
        matrix = np.asarray(lattice, dtype=float)
        coords = np.asarray(positions, dtype=float)
        if matrix.shape != (3, 3) or coords.ndim != 2 or coords.shape[1] != 3:
            return None
        if not np.isfinite(matrix).all() or not np.isfinite(coords).all():
            return None
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            return None
        struct = Structure(Lattice(matrix), species, coords, coords_are_cartesian=True)
        if len(struct) == 0 or struct.volume <= 1e-6:
            return None
        return struct
    except Exception:
        return None


def structure_fingerprint(structure: Structure, decimals: int = 5) -> str:
    """Deterministic exact-ish fingerprint for within-OQMD duplicate removal.

    This is intentionally conservative and is not a replacement for the later
    StructureMatcher-based overlap audit against the training cache.
    """
    s = structure.get_sorted_structure()
    frac = np.mod(np.asarray(s.frac_coords), 1.0)
    rows = []
    for site, xyz in zip(s.sites, frac):
        rows.append((site.species_string, *np.round(xyz, decimals).tolist()))
    rows.sort()
    payload = {
        "formula": s.composition.reduced_formula,
        "lattice": np.round(np.asarray(s.lattice.matrix), decimals).tolist(),
        "sites": rows,
    }
    return stable_json_hash(payload)


def target_bin(series: pd.Series, n_bins: int) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() < max(2, n_bins):
        return pd.Series("missing", index=series.index, dtype="object")
    try:
        bins = pd.qcut(valid, q=n_bins, duplicates="drop")
        return bins.astype(str).fillna("missing")
    except (ValueError, TypeError):
        return pd.Series("all", index=series.index, dtype="object")


def stratified_select(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.sort_values("source_material_id").reset_index(drop=True)

    work = df.copy()
    work["_ef_bin"] = target_bin(work["formation_energy_per_atom"], 8)
    work["_bg_bin"] = target_bin(work["band_gap"], 6)
    work["_size_bin"] = pd.cut(
        pd.to_numeric(work["nsites"], errors="coerce"),
        bins=[-np.inf, 2, 4, 8, 16, 32, 64, np.inf],
        labels=False,
    ).fillna(-1).astype(int)
    work["_chem_bin"] = pd.to_numeric(work["nelements"], errors="coerce").fillna(-1).astype(int)
    strata_cols = ["_chem_bin", "_size_bin", "_ef_bin", "_bg_bin"]

    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for idx, key in zip(work.index, work[strata_cols].itertuples(index=False, name=None)):
        groups[key].append(int(idx))
    for idxs in groups.values():
        rng.shuffle(idxs)

    chosen: List[int] = []
    active = sorted(groups.keys(), key=str)
    # Round-robin across strata gives small chemistry/target bins a chance to
    # survive while remaining deterministic under a frozen API snapshot.
    while active and len(chosen) < max_rows:
        next_active = []
        for key in active:
            idxs = groups[key]
            if idxs and len(chosen) < max_rows:
                chosen.append(idxs.pop())
            if idxs:
                next_active.append(key)
        active = next_active

    selected = work.loc[chosen].copy()
    selected = selected.drop(columns=["_ef_bin", "_bg_bin", "_size_bin", "_chem_bin"])
    return selected.sort_values("source_material_id").reset_index(drop=True)


def parse_record(
    resource: Dict[str, Any],
    structures_dir: Path,
    reject_disorder: bool,
    min_atoms: int,
    max_atoms: int,
    max_abs_formation_energy: float,
    require_formation_energy: bool,
    require_band_gap: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    attrs = resource.get("attributes") or {}
    resource_id = str(resource.get("id") or attrs.get("_oqmd_entry_id") or "").strip()
    if not resource_id:
        return None, "missing_id"

    structure = structure_from_attributes(attrs, reject_disorder=reject_disorder)
    if structure is None:
        return None, "invalid_or_disordered_structure"
    nsites = len(structure)
    if nsites < min_atoms or nsites > max_atoms:
        return None, "atom_count"

    formation = finite_float(attrs.get("_oqmd_delta_e"))
    band_gap = finite_float(attrs.get("_oqmd_band_gap"))
    if require_formation_energy and not math.isfinite(formation):
        return None, "missing_formation_energy"
    if require_band_gap and not math.isfinite(band_gap):
        return None, "missing_band_gap"
    if math.isfinite(formation) and abs(formation) > max_abs_formation_energy:
        return None, "formation_energy_range"
    if math.isfinite(band_gap) and band_gap < 0:
        return None, "negative_band_gap"

    entry_id = attrs.get("_oqmd_entry_id")
    source_id = f"oqmd-{entry_id}" if entry_id is not None else f"oqmd-{resource_id}"
    cif_name = f"{source_id}.cif"
    cif_path = structures_dir / cif_name
    try:
        structure.to(filename=str(cif_path), fmt="cif")
    except Exception:
        return None, "cif_write_failed"

    formula = attrs.get("chemical_formula_reduced") or structure.composition.reduced_formula
    elements = attrs.get("elements") or [str(x) for x in structure.composition.elements]
    row = {
        "benchmark_row_id": source_id,
        "source_material_id": source_id,
        "oqmd_resource_id": resource_id,
        "oqmd_entry_id": entry_id,
        "oqmd_calculation_id": attrs.get("_oqmd_calculation_id"),
        "icsd_id": attrs.get("_oqmd_icsd_id"),
        "structure_path": str(Path("structures") / cif_name),
        "formula": formula,
        "formation_energy_per_atom": formation,
        "band_gap": band_gap,
        "k_total": float("nan"),
        "dft_functional": "PBE",
        "functional_code": 1,
        "nelements": attrs.get("nelements", len(elements)),
        "nsites": nsites,
        "elements": "-".join(map(str, elements)),
        "spacegroup": attrs.get("_oqmd_spacegroup"),
        "prototype": attrs.get("_oqmd_prototype"),
        "stability": finite_float(attrs.get("_oqmd_stability")),
        "volume": finite_float(attrs.get("_oqmd_volume", structure.volume)),
        "structure_fingerprint": structure_fingerprint(structure),
    }
    return row, "accepted"


def remove_unselected_cifs(all_rows: pd.DataFrame, selected: pd.DataFrame, root: Path) -> int:
    keep = set(selected["structure_path"].astype(str))
    removed = 0
    for rel in all_rows["structure_path"].astype(str):
        if rel not in keep:
            path = root / rel
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError:
                LOG.warning("Could not remove unselected CIF: %s", path)
    return removed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build oqmd_selected.csv and local CIFs for Tier-1 external evaluation."
    )
    parser.add_argument("--output-dir", required=True, help="Output benchmark directory")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--filter",
        default="",
        help='OPTIMADE filter, e.g. \'elements HAS "O" AND _oqmd_delta_e IS KNOWN\'',
    )
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--max-abs-formation-energy", type=float, default=20.0)
    parser.add_argument("--require-formation-energy", action="store_true")
    parser.add_argument("--require-band-gap", action="store_true")
    parser.add_argument(
        "--allow-disordered",
        action="store_true",
        help="Reserved for future use; current implementation still rejects mixed-occupancy sites.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--sleep-between-pages", type=float, default=0.2)
    parser.add_argument(
        "--keep-unselected-cifs",
        action="store_true",
        help="Keep CIFs for accepted candidates that were not selected into max_rows.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    if args.max_candidates <= 0 or args.max_rows <= 0:
        raise ValueError("--max-candidates and --max-rows must be positive")
    if args.max_rows > args.max_candidates:
        LOG.warning("max_rows exceeds max_candidates; reducing max_rows to max_candidates")
        args.max_rows = args.max_candidates
    if args.page_limit < 1:
        raise ValueError("--page-limit must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    structures_dir = output_dir / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "oqmd_api_snapshot.jsonl"
    selected_csv = output_dir / "oqmd_selected.csv"
    candidates_csv = output_dir / "oqmd_candidates_quality_filtered.csv"
    rejected_csv = output_dir / "oqmd_rejections.csv"
    metadata_path = output_dir / "oqmd_benchmark_metadata.json"

    LOG.info("OQMD endpoint: %s", args.endpoint)
    LOG.info("OPTIMADE filter: %s", args.filter or "<none>")
    LOG.info("Candidate cap=%d, selected cap=%d", args.max_candidates, args.max_rows)

    accepted_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    n_resources = 0
    page_urls: List[str] = []
    started = time.time()

    with raw_jsonl.open("w", encoding="utf-8") as raw_fh:
        progress = tqdm(total=args.max_candidates, desc="OQMD candidates", unit="row")
        try:
            for page_number, (payload, page_url) in enumerate(
                iter_optimade_pages(
                    endpoint=args.endpoint,
                    query_filter=args.filter,
                    page_limit=args.page_limit,
                    max_candidates=args.max_candidates,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                    user_agent="ScientificAI-Tier1ExternalBenchmark/1.0",
                ),
                start=1,
            ):
                page_urls.append(page_url)
                resources = payload.get("data", [])
                for resource in resources:
                    if n_resources >= args.max_candidates:
                        break
                    n_resources += 1
                    raw_fh.write(json.dumps(resource, sort_keys=True) + "\n")
                    row, reason = parse_record(
                        resource,
                        structures_dir=structures_dir,
                        reject_disorder=not args.allow_disordered,
                        min_atoms=args.min_atoms,
                        max_atoms=args.max_atoms,
                        max_abs_formation_energy=args.max_abs_formation_energy,
                        require_formation_energy=args.require_formation_energy,
                        require_band_gap=args.require_band_gap,
                    )
                    resource_id = str(resource.get("id", ""))
                    if row is None:
                        rejected.append({"resource_id": resource_id, "reason": reason})
                    elif row["structure_fingerprint"] in seen_fingerprints:
                        rejected.append({"resource_id": resource_id, "reason": "within_oqmd_duplicate"})
                        try:
                            (output_dir / row["structure_path"]).unlink(missing_ok=True)
                        except OSError:
                            pass
                    else:
                        seen_fingerprints.add(row["structure_fingerprint"])
                        accepted_rows.append(row)
                    progress.update(1)
                raw_fh.flush()
                LOG.info(
                    "Page %d: resources=%d accepted=%d rejected=%d",
                    page_number,
                    n_resources,
                    len(accepted_rows),
                    len(rejected),
                )
                if n_resources >= args.max_candidates:
                    break
                time.sleep(max(0.0, args.sleep_between_pages))
        finally:
            progress.close()

    if not accepted_rows:
        raise RuntimeError(
            "No usable OQMD structures were collected. Check network access, endpoint/filter, "
            "and rerun with --log-level DEBUG."
        )

    candidates = pd.DataFrame(accepted_rows)
    candidates = candidates.sort_values("source_material_id").reset_index(drop=True)
    candidates.to_csv(candidates_csv, index=False)
    pd.DataFrame(rejected, columns=["resource_id", "reason"]).to_csv(rejected_csv, index=False)

    selected = stratified_select(candidates, max_rows=args.max_rows, seed=args.seed)
    selected.insert(1, "source_database", "OQMD")
    # Keep source-facing CSV compact but retain useful audit metadata.
    ordered = [
        "benchmark_row_id",
        "source_database",
        "source_material_id",
        "structure_path",
        "formula",
        "formation_energy_per_atom",
        "band_gap",
        "k_total",
        "dft_functional",
        "functional_code",
        "oqmd_resource_id",
        "oqmd_entry_id",
        "oqmd_calculation_id",
        "icsd_id",
        "nelements",
        "nsites",
        "elements",
        "spacegroup",
        "prototype",
        "stability",
        "volume",
        "structure_fingerprint",
    ]
    selected = selected[[c for c in ordered if c in selected.columns]]
    selected.to_csv(selected_csv, index=False)

    removed_cifs = 0
    if not args.keep_unselected_cifs:
        removed_cifs = remove_unselected_cifs(candidates, selected, output_dir)

    rejection_counts = Counter(item["reason"] for item in rejected)
    metadata = {
        "builder_version": "1.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oqmd_endpoint": args.endpoint,
        "optimade_filter": args.filter,
        "response_fields": DEFAULT_FIELDS.split(","),
        "page_limit": args.page_limit,
        "page_urls": page_urls,
        "selection_seed": args.seed,
        "max_candidates": args.max_candidates,
        "max_rows": args.max_rows,
        "quality_rules": {
            "min_atoms": args.min_atoms,
            "max_atoms": args.max_atoms,
            "max_abs_formation_energy": args.max_abs_formation_energy,
            "require_formation_energy": args.require_formation_energy,
            "require_band_gap": args.require_band_gap,
            "reject_disordered": not args.allow_disordered,
            "negative_band_gap_rejected": True,
            "within_oqmd_exact_fingerprint_dedup": True,
        },
        "counts": {
            "api_resources_scanned": n_resources,
            "quality_filtered_unique_candidates": len(candidates),
            "selected_rows": len(selected),
            "rejected": len(rejected),
            "unselected_cifs_removed": removed_cifs,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "target_counts_selected": {
            "formation_energy_per_atom": int(selected["formation_energy_per_atom"].notna().sum()),
            "band_gap": int(selected["band_gap"].notna().sum()),
            "k_total": int(selected["k_total"].notna().sum()),
        },
        "files": {
            "selected_csv": str(selected_csv),
            "selected_csv_sha256": file_sha256(selected_csv),
            "candidate_csv": str(candidates_csv),
            "candidate_csv_sha256": file_sha256(candidates_csv),
            "raw_snapshot_jsonl": str(raw_jsonl),
            "raw_snapshot_sha256": file_sha256(raw_jsonl),
            "rejections_csv": str(rejected_csv),
            "rejections_csv_sha256": file_sha256(rejected_csv),
        },
        "runtime_seconds": round(time.time() - started, 3),
        "license_note": "OQMD data are provided under CC-BY 4.0; cite OQMD in derived work.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOG.info("Selected benchmark: %s", selected_csv)
    LOG.info("Rows selected: %d", len(selected))
    LOG.info("Structures retained: %d", len(list(structures_dir.glob("*.cif"))))
    LOG.info("Selected CSV SHA-256: %s", metadata["files"]["selected_csv_sha256"])
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted by user; partial raw snapshot and CIFs may remain")
        raise SystemExit(130)
