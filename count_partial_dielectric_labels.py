#!/usr/bin/env python3
"""Audit stored JARVIS dielectric components without changing input datasets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def count_labels(df):
    """Accept an already loaded Tier-1 or Tier-2 pandas dataframe."""
    required = {"source", "k_total", "k_elec", "k_ionic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; cannot infer components from k_total alone.")
    jarvis = df["source"].astype(str).str.contains("jarvis", case=False, na=False)
    total = pd.to_numeric(df["k_total"], errors="coerce")
    # Count positive finite stored targets; do not recreate a training split.
    eligible = jarvis & np.isfinite(total) & (total > 0)
    electronic = pd.to_numeric(df["k_elec"], errors="coerce")
    ionic = pd.to_numeric(df["k_ionic"], errors="coerce")
    e = np.isfinite(electronic)
    i = np.isfinite(ionic)  # An observed zero is available, not missing.
    counts = {
        "dataframe_rows": len(df),
        "jarvis_rows": int(jarvis.sum()),
        "jarvis_positive_finite_k_total": int(eligible.sum()),
        "complete_response": int((eligible & e & i).sum()),
        "electronic_only": int((eligible & e & ~i).sum()),
        "ionic_only": int((eligible & ~e & i).sum()),
        "neither_component_available": int((eligible & ~e & ~i).sum()),
    }
    counts["partial_response_total"] = counts["electronic_only"] + counts["ionic_only"]
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="highk_project/data",
                        help="Data folder or path to tier1_foundation.h5")
    parser.add_argument("--key", default="data", help="Pandas HDF key (default: data)")
    parser.add_argument("--output", default="partial_response_counts.json")
    args = parser.parse_args()
    supplied = Path(args.input)
    tier1 = supplied / "tier1_foundation.h5" if supplied.is_dir() else supplied
    if not tier1.is_file():
        parser.error(f"Tier-1 file not found: {tier1}")
    result = {
        "scope": "Stored JARVIS rows with positive finite k_total; all partitions combined.",
        "caveat": "These counts describe stored components after deduplication/backfilling, not necessarily native JARVIS provenance or run-specific training membership.",
    }
    files = {"tier1": tier1}
    tier2 = tier1.parent / "tier2_domain.h5"
    if tier2.is_file():
        files["tier2"] = tier2
    else:
        result["tier2_status"] = "tier2_domain.h5 not found; Tier-2 count not inferred."
    for tier, path in files.items():
        result[tier] = {"file": str(path.resolve()),
                        **count_labels(pd.read_hdf(path, key=args.key))}
    output = json.dumps(result, indent=2)
    Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
