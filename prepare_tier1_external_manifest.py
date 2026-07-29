#!/usr/bin/env python3
"""Prepare and freeze a Tier-1 external benchmark manifest.

This utility performs source-agnostic schema normalization only. It does not
inspect model predictions or tune thresholds. Structural overlap is checked by
the main training script during tier1_external_eval.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    "benchmark_row_id": ["benchmark_row_id", "external_row_id", "source_material_id", "material_id", "oqmd_id", "aflow_id", "row_id"],
    "source_material_id": ["source_material_id", "material_id", "oqmd_id", "aflow_id", "benchmark_row_id", "row_id"],
    "structure_path": ["structure_path", "cif_path", "cif", "file"],
    "formula": ["formula", "reduced_formula", "composition"],
    "formation_energy_per_atom": ["formation_energy_per_atom", "target_ef", "formation_energy", "delta_e"],
    "band_gap": ["band_gap", "target_bg", "bandgap", "gap"],
    "k_total": ["k_total", "target_k", "dielectric_constant", "epsilon_total"],
}

def pick(df, names):
    return next((x for x in names if x in df.columns), None)

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--source-name', required=True)
    ap.add_argument('--dft-functional', default='PBE')
    ap.add_argument('--functional-code', type=int, default=1)
    ap.add_argument('--structure-root', default='')
    args=ap.parse_args()
    inp,out=Path(args.input),Path(args.output)
    df=pd.read_json(inp, lines=inp.suffix.lower()=='.jsonl') if inp.suffix.lower() in ('.json','.jsonl') else pd.read_csv(inp)
    result=pd.DataFrame(index=df.index)
    for dst,names in ALIASES.items():
        src=pick(df,names)
        if src is None:
            result[dst]=np.nan
        else:
            result[dst]=df[src]
    if result['benchmark_row_id'].isna().any():
        raise ValueError('Every row requires a stable material identity')
    result['benchmark_row_id']=result['benchmark_row_id'].astype(str).str.strip()
    if result['benchmark_row_id'].duplicated().any():
        raise ValueError('Duplicate benchmark_row_id values found')
    if result['source_material_id'].isna().any(): result['source_material_id']=result['benchmark_row_id']
    if result['structure_path'].isna().any():
        raise ValueError('Every row requires structure_path/cif_path')
    root=Path(args.structure_root) if args.structure_root else None
    if root:
        result['structure_path']=[str((root/str(x)).resolve()) if not Path(str(x)).is_absolute() else str(x) for x in result['structure_path']]
    for c in ['formation_energy_per_atom','band_gap','k_total']:
        result[c]=pd.to_numeric(result[c], errors='coerce')
    if not result[['formation_energy_per_atom','band_gap','k_total']].notna().any(axis=None):
        raise ValueError('No valid Tier-1 target values found')
    result.insert(1,'source_database',args.source_name)
    result['dft_functional']=args.dft_functional
    result['functional_code']=args.functional_code
    out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(out,index=False)
    digest=sha256(out)
    meta={'manifest':str(out),'sha256':digest,'rows':len(result),'source':args.source_name,
          'target_counts':{c:int(result[c].notna().sum()) for c in ['formation_energy_per_atom','band_gap','k_total']}}
    out.with_suffix(out.suffix+'.metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
