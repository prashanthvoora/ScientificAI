#!/usr/bin/env python3
"""Print missing paper cells from existing corrected diagnostics; no training.
Run: python extract_tier3_table_cells.py corrected_seed42_diagnostics
Only Python's standard library is required. No individual records are printed.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def metrics(y, pred):
    n = len(y)
    mean = sum(y) / n
    ky, kp = [math.exp(v) for v in y], [math.exp(v) for v in pred]
    kmean = sum(ky) / n
    ln_mae = sum(abs(a-b) for a,b in zip(y,pred)) / n
    k_mae = sum(abs(a-b) for a,b in zip(ky,kp)) / n
    ln_mad = sum(abs(v-mean) for v in y) / n
    k_mad = sum(abs(v-kmean) for v in ky) / n
    return dict(ln_MAE=ln_mae, ln_RMSE=math.sqrt(sum((a-b)**2 for a,b in zip(y,pred))/n),
                ln_MAD=ln_mad, ln_MAD_to_MAE=ln_mad/ln_mae if ln_mae else None,
                k_MAE=k_mae, k_RMSE=math.sqrt(sum((a-b)**2 for a,b in zip(ky,kp))/n),
                k_MAD=k_mad, k_MAD_to_MAE=k_mad/k_mae if k_mae else None)


def fmt(value):
    return 'NOT_AVAILABLE' if value is None else f'{value:.9f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root/'corrected_diagnostics.json').read_text())
    with (root/'corrected_rows_with_groups.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or len({r['record_id'] for r in rows}) != 30:
        raise ValueError('Expected 30 unique TEST records')
    y = [float(r['measured_ln_k']) for r in rows]
    predictions = {name: [float(r[col]) for r in rows] for name,col in
                   [('prior','prior_ln_prediction'), ('process_off','process_off_ln_prediction'),
                    ('full','full_ln_prediction')]}
    if not all(math.isfinite(v) for vs in [y,*predictions.values()] for v in vs):
        raise ValueError('Nonfinite input')
    mean_p = sum(float(r['process_correction']) for r in rows)/len(rows)
    predictions['mean_process_offset'] = [v+mean_p for v in predictions['process_off']]
    for r in rows:
        actual = float(r['full_ln_prediction'])
        components = sum(float(r[c]) for c in ['prior_ln_prediction','structural_correction','process_correction']) + float(r.get('family_offset',0))
        if abs(actual-components) > 2e-6:
            raise ValueError('Additive decomposition failed')
    print(f'TABLE CELLS: seed={report["seed"]}; N=30; conditions={report["condition_groups"]["count"]}')
    full_metrics = metrics(y,predictions['full'])
    for name,pred in predictions.items():
        m = metrics(y,pred)
        for key,value in m.items():
            recorded = report['metrics'][name].get(key)
            if recorded is not None and abs(value-recorded)>2e-6:
                raise ValueError(f'CSV/JSON mismatch: {name}/{key}')
        print(name+': '+ '; '.join(f'{key}={fmt(m[key])}' for key in
              ['ln_MAE','ln_RMSE','ln_MAD_to_MAE','k_MAE','k_RMSE','k_MAD_to_MAE']))
    print(f'TARGET MAD: ln={fmt(full_metrics["ln_MAD"])}; k={fmt(full_metrics["k_MAD"])}')
    for scale,key in [('ln','paired'),('k','paired_linear_k')]:
        target = y if scale=='ln' else [math.exp(v) for v in y]
        full = predictions['full'] if scale=='ln' else [math.exp(v) for v in predictions['full']]
        for name in ['prior','process_off','mean_process_offset']:
            pred = predictions[name] if scale=='ln' else [math.exp(v) for v in predictions[name]]
            diffs = [abs(a-b)-abs(c-b) for a,b,c in zip(pred,target,full)]
            improved = sum(v>0 for v in diffs)
            worse = sum(v<0 for v in diffs)
            tied = len(diffs)-improved-worse
            delta = sum(diffs)/len(diffs)
            relative = 100*delta/(sum(abs(a-b) for a,b in zip(pred,target))/len(diffs))
            parts=[]
            for unit in ['row','condition_group']:
                b=report[key][name+'_to_full'][unit]
                if abs(delta-b['delta_MAE'])>2e-6:
                    raise ValueError('Paired point estimate mismatch')
                lo,hi=b['CI95']
                parts.append(f'{unit}: CI95=[{fmt(lo)},{fmt(hi)}]; P_positive={fmt(b.get("P_delta_positive"))}')
            print(f'{scale} {name}->full: delta={fmt(delta)}; reduction_pct={relative:.6f}; improved/worse/tied={improved}/{worse}/{tied}; '+ '; '.join(parts))
    print('PASS: CSV metrics agree with JSON; no row-level information displayed.')
    print('Prior reconciliation remains separate: '+str(report.get('reference_prior_check',{}).get('status','NOT_AVAILABLE')))


if __name__=='__main__':
    main()
