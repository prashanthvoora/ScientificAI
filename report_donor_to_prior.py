#!/usr/bin/env python3
"""Report the missing seed-42 donor-to-prior comparison without training.

Run against the existing 30-row seed-42 export:
  python report_donor_to_prior.py --test_csv /path/to/test_predictions_seed42.csv
or /path/to/SEED42_RUN/reports/tier3_baselines/test_predictions.csv

Requires numpy and pandas. Writes one small donor_to_prior_seed42.md file.
Models and TRAIN-fitted donor means stay fixed during resampling.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def paired(target, donor, prior, group_ids, resamples=100000):
    difference=np.abs(donor-target)-np.abs(prior-target)
    _, groups=np.unique(group_ids,return_inverse=True)
    counts=np.bincount(groups);sums=np.bincount(groups,weights=difference)
    result={'delta':float(difference.mean()),
            'reduction':float(100*difference.mean()/np.abs(donor-target).mean()),
            'counts':tuple(int((difference==0).sum()) if s==0 else int((difference*s>0).sum()) for s in [1,-1,0])}
    for mode in ['row','group']:
        rng=np.random.default_rng(2026);samples=[]
        width=len(target) if mode=='row' else len(counts)
        for start in range(0,resamples,2000):
            indices=rng.integers(0,width,size=(min(2000,resamples-start),width))
            values=difference[indices].mean(1) if mode=='row' else sums[indices].sum(1)/counts[indices].sum(1)
            samples.append(values)
        samples=np.concatenate(samples)
        result[mode]=(*np.quantile(samples,[.025,.975]),float((samples>0).mean()))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test_csv',required=True)
    parser.add_argument('--output',default='donor_to_prior_seed42.md')
    args=parser.parse_args()
    f=pd.read_csv(args.test_csv,dtype={'record_id':str,'condition_group_id':str},keep_default_na=False)
    required=['record_id','condition_group_id','measured_ln_k','prior_ln_prediction','donor_mean_ln_prediction','full_ln_prediction']
    if set(required)-set(f):raise ValueError('Missing columns: '+', '.join(sorted(set(required)-set(f))))
    if len(f)!=30 or f.record_id.duplicated().any() or f.record_id.eq('').any():raise ValueError('Expected 30 unique TEST record identities')
    if f.condition_group_id.eq('').any() or f.condition_group_id.nunique()!=25:raise ValueError('Expected 25 nonempty process-condition groups')
    if 'split' in f and not f.split.eq('TEST').all():raise ValueError('Input must contain TEST only')
    if 'seed' in f and not pd.to_numeric(f.seed,errors='raise').eq(42).all():raise ValueError('Seed must be 42')
    f=f.sort_values('record_id').reset_index(drop=True)
    a=f[required[2:]].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(a).all():raise ValueError('All targets and predictions must be finite')
    y,prior,donor,full=a.T
    mae=[float(np.abs(z-y).mean()) for z in [donor,prior,full]]
    if not np.allclose(mae,[.196446,.1794,.130898],rtol=0,atol=.00006):
        raise ValueError(f'Export does not reproduce current seed-42 manuscript metrics: donor/prior/full={mae}. Use the corrected export; do not combine earlier runs.')
    r=paired(y,donor,prior,f.condition_group_id.to_numpy())
    counts='/'.join(map(str,r['counts']))
    row,group=r['row'],r['group']
    text=('# Seed-42 donor TRAIN-mean lookup to frozen structural prior\n\n'
          '| Comparison | Donor ln-MAE | Prior ln-MAE | Δln-MAE | Reduction | Row 95% CI | Group 95% CI | Improved/worse/tied | P positive row/group |\n'
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
          f'| Donor → prior | {mae[0]:.4f} | {mae[1]:.4f} | {r["delta"]:+.4f} | {r["reduction"]:.2f}% | [{row[0]:+.4f}, {row[1]:+.4f}] | [{group[0]:+.4f}, {group[1]:+.4f}] | {counts} | {row[2]:.3f}/{group[2]:.3f} |\n\n'
          'Δln-MAE = MAE(donor) − MAE(prior); positive values favor the prior. '
          'N = 30; process-condition groups = 25; 100,000 paired bootstrap resamples; bootstrap seed = 2026. '
          'Group resampling retains whole groups and estimates record-weighted ΔMAE. '
          'The inherited prior and TRAIN-fitted lookup are fixed during resampling; no TEST fitting or model training is performed. '
          'This comparison measures inherited-prior value relative to the lookup, not experimental-calibration or process-conditioning value.\n')
    destination=Path(args.output).resolve()
    if destination==Path(args.test_csv).resolve():raise ValueError('Output must not overwrite source evidence')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(text,encoding='utf-8')
    print(text);print('Saved:',destination)

if __name__=='__main__':main()
