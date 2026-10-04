#!/usr/bin/env python3
"""Complete Tier-3 reporting from saved exports; no model training.

python complete_tier3_baseline_reporting.py --run_roots SEED42 SEED124 SEED777 \
    --output_dir tier3_complete

Each root contains reports/tier3_baselines/{train,val,test}_predictions.csv
and baseline_summary.json. Requires numpy, pandas, matplotlib.
Optional --hf_membership CSV: record_id,is_hf_containing (0/1) for ALL 500
experimental TRAIN identities, verified from original film material metadata.
Never infer film family from donor identity or from TEST errors.
Without this manifest, the Hf-TRAIN variant is explicitly skipped.
VAL calibration is a development-set-fitted sensitivity baseline: the adapter
was already selected on VAL. It is not an independent calibration holdout.
All requested variants are reported; never select a variant by TEST score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = {
    'Independent adapter-only': 'independent_adapter_ln_prediction',
    'Adapter + global TRAIN scalar': 'adapter_scalar_ln_prediction',
    'Adapter + VAL scalar': 'adapter_val_scalar_ln_prediction',
    'Donor TRAIN-mean lookup': 'donor_mean_ln_prediction',
    'Full': 'full_ln_prediction',
}

def metrics(y, p):
    return dict(N=len(y), ln_MAE=float(np.abs(p-y).mean()),
                ln_RMSE=float(np.sqrt(np.square(p-y).mean())),
                linear_MAE=float(np.abs(np.exp(p)-np.exp(y)).mean()),
                linear_RMSE=float(np.sqrt(np.square(np.exp(p)-np.exp(y)).mean())))

def read_frame(folder, split, count):
    f = pd.read_csv(folder / (split.lower()+'_predictions.csv'),
                    dtype={'record_id':str,'donor_key':str,'condition_group_id':str},
                    keep_default_na=False).sort_values('record_id').reset_index(drop=True)
    required = ['record_id','split','is_experimental','donor_key','condition_group_id',
                'measured_ln_k','independent_adapter_ln_prediction']
    if not set(required).issubset(f.columns):
        raise ValueError(f'Missing required columns: {folder} {split}')
    if len(f)!=count or f.record_id.duplicated().any() or not f.split.eq(split).all():
        raise ValueError(f'Expected {count} unique {split} rows: {folder}')
    for c in ['measured_ln_k','independent_adapter_ln_prediction']:
        f[c]=pd.to_numeric(f[c],errors='raise')
        if not np.isfinite(f[c]).all(): raise ValueError('Nonfinite '+c)
    return f

def controls(train, val, hf=None):
    exp = train.loc[train.is_experimental.eq(1)].copy()
    if len(exp)!=500 or not val.is_experimental.eq(1).all():
        raise ValueError('Expected 500 experimental TRAIN and exclusively experimental VAL')
    offsets = {'global TRAIN':(float(np.median(exp.measured_ln_k-exp.independent_adapter_ln_prediction)),len(exp),'experimental TRAIN'),
               'VAL':(float(np.median(val.measured_ln_k-val.independent_adapter_ln_prediction)),len(val),'VAL')}
    if hf is not None:
        if set(hf.record_id)!=set(exp.record_id):
            raise ValueError('Hf membership manifest must match all experimental TRAIN identities exactly')
        selected=exp.loc[exp.record_id.isin(hf.loc[hf.is_hf_containing.eq(1),'record_id'])]
        if len(selected)==0: raise ValueError('No Hf-containing experimental TRAIN records; variant unavailable')
        offsets['Hf TRAIN']=(float(np.median(selected.measured_ln_k-selected.independent_adapter_ln_prediction)),len(selected),'Hf-containing experimental TRAIN')
    donor=exp.loc[exp.donor_key.ne('')].groupby('donor_key').agg(
        mean_ln_k=('measured_ln_k','mean'),fit_count=('record_id','size'))
    return offsets,donor,float(exp.measured_ln_k.mean())

def markdown(f):
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in f.itertuples(index=False,name=None))

def run(roots, out, manifest=None):
    out=Path(out).resolve()
    folders=[Path(r) if (Path(r)/'baseline_summary.json').is_file() else Path(r)/'reports/tier3_baselines' for r in roots]
    if any(out==f.resolve() or f.resolve() in out.parents for f in folders):
        raise ValueError('Output must be separate from source evidence')
    hf=None
    if manifest:
        hf=pd.read_csv(manifest,dtype={'record_id':str})
        if hf.record_id.duplicated().any() or not hf.is_hf_containing.isin([0,1]).all():
            raise ValueError('Manifest needs unique record_id and binary is_hf_containing')
    all_metrics=[];all_effects=[];audit=[];frames=[];seeds=[];reference=None
    method_map=dict(METHODS)
    if hf is not None:method_map['Adapter + Hf TRAIN scalar']='adapter_hf_train_scalar_ln_prediction'
    for folder in folders:
        summary=json.loads((folder/'baseline_summary.json').read_text())
        if summary.get('selection_partition')!='VAL' or summary.get('scalar_and_donor_fit_partition')!='experimental TRAIN':
            raise ValueError('Unexpected source fitting protocol')
        seed=int(summary['seed']);seeds.append(seed)
        train=read_frame(folder,'TRAIN',582);val=read_frame(folder,'VAL',20);test=read_frame(folder,'TEST',30)
        if not test.is_experimental.eq(1).all() or test.condition_group_id.nunique()!=25:
            raise ValueError('Expected 30 experimental TEST rows / 25 conditions')
        ids=[set(f.record_id) for f in [train,val,test]]
        if any(ids[i]&ids[j] for i,j in [(0,1),(0,2),(1,2)]):raise ValueError('Partitions overlap')
        fixed=['record_id','donor_key','atoms_sha256','condition_group_id']
        if reference is not None:
            for old,new in zip(reference,[train,val,test]):
                if not old[fixed].equals(new[fixed]):raise ValueError('Identities/donors/conditions differ across seeds')
                if not np.allclose(old.measured_ln_k,new.measured_ln_k,rtol=0,atol=2e-6):raise ValueError('Targets differ across seeds')
        else:reference=[f.copy() for f in [train,val,test]]
        offsets,donor,fallback=controls(train,val,hf)
        if not np.isclose(offsets['global TRAIN'][0],summary['scalar_ln_offset'],atol=2e-6,rtol=0):
            raise ValueError('Global TRAIN scalar does not reproduce source summary')
        original=test.adapter_scalar_ln_prediction.to_numpy(float)
        lookup=test.donor_key.map(donor.mean_ln_k)
        test['donor_lookup_fallback']=lookup.isna().astype(int)
        donor_pred=lookup.fillna(fallback)
        if 'donor_mean_ln_prediction' in test and not np.allclose(test.donor_mean_ln_prediction.astype(float),donor_pred,atol=2e-6,rtol=0):
            raise ValueError('Donor lookup does not reproduce original export')
        test['donor_mean_ln_prediction']=donor_pred
        for name,(c,n,partition) in offsets.items():
            col={'global TRAIN':'adapter_scalar_ln_prediction','VAL':'adapter_val_scalar_ln_prediction','Hf TRAIN':'adapter_hf_train_scalar_ln_prediction'}[name]
            test[col]=test.independent_adapter_ln_prediction+c
            audit.append(dict(seed=seed,variant=name,c_ln=c,sign='positive' if c>0 else 'negative' if c<0 else 'zero',fit_N=n,fit_partition=partition,
                              donor_fallback_TEST_N=int(test.donor_lookup_fallback.sum()),donor_fallback_ln=fallback))
        if not np.allclose(original,test.adapter_scalar_ln_prediction,atol=2e-6,rtol=0):raise ValueError('Source scalar prediction mismatch')
        y=test.measured_ln_k.to_numpy(float);full=test.full_ln_prediction.to_numpy(float)
        if not np.isfinite(full).all():raise ValueError('Nonfinite full prediction')
        for label,col in method_map.items():
            prediction=test[col].to_numpy(float)
            all_metrics.append(dict(seed=seed,configuration=label,**metrics(y,prediction)))
            if label!='Full':
                for space in ['ln-k','k']:
                    target,base,pred=(y,prediction,full) if space=='ln-k' else (np.exp(y),np.exp(prediction),np.exp(full))
                    all_effects.append(dict(seed=seed,comparison=label+' -> full',space=space,
                        **paired(target,base,pred,test.condition_group_id.to_numpy())))
        frames.append((seed,test,donor))
    if sorted(seeds)!=[42,124,777]:raise ValueError('Exactly seeds 42,124,777 required')
    out.mkdir(parents=True,exist_ok=True)
    m=pd.DataFrame(all_metrics);e=pd.DataFrame(all_effects);a=pd.DataFrame(audit)
    m.to_csv(out/'table_XII_all_methods_by_seed.csv',index=False)
    e.to_csv(out/'table_XIII_all_effects_by_seed.csv',index=False)
    a.to_csv(out/'scalar_offsets_and_fit_counts.csv',index=False)
    for seed,test,donor in frames:
        test.to_csv(out/f'test_predictions_seed{seed}.csv',index=False)
        donor.to_csv(out/f'donor_lookup_seed{seed}.csv')
    averages=m.groupby('configuration',sort=False)[['ln_MAE','ln_RMSE','linear_MAE','linear_RMSE']].agg(['mean','std'])
    averages.to_csv(out/'table_XII_mean_SD.csv')
    display=pd.DataFrame({'Configuration':averages.index})
    for metric in ['ln_MAE','ln_RMSE','linear_MAE','linear_RMSE']:
        display[metric]=[f'{mean:.4f} ± {sd:.4f}' for mean,sd in averages[metric].to_numpy()]
    primary=e.loc[e.seed.eq(42)&e.space.eq('ln-k')]
    edisplay=pd.DataFrame({'Comparison':primary.comparison,'Δln-MAE':primary.delta_MAE.map(lambda x:f'{x:+.4f}'),
        'Reduction':primary.relative_reduction_pct.map(lambda x:f'{x:.2f}%'),
        'Row CI':[f'[{l:+.4f}, {h:+.4f}]' for l,h in zip(primary.row_CI95_low,primary.row_CI95_high)],
        'Group CI':[f'[{l:+.4f}, {h:+.4f}]' for l,h in zip(primary.group_CI95_low,primary.group_CI95_high)],
        'Improved/worse/tied':[f'{w}/{l}/{t}' for w,l,t in zip(primary.rows_improved,primary.rows_worse,primary.rows_tied)],
        'P positive row/group':[f'{r:.3f}/{g:.3f}' for r,g in zip(primary.row_P_delta_positive,primary.group_P_delta_positive)]})
    adisplay=a[['seed','variant','c_ln','sign','fit_N','fit_partition']].copy();adisplay.c_ln=adisplay.c_ln.map(lambda x:f'{x:+.4f}')
    wins=e.loc[e.space.eq('ln-k')].groupby('comparison',sort=False).delta_MAE.apply(lambda x:int((x>0).sum())).reset_index(name='Seeds favoring full out of 3')
    text='# Completed baseline reporting\n\n'+markdown(display)+'\n\n'+markdown(edisplay)+'\n\nOffsets (ln κ):\n\n'+markdown(adisplay)+'\n\n'+markdown(wins)
    text+='\n\nMean ± sample SD describes initialization spread on the same 30 TEST records. All models and fits are fixed during paired resampling; VAL is reused for adapter selection and calibration. Global TRAIN scalar is a cross-family control. Family mismatch is a hypothesis, not established causation. Donor lookup uses global experimental TRAIN mean for unseen/missing donors; report fallback count and coverage. Donor lookup addresses one predictive alternative, not all provenance confounding.\n'
    if hf is None:text+='\nHf-TRAIN variant NOT computed: verified TRAIN-family membership required.\n'
    (out/'paper_extract_complete.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
    fig,ax=plt.subplots(figsize=(5.8,3.5))
    for label,col in list(METHODS.items()):
        if label not in ['Independent adapter-only','Adapter + global TRAIN scalar','Full']:continue
        sub=m.loc[m.configuration.eq(label)].sort_values('seed')
        ax.plot(range(3),sub.ln_MAE,'o-',label=label,linewidth=1)
    ax.set_xticks(range(3),['42','124','777']);ax.set_xlabel('Initialization seed');ax.set_ylabel('TEST ln-MAE')
    ax.legend(frameon=False,fontsize=7);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    fig.savefig(out/'all_three_configurations_by_seed.png',dpi=450);plt.close(fig)
    print(f'Completed without training: {out}/paper_extract_complete.md')
    print('Hf TRAIN variant:', 'computed' if hf is not None else 'not computed; membership needed')
    return m,e,a

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_roots',nargs=3,required=True);p.add_argument('--output_dir',required=True)
    p.add_argument('--hf_membership',help='Verified record_id,is_hf_containing CSV for all 500 experimental TRAIN rows')
    args=p.parse_args();run(args.run_roots,args.output_dir,args.hf_membership)



def paired(target, baseline, full, group_ids, resamples=100000):
    """Conditional paired uncertainty, not uncertainty from retraining models."""
    target, baseline, full = [np.asarray(x, dtype=float) for x in (target, baseline, full)]
    if (target.ndim != 1 or not len(target) or target.shape != baseline.shape or target.shape != full.shape
            or not all(np.isfinite(x).all() for x in (target, baseline, full))):
        raise RuntimeError("Paired inputs must be finite, nonempty, aligned arrays")
    if len(group_ids) != len(target) or resamples < 1:
        raise RuntimeError("Invalid group labels/resample count")
    difference = np.abs(baseline - target) - np.abs(full - target)
    _, groups = np.unique(np.asarray(group_ids, dtype=str), return_inverse=True)
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=difference)
    output = {"delta_MAE": float(difference.mean()),
        "relative_reduction_pct": (float(100 * difference.mean() / np.abs(baseline-target).mean())
                                   if np.abs(baseline-target).mean() else None),
        "rows_improved": int((difference > 0).sum()),
        "rows_worse": int((difference < 0).sum()), "rows_tied": int((difference == 0).sum()),
        "N": len(target), "condition_groups": len(counts), "resamples": resamples,
        "bootstrap_seed": 2026, "group_estimand": "record-weighted; whole groups retained"}
    for mode in ("row", "group"):
        rng = np.random.default_rng(2026)
        values = []
        width = len(target) if mode == "row" else len(counts)
        for start in range(0, resamples, 2000):
            indices = rng.integers(0, width, size=(min(2000, resamples-start), width))
            sample = (difference[indices].mean(axis=1) if mode == "row" else
                      sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
            values.append(sample)
        values = np.concatenate(values)
        low, high = np.quantile(values, [.025, .975])
        output.update({mode+"_CI95_low": float(low), mode+"_CI95_high": float(high),
                       mode+"_P_delta_positive": float((values > 0).mean())})
    return output

if __name__ == "__main__":
    main()
