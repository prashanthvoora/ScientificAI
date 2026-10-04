#!/usr/bin/env python3
"""Recompute corrected Tier-3 TEST diagnostics without model loading/training.
Needs numpy, pandas, matplotlib. Use only your trusted saved pickle frame.
"""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def metrics(y, pred):
    err=pred-y; k=np.exp(y); ke=np.exp(pred)-k
    mad=float(np.abs(k-k.mean()).mean()); mae=float(np.abs(ke).mean())
    ln_mae=float(np.abs(err).mean());ln_mad=float(np.abs(y-y.mean()).mean())
    return dict(ln_MAE=ln_mae,ln_MAD=ln_mad,ln_MAD_to_MAE=ln_mad/ln_mae if ln_mae else None,ln_RMSE=float(np.sqrt((err**2).mean())),
                k_MAE=mae,k_RMSE=float(np.sqrt((ke**2).mean())),k_MAD=mad,k_MAD_to_MAE=mad/mae if mae else None)

def corr(x,y):
    if np.std(x)==0 or np.std(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])

def bootstrap(difference, groups, repetitions, seed):
    # Cluster bootstrap: sample G whole groups with replacement; retain every
    # member; estimate record-weighted MAE difference, with variable sample N.
    rng=np.random.default_rng(seed); groups=np.asarray(groups)
    unique=np.unique(groups); sums=np.array([difference[groups==g].sum() for g in unique])
    sizes=np.array([(groups==g).sum() for g in unique]); draws=[]
    for begin in range(0,repetitions,2000):
        ids=rng.integers(0,len(unique),(min(2000,repetitions-begin),len(unique)))
        draws.append(sums[ids].sum(1)/sizes[ids].sum(1))
    values=np.concatenate(draws)
    return dict(delta_MAE=float(difference.mean()),CI95=np.quantile(values,[.025,.975]).tolist(),
                P_delta_positive=float((values>0).mean()),groups=len(unique),resamples=repetitions)

def clean(v):
    if v is None or pd.isna(v):return None
    s=str(v).strip()
    return None if s.lower() in {'','nan','none','null','na','n/a'} else s

def token(v):return re.sub(r'[\s_\-]+','',v).lower()

def conditions(df, frame_path, meta_path):
    frame=pd.read_pickle(frame_path); meta=json.loads(Path(meta_path).read_text())
    if meta.get('schema')!='v4.60.4' or meta.get('process_input_dim')!=38:
        raise ValueError('Expected saved corrected v4.60.4 preprocessing with 38 inputs')
    if not frame.index.is_unique:raise ValueError('Saved frame index is not unique')
    keys=[]; available=[]; means=np.asarray(meta['num_mean'],np.float32);std=np.asarray(meta['num_std'],np.float32)
    if not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid saved numerical scales')
    for _,item in df.iterrows():
        row=frame.loc[int(item.dataset_row_idx)]
        donor=clean(row.get('imputed_from',row.get('jid',row.get('mp_id'))))
        if donor!=clean(item.get('donor_id')):raise ValueError('Frame/export donor mismatch')
        nums=[]; masks=[]
        for field in meta['numeric_fields']:
            v=pd.to_numeric(row.get(field),errors='coerce');present=v is not None and bool(np.isfinite(v))
            if present and field in meta['log_fields']:v=np.log1p(max(float(v),0.0))
            nums.append(float(v) if present else 0.0);masks.append(float(present))
        cats=[];cat_present=False
        for field,vocab in meta['categorical_vocabulary'].items():
            v=clean(row.get(field));cat_present |= v is not None
            if v is None:idx=len(vocab)
            else:
                canonical=meta['alias_rules'].get(field,{}).get(token(v))
                if canonical is None:canonical=next((c for c in vocab if token(c)==token(v)),v)
                idx=vocab.index(canonical) if canonical in vocab else len(vocab)+1
            cats.append(idx)
        avail=float(any(masks) or cat_present);mask=np.asarray(masks,np.float32)
        normalized=(np.asarray(nums,np.float32)-means)/std*mask*avail
        # Exact float32 standardized descriptors + masks + categorical indices.
        # Fixed embeddings depend only on these indices; availability is explicit.
        key=(normalized.tobytes()+mask.tobytes()+np.asarray(cats,np.int64).tobytes()+np.float32(avail).tobytes())
        keys.append(hashlib.sha256(key).hexdigest());available.append(avail)
    return keys,np.asarray(available),meta

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path)
    ap.add_argument('--frame',required=True,type=Path)
    ap.add_argument('--metadata',required=True,type=Path)
    ap.add_argument('--out',type=Path,default=Path('corrected_diagnostics'))
    ap.add_argument('--reference-csv',type=Path,help='Optional v63 row-level export for prior reconciliation')
    ap.add_argument('--resamples',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.resamples<1:ap.error('--resamples must be positive')
    df=pd.read_csv(args.predictions,dtype={'record_id':str,'donor_id':str}).sort_values('record_id').reset_index(drop=True)
    cols=['measured_ln_k','prior_ln_prediction','structural_correction','process_correction','family_offset','process_off_ln_prediction','full_ln_prediction']
    if len(df)!=30 or df.record_id.duplicated().any():raise ValueError('Expected exactly 30 unique TEST records')
    if not np.isfinite(df[cols].to_numpy(float)).all():raise ValueError('Nonfinite prediction/target')
    y,z,s,p,f,off,full=(df[c].to_numpy(float) for c in cols)
    if np.max(np.abs(full-(z+s+p+f)))>2e-6 or np.max(np.abs(off-(z+s+f)))>2e-6:
        raise ValueError('Component decomposition failed')
    if np.max(np.abs(f))>1e-10:raise ValueError('Expected disabled family offsets; resolve nonzero offsets before using these diagnostics')
    groups,available,meta=conditions(df,args.frame,args.metadata)
    if np.any((available==0)&(np.abs(p)>2e-6)):raise ValueError('Inactive process rows have nonzero process corrections')
    if np.max(np.abs(p))>.15+2e-6:raise ValueError('Process correction exceeds recorded 0.15 bound')
    required=y-off; scalar=float(np.median(required));mean=float(p.mean())
    predictions={'prior':z,'process_off':off,'full':full,'mean_process_offset':off+mean,'TEST_optimal_ln_MAE_scalar':off+scalar}
    result=dict(N=30,seed=meta['training_seed'],provenance={},metrics={k:metrics(y,v) for k,v in predictions.items()},
                scalar_controls={'mean_p':mean,'TEST_optimal_ln_MAE_scalar':scalar,'fit_population':'TEST; diagnostic only, not an independently fitted baseline'},
                alignment={'Pearson':corr(p,required),'Spearman':corr(pd.Series(p).rank().to_numpy(),pd.Series(required).rank().to_numpy())},
                bound={'bound':.15,'mean_occupancy':float(np.mean(np.abs(p)/.15)),'max_occupancy':float(np.max(np.abs(p)/.15)),
                       'p_min':float(p.min()),'p_max':float(p.max()),'p_span':float(np.ptp(p)),'p_sample_sd':float(p.std(ddof=1))},
                condition_groups={'count':len(set(groups)),'sizes':sorted(pd.Series(groups).value_counts().tolist()),
                                  'warning':'Only one condition group; bootstrap cannot estimate between-group uncertainty' if len(set(groups))==1 else None,
                                  'definition':'Exact corrected float32 numerical descriptors, numerical masks, categorical indices and availability; record-weighted whole-group bootstrap; not publication groups'},
                paired={},paired_linear_k={},input_output_mapping={'unique_process_conditions':len(set(groups)),
                'unique_process_corrections_float32':int(len(np.unique(p.astype(np.float32)))),
                'max_within_condition_p_range':float(max(np.ptp(p[np.asarray(groups)==g]) for g in set(groups)))},reference_prior_check={'status':'UNRESOLVED: no v63 row-level reference supplied'})
    for name,path in [('predictions',args.predictions),('frame',args.frame),('metadata',args.metadata)]:
        result['provenance'][name]={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    errors=np.abs(full-y)
    for label,pred in predictions.items():
        if label=='full':continue
        difference=np.abs(pred-y)-errors
        result['paired'][label+'_to_full']={'row':bootstrap(difference,np.arange(30),args.resamples,args.seed),
            'condition_group':bootstrap(difference,groups,args.resamples,args.seed)}
        linear_difference=np.abs(np.exp(pred)-np.exp(y))-np.abs(np.exp(full)-np.exp(y))
        result['paired_linear_k'][label+'_to_full']={'row':bootstrap(linear_difference,np.arange(30),args.resamples,args.seed),
            'condition_group':bootstrap(linear_difference,groups,args.resamples,args.seed)}
    rng=np.random.default_rng(args.seed);null=[]
    for begin in range(0,args.resamples,2000):
        count=min(2000,args.resamples-begin)
        shuffled=np.stack([rng.permutation(p) for _ in range(count)])
        null.append(np.abs(off[None,:]+shuffled-y[None,:]).mean(1))
    null=np.concatenate(null);observed=float(errors.mean())
    result['permutation']={'resamples':args.resamples,'seed':args.seed,'observed_ln_MAE':observed,
        'null_mean_ln_MAE':float(null.mean()),'null_sample_sd':float(null.std(ddof=1)),
        'fraction_null_lower_or_equal':float((null<=observed).mean()),
        'one_sided_empirical_p':float((1+(null<=observed).sum())/(args.resamples+1)),
        'scope':'Unrestricted row shuffle of p; exploratory assignment diagnostic, not causal or independently trained evidence'}
    if args.reference_csv:
        ref=pd.read_csv(args.reference_csv,dtype=str)
        idcol=next((c for c in ['record_id','split_identity'] if c in ref),None)
        zcol=next((c for c in ['prior_ln_prediction','k_dft_log'] if c in ref),None)
        ycol=next((c for c in ['measured_ln_k','k_true_log'] if c in ref),None)
        if not all([idcol,zcol,ycol]):raise ValueError('Reference needs record_id/split_identity, prior_ln_prediction/k_dft_log, measured_ln_k/k_true_log')
        if 'split' in ref:ref=ref[ref['split'].str.lower().isin(['test','test-best','test-evaluate'])]
        if len(ref)!=30 or ref[idcol].duplicated().any():raise ValueError('Reference must contain 30 unique TEST records')
        ref=ref.set_index(idcol)
        if set(ref.index)!=set(df.record_id):raise ValueError('Reference TEST identities differ')
        ref=ref.loc[df.record_id]; rz=pd.to_numeric(ref[zcol]).to_numpy();ry=pd.to_numeric(ref[ycol]).to_numpy()
        if not np.isfinite(rz).all() or not np.isfinite(ry).all():raise ValueError('Nonfinite reference')
        target_diff=float(np.max(np.abs(ry-y)));prior_diff=float(np.max(np.abs(rz-z)))
        result['reference_prior_check']={'same_targets':target_diff<=2e-6,'max_target_change':target_diff,'max_prior_prediction_change':prior_diff,
            'reference_ln_MAE':float(np.abs(rz-ry).mean()),'corrected_ln_MAE':float(np.abs(z-y).mean()),
            'status':'PREDICTION_PARITY_PASS' if max(target_diff,prior_diff)<=2e-6 else 'UNRESOLVED: cross-version mismatch',
            'limitation':'Prediction parity does not independently verify checkpoint tensors or graph inputs'}
        result['provenance']['reference_csv']={'filename':args.reference_csv.name,'sha256':hashlib.sha256(args.reference_csv.read_bytes()).hexdigest()}
    args.out.mkdir(parents=True,exist_ok=True)
    df['condition_group']=groups;df['process_available']=available;df.to_csv(args.out/'corrected_rows_with_groups.csv',index=False)
    pd.DataFrame(result['metrics']).T.to_csv(args.out/'corrected_component_metrics.csv',index_label='configuration')
    (args.out/'corrected_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['CORRECTED TIER3 DIAGNOSTICS',f'N = 30; seed = {meta["training_seed"]}',f'prior_reconciliation = {result["reference_prior_check"]["status"]}',f'condition_groups = {len(set(groups))}',f'prior_checkpoint_sha256 = {meta.get("prior_sha256","not recorded")}']
    for name,m in result['metrics'].items():lines.append(f'{name}: ln_MAE={m["ln_MAE"]:.9f}; ln_RMSE={m["ln_RMSE"]:.9f}; k_MAE={m["k_MAE"]:.9f}; k_MAD_to_MAE={m["k_MAD_to_MAE"]:.9f}')
    for name in ['scalar_controls','alignment','bound','input_output_mapping','permutation']:lines.append(name+' = '+json.dumps(result[name]))
    for scale,key in [('ln','paired'),('linear_k','paired_linear_k')]:
        for name,v in result[key].items():
            for unit,value in v.items():lines.append(scale+' '+name+' '+unit+' = '+json.dumps(value))
    lines.append('TEST-fitted scalar/offset and permutation are diagnostics; independent TRAIN/VAL-fitted baselines remain issue 2.')
    (args.out/'paper_update_short.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=np.argsort(y,kind='stable');x=np.arange(1,31)
    fig,axes=plt.subplots(2,1,figsize=(7,7),sharex=True)
    for label,v in [('Measured',y),('Frozen prior',z),('Structure + adapter',off),('Full model',full)]:axes[0].plot(x,v[order],marker='.',label=label)
    axes[0].set_ylabel('ln κ');axes[0].legend(ncol=2,fontsize=9);axes[0].set_title('(a) Corrected predictions on held-out TEST')
    axes[1].bar(x-.2,s[order],width=.4,label='Structural');axes[1].bar(x+.2,p[order],width=.4,label='Process')
    axes[1].axhline(.15,color='gray',ls=':',label='Process bound ±0.15');axes[1].axhline(-.15,color='gray',ls=':')
    axes[1].set_ylabel('Correction in ln κ');axes[1].set_xlabel('TEST record ordered by measured ln κ');axes[1].legend(fontsize=9)
    axes[1].set_title('(b) Corrected additive contributions');fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(args.out/f'Fig6_corrected.{suffix}',dpi=300)
    plt.close(fig)
    print('\n'.join(lines));print('Outputs:',args.out.resolve())
if __name__=='__main__':main()
