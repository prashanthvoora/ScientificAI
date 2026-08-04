#!/usr/bin/env python3
"""Independent verifier for Tier-2 external linear dielectric metrics.

Recomputes metrics from tier2_external_prediction_audit.csv and optionally
compares them with tier2_external_metrics.csv and
 tier2_vs_tier1_paired_bootstrap.csv.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, mad=np.nan, mad_mae=np.nan,
                    median_ae=np.nan, p90_ae=np.nan, mean_signed_error=np.nan)
    e = p-y; ae=np.abs(e); mae=float(ae.mean())
    mad=float(np.mean(np.abs(y-y.mean())))
    return dict(n=len(y), mae=mae, rmse=float(np.sqrt(np.mean(e*e))), mad=mad,
                mad_mae=mad/mae if mae>0 else np.inf,
                median_ae=float(np.median(ae)), p90_ae=float(np.percentile(ae,90)),
                mean_signed_error=float(e.mean()))


def paired(y, cand, base):
    y=np.asarray(y,float); cand=np.asarray(cand,float); base=np.asarray(base,float)
    m=np.isfinite(y)&np.isfinite(cand)&np.isfinite(base)
    y,cand,base=y[m],cand[m],base[m]
    ac=np.abs(cand-y); ab=np.abs(base-y)
    mc=float(ac.mean()); mb=float(ab.mean()); mad=float(np.mean(np.abs(y-y.mean())))
    return dict(n=len(y), mae_candidate=mc, mae_baseline=mb,
                mae_improvement_baseline_minus_candidate=mb-mc,
                mad_mae_candidate=mad/mc, mad_mae_baseline=mad/mb,
                mad_mae_improvement_candidate_minus_baseline=mad/mc-mad/mb,
                candidate_row_wins=int(np.sum(ac<ab)), baseline_row_wins=int(np.sum(ab<ac)),
                ties=int(np.sum(np.isclose(ac,ab,rtol=0,atol=1e-12))))


def close(a,b,tol=5e-6):
    return (pd.isna(a) and pd.isna(b)) or abs(float(a)-float(b)) <= tol


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--tolerance', type=float, default=5e-6)
    args=ap.parse_args()
    out=Path(args.outdir)
    audit=pd.read_csv(out/'tier2_external_prediction_audit.csv')
    reported=pd.read_csv(out/'tier2_external_metrics.csv') if (out/'tier2_external_metrics.csv').exists() else None
    paired_rep=pd.read_csv(out/'tier2_vs_tier1_paired_bootstrap.csv') if (out/'tier2_vs_tier1_paired_bootstrap.csv').exists() else None

    required={'model','benchmark_row_id','task','y_true','y_pred','is_strict_external'}
    miss=required-set(audit.columns)
    if miss: raise SystemExit(f'Missing audit columns: {sorted(miss)}')
    logdf=audit[audit.task=='k_total_log'].copy()
    if logdf.empty: raise SystemExit('No k_total_log rows found')
    # invariant: same row/model should be unique
    if logdf.duplicated(['model','benchmark_row_id']).any():
        raise SystemExit('Duplicate model/row prediction pairs found')

    failures=[]
    rows=[]
    for model in sorted(logdf.model.unique()):
        for pop, sub in [('all_benchmark',logdf[logdf.model==model]),
                         ('strict_external',logdf[(logdf.model==model)&(logdf.is_strict_external.astype(bool))])]:
            y=np.exp(sub.y_true.to_numpy(float)); p=np.exp(sub.y_pred.to_numpy(float))
            r=metrics(y,p); rows.append({'model':model,'population':pop,**r})
            print(f"{model:6s} {pop:15s} n={r['n']:4d} MAE={r['mae']:.6f} RMSE={r['rmse']:.6f} MAD={r['mad']:.6f} MAD::MAE={r['mad_mae']:.6f}")
            if reported is not None:
                q=reported[(reported.model==model)&(reported.population==pop)&(reported.task=='k_total')&(reported.scale=='linear')]
                if len(q)!=1: failures.append(f'Missing/duplicate reported row {model}/{pop}/k_total')
                else:
                    q=q.iloc[0]
                    for k in ['n','mae','rmse','mad','mad_mae','median_ae','p90_ae','mean_signed_error']:
                        if k in q.index and not close(r[k],q[k],args.tolerance): failures.append(f'{model}/{pop} {k}: recomputed={r[k]} reported={q[k]}')

    if set(logdf.model.unique()) >= {'tier1','tier2'}:
        for pop, filt in [('all_benchmark', np.ones(len(logdf),dtype=bool)), ('strict_external', logdf.is_strict_external.astype(bool).to_numpy())]:
            s=logdf[filt].pivot(index='benchmark_row_id',columns='model',values=['y_true','y_pred'])
            # use tier2 true target; verify target equality
            y1=s[('y_true','tier1')].to_numpy(float); y2=s[('y_true','tier2')].to_numpy(float)
            if not np.allclose(y1,y2,rtol=0,atol=1e-12,equal_nan=True): failures.append(f'{pop}: tier1/tier2 targets differ')
            y=np.exp(y2); c=np.exp(s[('y_pred','tier2')].to_numpy(float)); b=np.exp(s[('y_pred','tier1')].to_numpy(float))
            r=paired(y,c,b)
            print(f"PAIRED {pop:15s} n={r['n']:4d} T2_MAE={r['mae_candidate']:.6f} T1_MAE={r['mae_baseline']:.6f} improvement={r['mae_improvement_baseline_minus_candidate']:.6f} wins={r['candidate_row_wins']}/{r['baseline_row_wins']}/{r['ties']}")
            if paired_rep is not None:
                q=paired_rep[(paired_rep.population==pop)&(paired_rep.task=='k_total')&(paired_rep.scale=='linear')]
                if len(q)!=1: failures.append(f'Missing/duplicate paired row {pop}/k_total')
                else:
                    q=q.iloc[0]
                    for k in ['n','mae_candidate','mae_baseline','mae_improvement_baseline_minus_candidate','mad_mae_candidate','mad_mae_baseline','mad_mae_improvement_candidate_minus_baseline','candidate_row_wins','baseline_row_wins','ties']:
                        if not close(r[k],q[k],args.tolerance): failures.append(f'paired {pop} {k}: recomputed={r[k]} reported={q[k]}')

    if failures:
        print('\nFAIL')
        for f in failures: print(' -',f)
        raise SystemExit(1)
    print('\nPASS: linear-space point metrics and paired point comparisons reproduce from row-level audit.')

if __name__=='__main__': main()
