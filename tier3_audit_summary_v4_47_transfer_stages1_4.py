#!/usr/bin/env python3
"""Compare two Tier-3 audit runs produced from different Tier-2 checkpoints.

Rows are aligned by dataset_row_id. Outputs are written under --output-dir.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

IDS=("dataset_row_id","dataset_row_idx","row_id","split_identity")
TRUE_K=("k_true","true_k","target_k","k_target","k_measured")
TRUE_LOG=("k_true_log","true_log_k","target_log_k","k_target_log")
DFT_K=("k_dft","k_DFT","k_dft_linear","base_prediction","base_pred","base_k_pred")
DFT_LOG=("k_dft_log","k_DFT_log","base_prediction_log","base_pred_log","base_log_k")
PRED_K=("k_pred","pred_k","final_prediction","final_pred","prediction")
PRED_LOG=("k_pred_log","pred_log_k","final_prediction_log","final_pred_log")
DELTA_LOG=("delta_log_pred","bounded_process_delta","applied_process_delta","process_delta")
META=("material","formula","phase","donor","precursor","oxidant","substrate","deposition_method","paper_id","doi","source","substrate_temp_C","anneal_temp_C","film_thickness_nm","is_experimental","imputed_structure","proc_avail")

def pick(cols,cands):
    m={str(c).lower():str(c) for c in cols}
    for c in cands:
        if c.lower() in m:return m[c.lower()]
    return None

def locate_normal(p:Path)->Path:
    p=p.expanduser().resolve()
    for q in (p,p/'normal',p/'tier3_prediction_audit'/'normal',p/'reports'/'tier3_prediction_audit'/'normal'):
        if q.is_dir() and list(q.glob('tier3_*prediction_audit.csv')): return q
    raise FileNotFoundError(f'normal audit folder not found below {p}')

def locate_summary(p:Path)->Path|None:
    p=p.expanduser().resolve()
    for q in (p/'summary',p.parent/'summary',p/'tier3_prediction_audit'/'summary',p/'reports'/'tier3_prediction_audit'/'summary'):
        if q.is_dir(): return q
    return None

def find_pred(d:Path,split:str)->Path:
    for n in (f'tier3_best_checkpoint_{split}_prediction_audit.csv',f'tier3_best_{split}_prediction_audit.csv'):
        q=d/n
        if q.exists():return q
    m=sorted(d.glob(f'tier3_*best*_{split}_prediction_audit.csv'))
    if not m:raise FileNotFoundError(f'best {split} audit missing in {d}')
    return m[0]

def find_top20(normal:Path,summary:Path|None)->Path|None:
    for root in [x for x in (normal,summary) if x]:
        for n in ('tier3_best_checkpoint_test_top20_errors.csv','tier3_best_test_top20_errors.csv'):
            q=root/n
            if q.exists():return q
        m=sorted(root.glob('tier3_*best*_test_top20_errors.csv'))
        if m:return m[0]
    return None

def norm(df:pd.DataFrame,label:str):
    cols=df.columns; idc=pick(cols,IDS)
    if not idc: raise ValueError(f'{label}: no dataset row ID column')
    mp={'true_k':pick(cols,TRUE_K),'true_log':pick(cols,TRUE_LOG),'dft_k':pick(cols,DFT_K),'dft_log':pick(cols,DFT_LOG),'pred_k':pick(cols,PRED_K),'pred_log':pick(cols,PRED_LOG),'delta_log':pick(cols,DELTA_LOG)}
    out=pd.DataFrame({'dataset_row_id':df[idc].astype(str)})
    for k,v in mp.items():
        if v: out[k]=pd.to_numeric(df[v],errors='coerce')
    if 'true_k' not in out and 'true_log' in out: out['true_k']=np.exp(out.true_log)
    if 'true_log' not in out and 'true_k' in out: out['true_log']=np.log(out.true_k.where(out.true_k>0))
    if 'dft_k' not in out and 'dft_log' in out: out['dft_k']=np.exp(out.dft_log)
    if 'dft_log' not in out and 'dft_k' in out: out['dft_log']=np.log(out.dft_k.where(out.dft_k>0))
    if 'pred_k' not in out and 'pred_log' in out: out['pred_k']=np.exp(out.pred_log)
    if 'pred_log' not in out and 'pred_k' in out: out['pred_log']=np.log(out.pred_k.where(out.pred_k>0))
    if 'delta_log' not in out and {'pred_log','dft_log'}<=set(out): out['delta_log']=out.pred_log-out.dft_log
    for c in META:
        a=pick(cols,(c,))
        if a: out[c]=df[a]
    if out.dataset_row_id.duplicated().any(): raise ValueError(f'{label}: duplicate dataset_row_id')
    return out,{'id':idc,**mp}

def mad(s):
    s=pd.to_numeric(s,errors='coerce').dropna()
    return float((s-s.mean()).abs().mean()) if len(s) else np.nan

def compare(pa:Path,pb:Path,la:str,lb:str,split:str,bound:float):
    a,ma=norm(pd.read_csv(pa),la); b,mb=norm(pd.read_csv(pb),lb)
    same=set(a.dataset_row_id)==set(b.dataset_row_id)
    m=a.merge(b,on='dataset_row_id',how='outer',suffixes=(f'_{la}',f'_{lb}'),indicator=True,validate='one_to_one')
    m.insert(1,'split',split)
    for base in ('true_k','true_log','dft_k','dft_log','pred_k','pred_log','delta_log'):
        ca,cb=f'{base}_{la}',f'{base}_{lb}'
        if ca not in m and base in m: m[ca]=m[base]
        if cb not in m and base in m: m[cb]=m[base]
    if f'true_k_{la}' in m:m['k_true']=m[f'true_k_{la}']
    for l in (la,lb):
        if {f'true_k_{l}',f'dft_k_{l}'}<=set(m):
            m[f'ideal_delta_k_{l}']=m[f'true_k_{l}']-m[f'dft_k_{l}']; m[f'base_abs_error_k_{l}']=m[f'ideal_delta_k_{l}'].abs()
        if {f'true_log_{l}',f'dft_log_{l}'}<=set(m): m[f'ideal_delta_log_{l}']=m[f'true_log_{l}']-m[f'dft_log_{l}']
        if {f'true_k_{l}',f'pred_k_{l}'}<=set(m):
            m[f'final_abs_error_k_{l}']=(m[f'true_k_{l}']-m[f'pred_k_{l}']).abs()
            if f'base_abs_error_k_{l}' in m:
                m[f'improvement_k_{l}']=m[f'base_abs_error_k_{l}']-m[f'final_abs_error_k_{l}']; m[f'helped_{l}']=m[f'improvement_k_{l}']>0
        if f'delta_log_{l}' in m and f'ideal_delta_log_{l}' in m:
            m[f'delta_target_abs_error_log_{l}']=(m[f'ideal_delta_log_{l}']-m[f'delta_log_{l}']).abs()
            m[f'correct_delta_direction_{l}']=np.sign(m[f'ideal_delta_log_{l}'])==np.sign(m[f'delta_log_{l}'])
            m[f'saturated_{l}']=m[f'delta_log_{l}'].abs()>=0.95*bound
    if {f'dft_k_{la}',f'dft_k_{lb}'}<=set(m):
        m['k_dft_difference_b_minus_a']=m[f'dft_k_{lb}']-m[f'dft_k_{la}'];m['abs_k_dft_difference']=m.k_dft_difference_b_minus_a.abs()
    if {f'ideal_delta_log_{la}',f'ideal_delta_log_{lb}'}<=set(m):
        m['ideal_delta_log_change_b_minus_a']=m[f'ideal_delta_log_{lb}']-m[f'ideal_delta_log_{la}'];m['abs_ideal_delta_log_change']=m.ideal_delta_log_change_b_minus_a.abs()
    if {f'final_abs_error_k_{la}',f'final_abs_error_k_{lb}'}<=set(m):
        m[f'final_error_advantage_{la}_over_{lb}']=m[f'final_abs_error_k_{lb}']-m[f'final_abs_error_k_{la}']
    rows=[{'split':split,'metric':'same_row_ids','run_a':same,'run_b':same,'difference_b_minus_a':0,'detail':f'nA={len(a)} nB={len(b)}'}]
    if {f'true_k_{la}',f'true_k_{lb}'}<=set(m): rows.append({'split':split,'metric':'max_abs_target_difference','difference_b_minus_a':float((m[f'true_k_{lb}']-m[f'true_k_{la}']).abs().max())})
    if 'abs_k_dft_difference' in m:
        rows += [{'split':split,'metric':'mean_abs_k_dft_difference','difference_b_minus_a':float(m.abs_k_dft_difference.mean())},{'split':split,'metric':'max_abs_k_dft_difference','difference_b_minus_a':float(m.abs_k_dft_difference.max())}]
    for l in (la,lb):
        base=float(m[f'base_abs_error_k_{l}'].mean()) if f'base_abs_error_k_{l}' in m else np.nan
        final=float(m[f'final_abs_error_k_{l}'].mean()) if f'final_abs_error_k_{l}' in m else np.nan
        mm=mad(m[f'true_k_{l}']) if f'true_k_{l}' in m else np.nan
        for metric,val,detail in [
            ('base_linear_mae',base,'Frozen k_DFT MAE'),('final_linear_mae',final,'Final k_pred MAE'),('final_mad_mae',mm/final if final>0 else np.nan,'MAD/MAE'),
            ('help_rate_pct',100*float(m[f'helped_{l}'].mean()) if f'helped_{l}' in m else np.nan,'Rows improved by residual'),
            ('correct_delta_direction_pct',100*float(m[f'correct_delta_direction_{l}'].mean()) if f'correct_delta_direction_{l}' in m else np.nan,'Delta sign matches ideal sign'),
            ('mean_delta_target_abs_error_log',float(m[f'delta_target_abs_error_log_{l}'].mean()) if f'delta_target_abs_error_log_{l}' in m else np.nan,'Learned-vs-ideal delta'),
            ('saturation_pct',100*float(m[f'saturated_{l}'].mean()) if f'saturated_{l}' in m else np.nan,'|delta_log| >= 95% bound')]:
            rows.append({'split':split,'metric':metric,l:val,'detail':detail})
    s=pd.DataFrame(rows)
    out=[]
    for (_,metric),g in s.groupby(['split','metric'],dropna=False):
        ra={'split':split,'metric':metric,'detail':next((str(x) for x in g.get('detail',[]) if pd.notna(x)),'')}
        for l in (la,lb):
            x=pd.to_numeric(g.get(l),errors='coerce').dropna() if l in g else pd.Series(dtype=float);ra[l]=x.iloc[0] if len(x) else np.nan
        x=pd.to_numeric(g.get('difference_b_minus_a'),errors='coerce').dropna() if 'difference_b_minus_a' in g else pd.Series(dtype=float)
        ra['difference_b_minus_a']=ra[lb]-ra[la] if np.isfinite(ra[la]) and np.isfinite(ra[lb]) else (x.iloc[0] if len(x) else np.nan)
        out.append(ra)
    m.attrs['same_ids']=same;m.attrs['mapping']={'a':ma,'b':mb}
    return m,pd.DataFrame(out)

def category_summary(m,la,lb):
    adv=f'final_error_advantage_{la}_over_{lb}';rows=[]
    for c in META:
        src=f'{c}_{la}' if f'{c}_{la}' in m else f'{c}_{lb}' if f'{c}_{lb}' in m else c if c in m else None
        if not src:continue
        for val,g in m.assign(_v=m[src].fillna('<missing>').astype(str)).groupby('_v'):
            rows.append({'category':c,'value':val,'n':len(g),f'final_mae_{la}':pd.to_numeric(g.get(f'final_abs_error_k_{la}'),errors='coerce').mean(),f'final_mae_{lb}':pd.to_numeric(g.get(f'final_abs_error_k_{lb}'),errors='coerce').mean(),f'help_rate_{la}':100*pd.to_numeric(g.get(f'helped_{la}'),errors='coerce').mean(),f'help_rate_{lb}':100*pd.to_numeric(g.get(f'helped_{lb}'),errors='coerce').mean(),f'mean_advantage_{la}':pd.to_numeric(g.get(adv),errors='coerce').mean()})
    return pd.DataFrame(rows).sort_values(f'mean_advantage_{la}',ascending=False) if rows else pd.DataFrame()

def top20_overlap(pa,pb,la,lb):
    if not pa or not pb:return pd.DataFrame([{'status':'TOP20_FILE_MISSING','file_a':str(pa or ''),'file_b':str(pb or '')}])
    a,b=pd.read_csv(pa),pd.read_csv(pb);ia,ib=pick(a.columns,IDS),pick(b.columns,IDS)
    if not ia or not ib:return pd.DataFrame([{'status':'TOP20_ID_COLUMN_MISSING'}])
    aa=pd.DataFrame({'dataset_row_id':a[ia].astype(str),f'rank_{la}':range(1,len(a)+1)});bb=pd.DataFrame({'dataset_row_id':b[ib].astype(str),f'rank_{lb}':range(1,len(b)+1)})
    o=aa.merge(bb,on='dataset_row_id',how='outer',indicator=True);o['in_both_top20']=o._merge.eq('both');o['rank_shift_b_minus_a']=o[f'rank_{lb}']-o[f'rank_{la}'];o.insert(0,'overlap_count',int(o.in_both_top20.sum()));return o


def row_level_mae_gap_contributors(m: pd.DataFrame, la: str, lb: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact per-row MAE-gap attribution and concentration summary.

    Positive row_mae_gap_contribution means run B is worse than run A.
    The arithmetic mean of this column must equal MAE_B - MAE_A exactly
    (within floating-point tolerance), because MAE is the mean row error.
    """
    ea=f'final_abs_error_k_{la}'; eb=f'final_abs_error_k_{lb}'
    if ea not in m or eb not in m:
        return pd.DataFrame(), pd.DataFrame([{'status':'MISSING_FINAL_ERROR_COLUMNS'}])
    out=m.copy()
    out['row_mae_gap_contribution']=pd.to_numeric(out[eb],errors='coerce')-pd.to_numeric(out[ea],errors='coerce')
    out['abs_row_mae_gap_contribution']=out['row_mae_gap_contribution'].abs()
    out['gap_direction']=np.select(
        [out['row_mae_gap_contribution']>0,out['row_mae_gap_contribution']<0],
        [f'{lb}_worse',f'{lb}_better'],default='tie')
    out=out.sort_values('row_mae_gap_contribution',ascending=False).reset_index(drop=True)
    out.insert(0,'mae_gap_rank',np.arange(1,len(out)+1))
    n=max(len(out),1)
    total_error_sum_gap=float(out['row_mae_gap_contribution'].sum())
    mean_mae_gap=float(out['row_mae_gap_contribution'].mean())
    out['share_of_net_mae_gap_pct']=np.where(
        abs(total_error_sum_gap)>1e-15,
        100.0*out['row_mae_gap_contribution']/total_error_sum_gap,
        np.nan)
    out['cumulative_error_sum_gap']=out['row_mae_gap_contribution'].cumsum()
    out['cumulative_mae_gap']=out['cumulative_error_sum_gap']/n
    positive_total=float(out.loc[out['row_mae_gap_contribution']>0,'row_mae_gap_contribution'].sum())
    out['share_of_positive_degradation_pct']=np.where(
        (out['row_mae_gap_contribution']>0)&(positive_total>0),
        100.0*out['row_mae_gap_contribution']/positive_total,
        0.0)
    out['cumulative_positive_degradation_pct']=out['share_of_positive_degradation_pct'].cumsum()

    # Put the decisive columns first while retaining all metadata and predictions.
    lead=['mae_gap_rank','dataset_row_id','split','gap_direction',
          'row_mae_gap_contribution','abs_row_mae_gap_contribution',
          'share_of_net_mae_gap_pct','share_of_positive_degradation_pct',
          'cumulative_positive_degradation_pct','cumulative_mae_gap',
          'k_true',ea,eb,f'base_abs_error_k_{la}',f'base_abs_error_k_{lb}',
          f'ideal_delta_log_{la}',f'ideal_delta_log_{lb}',
          f'delta_log_{la}',f'delta_log_{lb}',f'helped_{la}',f'helped_{lb}']
    cols=[c for c in lead if c in out.columns]+[c for c in out.columns if c not in lead]
    out=out[cols]

    aggregate_gap=float(pd.to_numeric(m[eb],errors='coerce').mean()-pd.to_numeric(m[ea],errors='coerce').mean())
    reconcile_error=mean_mae_gap-aggregate_gap
    rows=[]
    positive=out[out['row_mae_gap_contribution']>0]
    for k in (1,3,5,10,20):
        kk=min(k,len(positive))
        captured=float(positive.head(kk)['row_mae_gap_contribution'].sum()) if kk else 0.0
        rows.append({'metric':f'top_{k}_positive_rows_share_pct','value':100.0*captured/positive_total if positive_total>0 else np.nan,'detail':f'{kk} available positive rows'})
    rows += [
        {'metric':'n_rows','value':len(out),'detail':''},
        {'metric':f'n_rows_{lb}_worse','value':int((out.row_mae_gap_contribution>0).sum()),'detail':'positive contribution'},
        {'metric':f'n_rows_{lb}_better','value':int((out.row_mae_gap_contribution<0).sum()),'detail':'negative contribution'},
        {'metric':'net_error_sum_gap','value':total_error_sum_gap,'detail':f'sum(abs_error_{lb}-abs_error_{la})'},
        {'metric':'exact_mean_mae_gap','value':mean_mae_gap,'detail':'mean row contribution'},
        {'metric':'aggregate_mae_gap','value':aggregate_gap,'detail':f'MAE_{lb}-MAE_{la}'},
        {'metric':'reconciliation_error','value':reconcile_error,'detail':'must be approximately zero'},
        {'metric':'reconciliation_status','value':'PASS' if abs(reconcile_error)<=1e-10 else 'FAIL','detail':'row contributions reconcile to aggregate MAE gap'},
        {'metric':'positive_degradation_sum','value':positive_total,'detail':f'rows where {lb} is worse'},
        {'metric':'offsetting_improvement_sum','value':float(out.loc[out.row_mae_gap_contribution<0,'row_mae_gap_contribution'].sum()),'detail':f'rows where {lb} is better'},
    ]
    return out,pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-a',required=True,type=Path);ap.add_argument('--run-b',required=True,type=Path);ap.add_argument('--label-a',default='v426');ap.add_argument('--label-b',default='v452');ap.add_argument('--output-dir',type=Path);ap.add_argument('--process-delta-bound',type=float,default=.15);a=ap.parse_args()
    na,nb=locate_normal(a.run_a),locate_normal(a.run_b);sa,sb=locate_summary(a.run_a),locate_summary(a.run_b)
    out=(a.output_dir or ((sb or nb.parent/'summary')/'checkpoint_transfer_analysis')).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    allr=[];alls=[];maps={}
    pred_paths={}
    for split in ('test','val'):
        pa,pb=find_pred(na,split),find_pred(nb,split);pred_paths[split]=(pa.resolve(),pb.resolve())
        r,s=compare(pa,pb,a.label_a,a.label_b,split,a.process_delta_bound);r.to_csv(out/f'tier3_checkpoint_transfer_row_comparison_{split}.csv',index=False);allr.append(r);alls.append(s);maps[split]=r.attrs['mapping']
    if pred_paths['test'][0]==pred_paths['val'][0] or pred_paths['test'][1]==pred_paths['val'][1]:
        raise RuntimeError(f'TEST/VAL source collision detected: {pred_paths}')
    test=allr[0];summary=pd.concat(alls,ignore_index=True);summary.to_csv(out/'tier3_checkpoint_transfer_summary.csv',index=False)
    adv=f'final_error_advantage_{a.label_a}_over_{a.label_b}';sort=adv if adv in test else 'abs_k_dft_difference';top=test.sort_values(sort,ascending=False).head(20);top.to_csv(out/'tier3_checkpoint_transfer_top_contributors.csv',index=False)
    # STAGE-2: exact row-level MAE-gap attribution as a dedicated summary category.
    gap_rows,gap_summary=row_level_mae_gap_contributors(test,a.label_a,a.label_b)
    gap_rows.to_csv(out/'tier3_checkpoint_transfer_row_mae_gap_contributors.csv',index=False)
    gap_summary.to_csv(out/'tier3_checkpoint_transfer_row_mae_gap_summary.csv',index=False)
    cat=category_summary(test,a.label_a,a.label_b);cat.to_csv(out/'tier3_checkpoint_transfer_category_summary.csv',index=False)
    t20=top20_overlap(find_top20(na,sa),find_top20(nb,sb),a.label_a,a.label_b);t20.to_csv(out/'tier3_checkpoint_transfer_top20_overlap.csv',index=False)
    def mv(name,l):
        q=summary[(summary.split=='test')&(summary.metric==name)];return float(q.iloc[0][l]) if len(q) and pd.notna(q.iloc[0][l]) else np.nan
    root=pd.DataFrame([
        {'check':'same evaluation rows','status':'PASS' if test.attrs['same_ids'] else 'FAIL','evidence':str(test.attrs['same_ids'])},
        {'check':'checkpoint-dependent frozen base','status':'CONFIRMED' if 'abs_k_dft_difference' in test and test.abs_k_dft_difference.mean()>1e-8 else 'NOT_DETECTED','evidence':f"mean_abs_k_DFT_diff={test.get('abs_k_dft_difference',pd.Series([np.nan])).mean():.6g}"},
        {'check':'residual help rate','status':'COMPARE','evidence':f"{a.label_a}={mv('help_rate_pct',a.label_a):.2f}% {a.label_b}={mv('help_rate_pct',a.label_b):.2f}%"},
        {'check':'delta direction','status':'COMPARE','evidence':f"{a.label_a}={mv('correct_delta_direction_pct',a.label_a):.2f}% {a.label_b}={mv('correct_delta_direction_pct',a.label_b):.2f}%"},
        {'check':'delta magnitude fit','status':'COMPARE','evidence':f"{a.label_a}={mv('mean_delta_target_abs_error_log',a.label_a):.6g} {a.label_b}={mv('mean_delta_target_abs_error_log',a.label_b):.6g}"},
        {'check':'final transfer quality','status':'BETTER_'+(a.label_a if mv('final_linear_mae',a.label_a)<mv('final_linear_mae',a.label_b) else a.label_b),'evidence':f"MAE {a.label_a}={mv('final_linear_mae',a.label_a):.6g} {a.label_b}={mv('final_linear_mae',a.label_b):.6g}"},
    ]);root.to_csv(out/'tier3_checkpoint_transfer_root_cause.csv',index=False)
    with pd.ExcelWriter(out/'tier3_checkpoint_transfer_analysis.xlsx',engine='openpyxl') as w:
        root.to_excel(w,'Root_Cause',index=False);summary.to_excel(w,'Metric_Summary',index=False);gap_summary.to_excel(w,'MAE_Gap_Summary',index=False);gap_rows.to_excel(w,'Row_MAE_Gap_Contrib',index=False);top.to_excel(w,'Top_Contributors',index=False);cat.to_excel(w,'Category_Summary',index=False);t20.to_excel(w,'Top20_Overlap',index=False);test.to_excel(w,'TEST_Row_Comparison',index=False);allr[1].to_excel(w,'VAL_Row_Comparison',index=False)
    md=['# Tier-3 Checkpoint Transfer Analysis','',f'**{a.label_a} vs {a.label_b}**','','Rows are aligned by `dataset_row_id`; CSV order is ignored.','','## Root-cause table','',root.to_markdown(index=False),'','## Metrics','',summary.to_markdown(index=False),'','## Exact Row-Level MAE Gap Contributors','',gap_summary.to_markdown(index=False),'',gap_rows.head(20).to_markdown(index=False),'','## Interpretation','','- Non-zero `k_DFT` differences confirm checkpoint-dependent frozen base predictions.','- This changes the ideal Tier-3 residual target `k_true - k_DFT`.','- Help rate, delta-direction accuracy, delta-fit error and saturation identify why transfer differs.','- Top-contributor and category sheets identify the exact rows and process groups causing the gap.','','## Detected columns','','```json',json.dumps(maps,indent=2),'```']
    (out/'tier3_checkpoint_transfer_analysis.md').write_text('\n'.join(md))
    print(f'Wrote transfer analysis to {out}')
if __name__=='__main__':main()
